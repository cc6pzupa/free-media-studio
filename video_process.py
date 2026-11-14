"""Persistent video worker in a separate Python environment on Kaggle."""
import hashlib
import json
import os
from pathlib import Path
import select
import signal
import subprocess
import sys
import time
import uuid


class VideoProcess:
    def __init__(self, model_id, acceleration='auto', progress=None, cache_handle=None,
                 cache_token=None, install=True, environment='/kaggle/temp/cc-media-video-environment'):
        from video_registry import video_profile
        self.profile=video_profile(model_id);self.progress=progress or (lambda event:None)
        self.acceleration=acceleration;self.process=None;self.buffer=b''
        requirements=Path(__file__).with_name('requirements-video-remote.txt')
        environment=Path(environment);python=environment/'bin/python'
        if install:
            if not python.exists():
                subprocess.run([sys.executable,'-m','venv','--without-pip','--system-site-packages',str(environment)],check=True)
            digest=hashlib.sha256(requirements.read_bytes()).hexdigest()
            stamp=environment/'cc-requirements.sha256'
            if not stamp.exists() or stamp.read_text()!=digest:
                self.progress({'phase':'installing'})
                subprocess.run([str(python),'-m','pip','install','-q','-r',str(requirements)],check=True,timeout=1200)
                stamp.write_text(digest)
        elif not python.exists():python=Path(sys.executable)
        env=dict(os.environ,USE_TF='0',USE_FLAX='0',TOKENIZERS_PARALLELISM='false',CC_VIDEO_ACTOR='1')
        self.process=subprocess.Popen([str(python),'-u',str(Path(__file__).resolve()),'--worker'],
            stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=None,start_new_session=True,env=env)
        try:self.request('configure',model_id=model_id,acceleration=acceleration,
                         cache_handle=cache_handle,cache_token=cache_token)
        except BaseException:self.close();raise

    def alive(self):return self.process is not None and self.process.poll() is None

    def request(self,action,**params):
        if not self.alive():raise RuntimeError('Video process stopped')
        request_id=uuid.uuid4().hex
        self.process.stdin.write((json.dumps(dict(id=request_id,action=action,**params))+'\n').encode())
        self.process.stdin.flush();deadline=time.monotonic()+14400
        while time.monotonic()<deadline:
            if b'\n' not in self.buffer:
                if not select.select([self.process.stdout],[],[],min(5,max(0,deadline-time.monotonic())))[0]:
                    if not self.alive():raise RuntimeError('Video process exited')
                    continue
                chunk=os.read(self.process.stdout.fileno(),65536)
                if not chunk:raise RuntimeError('Video process output closed')
                self.buffer+=chunk
                if len(self.buffer)>2**20:raise RuntimeError('Video response exceeded limit')
                continue
            line,self.buffer=self.buffer.split(b'\n',1)
            if not line.startswith(b'CC_VIDEO_PROCESS:'):continue
            message=json.loads(line.split(b':',1)[1])
            if message.get('id')!=request_id:raise RuntimeError('Video response mismatch')
            if 'progress' in message:self.progress(message['progress']);continue
            if 'error' in message:raise RuntimeError(message['error'])
            return message.get('result')
        self.close();raise TimeoutError('Video request timed out')

    def generate(self,prompt,seed=42,steps=None,frames=None,output=None,image=None,negative_prompt=None):
        # The first frame travels as base64 on the worker's JSON stdin line.
        extra={k:v for k,v in (('image',image),('negative_prompt',negative_prompt)) if v is not None}
        result=self.request('generate',prompt=prompt,seed=seed,steps=steps,frames=frames,
                            output=output,acceleration=self.acceleration,**extra)
        return Path(result['path']),result['stats']

    def close(self):
        process=self.process;self.process=None
        if process is None:return
        try:
            if process.poll() is None:
                process.stdin.write((json.dumps({'id':uuid.uuid4().hex,'action':'close'})+'\n').encode());process.stdin.flush()
                try:process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid,signal.SIGTERM)
                    try:process.wait(timeout=10)
                    except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=10)
        except (BrokenPipeError,ProcessLookupError):pass
        finally:
            process.stdin.close();process.stdout.close()


def worker():
    import contextlib
    import traceback
    wire=sys.stdout;runtime=None;request_id=None
    def send(**data):
        wire.write('CC_VIDEO_PROCESS:'+json.dumps(dict(id=request_id,**data))+'\n');wire.flush()
    try:
        for line in sys.stdin:
            command=json.loads(line);request_id=command.pop('id');action=command.pop('action')
            try:
                if action=='close':break
                with contextlib.redirect_stdout(sys.stderr):
                    if action=='configure':
                        from video_runtime import VideoRuntime
                        runtime=VideoRuntime(**command,progress=lambda event:send(progress=event)).prepare()
                        result={'ready':True}
                    elif action=='generate' and runtime is not None:
                        runtime.acceleration=command.pop('acceleration')
                        path,stats=runtime.generate(**command);result={'path':str(path),'stats':stats}
                    else:raise ValueError('Unknown video command')
                send(result=result)
            except Exception as error:
                traceback.print_exc(file=sys.stderr)
                # Include wrapped causes: libraries often re-raise a generic error.
                parts,current=[],error
                while current is not None and len(parts)<6:
                    parts.append(type(current).__name__+': '+str(current)[:300]);current=current.__cause__ or current.__context__
                send(error=' <- '.join(parts))
    finally:
        if runtime is not None:
            with contextlib.redirect_stdout(sys.stderr):runtime.close()


if __name__=='__main__' and '--worker' in sys.argv:worker()
