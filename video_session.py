"""Video session adapter. Keeps MP4 separate from the existing PNG protocol."""
import base64
import json
import threading
from pathlib import Path
from session_runtime import JupyterTransport,ModelSession
from session_backend import RELEASE_FAILURE_CODE
from video_registry import video_profile,validate_video

VIDEO_FILES=('asset_manager.py','cache_manager.py','video_precision.py','video_registry.py','video_assets.py',
             'video_parallel.py','video_attention.py','video_schedule.py','video_context.py','video_runtime.py','video_process.py','requirements-video-remote.txt')


def video_bootstrap_code(model_id,acceleration='auto',install=True,cache_handle=None,api_token=None):
    video_profile(model_id)
    if acceleration not in ('auto','split'):raise ValueError('Invalid acceleration')
    sources={name:Path(__file__).with_name(name).read_text(encoding='utf-8') for name in VIDEO_FILES}
    return f'''
import sys,json,gc,hashlib
from pathlib import Path
_cc_video_modules=Path('/kaggle/temp/cc-media-video-runtime')
_cc_video_modules.mkdir(parents=True,exist_ok=True)
_cc_video_sources={sources!r}
_cc_video_hash=hashlib.sha256(json.dumps(_cc_video_sources,sort_keys=True).encode()).hexdigest()
_cc_existing_video=globals().get('_CC_VIDEO_PROCESS')
if _cc_existing_video is not None and (not _cc_existing_video.alive() or _cc_existing_video.profile['id']!={model_id!r} or globals().get('_CC_VIDEO_SOURCE_HASH')!=_cc_video_hash):
    _cc_existing_video.close();globals().pop('_CC_VIDEO_PROCESS',None)
if globals().get('_CC_VIDEO_PROCESS') is None:
    if (globals().get('pipe') is not None or globals().get('encoder') is not None) and not (globals().get('_CC_MANAGED_MODEL') or globals().get('_COMFY_QWEN_STATE')):
        raise RuntimeError('Unmanaged model exists; select a separate video notebook')
    if globals().get('_CC_VIDEO_RUNTIME') is not None:globals()['_CC_VIDEO_RUNTIME'].close()
    for _name in ('pipe','encoder','transformer','vae','processor','scheduler','_COMFY_QWEN_STATE','_CC_DUAL_STATE','_CC_NEGATIVE_CACHE','_CC_PROMPT_CACHE','_CC_TASK_PIPES','_value','_module','_decoded','generate_image','_CC_VIDEO_RUNTIME','_cc_existing_video'):
        globals().pop(_name,None)
    # An earlier failed image cell can still pin the image model through its traceback.
    exec(compile({RELEASE_FAILURE_CODE!r},'<cc-release-failure>','exec'),globals())
    gc.collect()
    if 'torch' in sys.modules:sys.modules['torch'].cuda.empty_cache()
    for _name,_source in _cc_video_sources.items():
        (_cc_video_modules/_name).write_text(_source,encoding='utf-8')
    if str(_cc_video_modules) not in sys.path:sys.path.insert(0,str(_cc_video_modules))
    sys.modules.pop('video_process',None)
    from video_process import VideoProcess
    def _cc_video_progress(event):print('COMFY_PROGRESS:'+json.dumps(event),flush=True)
    _CC_VIDEO_PROCESS=VideoProcess({model_id!r},acceleration={acceleration!r},progress=_cc_video_progress,
        cache_handle={cache_handle!r},cache_token={api_token!r},install={install!r})
    _CC_VIDEO_SOURCE_HASH=_cc_video_hash
_CC_VIDEO_PROCESS.acceleration={acceleration!r}
_CC_MODEL_ID={model_id!r};_CC_MANAGED_MODEL=True
print('CC_VIDEO_READY',flush=True)
'''



def video_generation_code(prompt,seed=42,steps=None,model_id='wan21-t2v-1.3b',frames=None,image=None,negative_prompt=None):
    if image is not None and not isinstance(image,str):image=base64.b64encode(image).decode('ascii')
    args=validate_video(prompt,seed,steps,model_id,frames,image,negative_prompt)
    return f'''
import json,base64
assert globals().get('_CC_MODEL_ID')=={model_id!r},'Active model changed'
_cc_video_args=json.loads({json.dumps(json.dumps(args))})
_cc_video_args.pop('model_id')
_cc_mp4,_cc_video_stats=_CC_VIDEO_PROCESS.generate(**_cc_video_args)
print('CC_VIDEO_RESULT:'+json.dumps({{'mp4':base64.b64encode(_cc_mp4.read_bytes()).decode('ascii'),'stats':_cc_video_stats}}),flush=True)
'''


class VideoSessionBackend:
    def __init__(self,url,model_id,acceleration='auto',transport_factory=None,cache_handle=None,api_token=None):
        self.model_id=video_profile(model_id)['id'];self.lock=threading.Lock();self.progress=None;self.buffer=''
        self.session=ModelSession(transport_factory or (lambda:JupyterTransport(url,output_callback=self._output)),
            video_bootstrap_code(model_id,acceleration,cache_handle=cache_handle,api_token=api_token),lifetime_seconds=None)

    def _output(self,text):
        if not self.progress:return
        self.buffer+=text
        lines=self.buffer.split('\n');self.buffer=lines.pop()
        for line in lines:
            if line.startswith('COMFY_PROGRESS:'):
                try:self.progress(dict(json.loads(line.split(':',1)[1]),media_type='video'))
                except (ValueError,TypeError):pass
        if len(self.buffer)>2*1024*1024:self.buffer=''

    def generate_with_progress(self,prompt,seed=42,steps=None,model_id=None,frames=None,progress=None,image=None,negative_prompt=None):
        if model_id is not None and model_id!=self.model_id:raise ValueError('Wrong video model')
        with self.lock:
            self.progress=progress
            try:
                result=self.session.execute(video_generation_code(prompt,seed,steps,self.model_id,frames,image,negative_prompt),timeout=14400)
                lines=[line.split(':',1)[1] for line in result.splitlines() if line.startswith('CC_VIDEO_RESULT:')]
                if len(lines)!=1:raise RuntimeError('Missing video result')
                data=json.loads(lines[0]);raw=base64.b64decode(data['mp4'],validate=True)
                if len(raw)<12 or raw[4:8]!=b'ftyp':raise RuntimeError('Invalid MP4')
                return raw,data['stats']
            except Exception:
                raise RuntimeError('视频生成失败，请查看专用 Notebook 日志；该任务没有自动重发。') from None
            finally:self.progress=None;self.buffer=''

    def close(self):self.session.close()
