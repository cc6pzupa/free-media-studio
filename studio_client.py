"""Authenticated localhost client and GUI/ComfyUI backend adapter."""
import base64
import json
import os
import secrets
import subprocess
import sys
import threading
import time
from pathlib import Path
from urllib.parse import urlsplit

import requests

if __package__:
    from .studio_core import config_path,load_config,normalize_config,state_directory
else:
    from studio_core import config_path,load_config,normalize_config,state_directory


class StudioHTTPError(RuntimeError):
    def __init__(self,status_code,message):
        super().__init__(message);self.status_code=status_code


class StudioClient:
    def __init__(self,directory=None,autostart=True):
        self.directory=Path(directory or state_directory())
        self.autostart=autostart
        self.http=requests.Session()
        self.http.trust_env=False  # Never send the local bearer token through a proxy.
        self.url=None;self.token=None;self.model_selection_supported=False;self.acceleration_supported=False;self.video_supported=False
        self.connect_lock=threading.Lock()

    def _read_connection(self):
        try:
            data=json.loads((self.directory/'connection.json').read_text(encoding='utf-8'))
            url=urlsplit(data['url'])
            if (url.scheme!='http' or url.hostname!='127.0.0.1' or not url.port
                or url.path or url.query or url.fragment or url.username or url.password
                or not isinstance(data['token'],str) or len(data['token'])<24):return None
            return data
        except (OSError,ValueError,KeyError,TypeError):return None

    def _probe(self,data):
        if not data:return False
        try:
            response=self.http.get(data['url']+'/health',headers={'Authorization':'Bearer '+data['token']},timeout=1)
            result=response.json()
            if response.ok and result.get('app')=='qwen-studio-worker' and result.get('protocol')==1:
                self.url,self.token=data['url'],data['token']
                self.model_selection_supported=result.get('model_selection') is True
                self.acceleration_supported=result.get('dual_cfg') is True
                self.video_supported=result.get('video') is True
                return True
        except (requests.RequestException,ValueError):pass
        return False

    def ensure(self,force=False):
        if self.url and not force:return
        with self.connect_lock:
            if self._probe(self._read_connection()):return
            if not self.autostart:raise RuntimeError('本机后台未运行。')
            self.directory.mkdir(parents=True,exist_ok=True,mode=0o700)
            log_path=self.directory/'worker.log'
            fd=os.open(log_path,os.O_CREAT|os.O_APPEND|os.O_WRONLY,0o600)
            with os.fdopen(fd,'ab') as output:
                options={'creationflags':subprocess.CREATE_NEW_PROCESS_GROUP|subprocess.DETACHED_PROCESS|subprocess.CREATE_BREAKAWAY_FROM_JOB} if os.name=='nt' else {'start_new_session':True}
                environment=dict(os.environ,EASY_CONFIG_PATH=str(config_path()),PYTHONUTF8='1')
                command=[sys.executable,str(Path(__file__).with_name('studio_worker.py'))]
                if getattr(sys,'frozen',False):
                    worker_exe=Path(sys.executable).with_name('cc-media-cli'+('.exe' if os.name=='nt' else ''))
                    command=[str(worker_exe),'--worker']
                    environment['PYINSTALLER_RESET_ENVIRONMENT']='1'
                try:
                    subprocess.Popen([*command,'--state-dir',str(self.directory.resolve())],stdin=subprocess.DEVNULL,stdout=output,stderr=output,
                        cwd=str(Path(__file__).resolve().parent),env=environment,close_fds=True,**options)
                except OSError:
                    raise RuntimeError('无法启动独立后台。请先在终端运行 python studio_worker.py，再连接 MCP。 / Start python studio_worker.py in a terminal before connecting MCP.') from None
            deadline=time.monotonic()+20
            while time.monotonic()<deadline:
                if self._probe(self._read_connection()):return
                time.sleep(.1)
            raise RuntimeError('本机后台未能启动。请检查依赖和本机配置，或先停止旧版后台再重试。')

    def _request(self,method,path,body=None,binary=False):
        self.ensure()
        if method=='POST' and path in ('/configure','/jobs') and not self.acceleration_supported:
            raise RuntimeError('请先停止旧后台并重启客户端以启用双卡加速。 / Restart the old worker to enable dual-GPU acceleration.')
        for attempt in range(2):
            try:
                response=self.http.request(method,self.url+path,json=body,
                    headers={'Authorization':'Bearer '+self.token},timeout=(2,15))
            except requests.RequestException:
                if attempt==0 and self.autostart:
                    self.ensure(force=True);continue
                raise RuntimeError('本机后台连接中断。任务可能已提交，请使用同一 request_id 或原任务 ID 检查。') from None
            if not response.ok:
                try:message=response.json().get('error','本机后台请求失败。')
                except ValueError:message='本机后台请求失败。'
                raise StudioHTTPError(response.status_code,message)
            return response.content if binary else response.json()
        raise RuntimeError('本机后台不可用。')

    def configure(self,config,remember=None,api_key=None):
        normalized=normalize_config(config)
        body={'config':normalized,'remember':remember,'api_key':api_key} if remember is not None else normalized
        return self._request('POST','/configure',body)

    def status(self):
        result=self._request('GET','/status')
        if not result['configured']:
            config=load_config()
            if config:result=self.configure(config)
        return result

    def quota(self):return self._request('GET','/quota')

    def history(self,query_string=''):
        return self._request('GET','/history'+('?' + query_string if query_string else ''))

    def edit_history(self,job_id,body):
        self.get_job(job_id)
        return self._request('POST','/history/'+job_id,body)

    def submit(self,prompt,seed=42,steps=None,request_id=None,model_id=None):
        self.status()
        if model_id not in (None,'qwen-image-2.1') and not self.model_selection_supported:
            raise RuntimeError('请先停止旧后台并重启客户端，再使用新增模型。 / Restart the old worker to select new models.')
        # A retried POST in _request uses this exact same ID and payload.
        return self._request('POST','/jobs',{'prompt':prompt,'seed':seed,'steps':steps,
            'request_id':request_id or secrets.token_hex(16),'model_id':model_id})

    def get_job(self,job_id):
        # Validate before assembling a URL; never accept traversal/query fragments.
        import re
        if not isinstance(job_id,str) or not re.fullmatch(r'[a-f0-9]{32}',job_id):
            raise ValueError('任务 ID 格式不正确。')
        return self._request('GET','/jobs/'+job_id)

    def get_image(self,job_id):
        self.get_job(job_id)
        return self._request('GET','/jobs/'+job_id+'/image',binary=True)

    def submit_video(self,prompt,seed=42,steps=None,request_id=None,model_id='wan21-t2v-1.3b',frames=None):
        if __package__:
            from .video_registry import validate_video
        else:
            from video_registry import validate_video
        params=validate_video(prompt,seed,steps,model_id,frames)
        self.status()
        if not self.video_supported:raise RuntimeError('请重启本机后台以启用视频 / Restart the worker to enable video.')
        return self._request('POST','/video/jobs',dict(params,request_id=request_id or secrets.token_hex(16)))

    def get_video(self,job_id):
        self.get_job(job_id)
        return self._request('GET','/jobs/'+job_id+'/video',binary=True)

    def wait(self,job_id,progress=None,timeout=14400):
        deadline=time.monotonic()+timeout
        previous=None
        while time.monotonic()<deadline:
            job=self.get_job(job_id)
            if progress and job!=previous:progress(job)
            previous=job
            if job['state']=='complete':return (self.get_video(job_id) if job.get('media_type')=='video' else self.get_image(job_id)),job['usage']
            if job['state'] in ('error','interrupted'):raise RuntimeError(job['message'])
            time.sleep(.5)
        raise TimeoutError('等待超时；任务未取消，请使用原任务 ID 继续查询。')

    def shutdown(self):return self._request('POST','/shutdown',{})
    def close(self):self.http.close()  # Never stop the worker or the Kaggle kernel.


class SharedBackend:
    """Same backend contract, backed by the one durable local worker."""
    def __init__(self,config,client=None,activate=True):
        self.client=client or StudioClient()
        normalized=normalize_config(config)
        self.config=normalized
        self.mode=normalized['mode'];self.user=normalized['user'];self.dataset=normalized['dataset']
        self.token=normalized.get('token','');self.jupyter_url=normalized.get('jupyter_url','')
        self.kernel_id=normalized.get('kernel_id')
        self.acceleration=normalized['acceleration']
        self.keep_alive_minutes=normalized['keep_alive_minutes']
        if activate:self.activate()

    def activate(self,remember=None,api_key=None):
        self.client.configure(self.config,remember,api_key)

    @classmethod
    def from_running(cls,client,status):
        result=cls.__new__(cls)
        result.client=client;result.mode=status['mode'];result.user=status['user'];result.dataset=status['dataset']
        result.token='';result.jupyter_url='';result.kernel_id=None;result.config=None
        result.acceleration=status.get('acceleration','auto')
        result.keep_alive_minutes=status.get('keep_alive_minutes',15)
        result.client.autostart=True
        return result

    def submit(self,prompt,seed=42,steps=None,request_id=None,model_id=None):
        if self.config:
            config={k:v for k,v in self.config.items() if k!='keep_alive_minutes'}
            self.client._request('POST','/configure',config)
        return self.client.submit(prompt,seed,steps,request_id,model_id)
    def generate(self,prompt,seed=42,steps=None,model_id=None):return self.generate_with_progress(prompt,seed,steps,model_id=model_id)
    def generate_with_progress(self,prompt,seed=42,steps=None,progress=None,model_id=None):
        return self.client.wait(self.submit(prompt,seed,steps,model_id=model_id)['job_id'],progress)
    def close(self):self.client.close()
