"""Account/key-only provisioning through Kaggle's public long-running API."""
import json
import os
import re
import threading
import time
from pathlib import Path
from urllib.parse import urlsplit
import requests
if __package__:
    from .session_backend import SessionBackend
else:
    from session_backend import SessionBackend


class KaggleApiError(RuntimeError):
    def __init__(self, status):
        self.status=status
        super().__init__(f'Kaggle API 操作失败（HTTP {status}）。请检查账号权限、手机号验证和 GPU 额度。')


class AccountBackend:
    mode='account'

    def __init__(self,user,token,dataset=None,notebook_slug='qwen-image-studio',cache_dir=None,http=None):
        if not re.fullmatch(r'[A-Za-z0-9_-]+',user):raise ValueError('请填写 Kaggle 账号用户名')
        if not isinstance(token,str) or not token.strip():raise ValueError('请填写 Kaggle API Key / Token')
        if not re.fullmatch(r'[A-Za-z0-9_-]+',notebook_slug):raise ValueError('Invalid notebook slug')
        self.user,self.token=user,token.strip()
        self.dataset=dataset or user+'/qwen-image-t4-assets'
        self.slug=user+'/'+notebook_slug
        self.http=http or requests.Session()
        if self.token.startswith('KGAT_'):self.http.headers['Authorization']='Bearer '+self.token
        else:self.http.auth=(self.user,self.token)
        self.http.headers.update({'Content-Type':'application/json','User-Agent':'qwen-image-studio/1'})
        directory=Path(cache_dir) if cache_dir else Path(os.environ.get('EASY_CONFIG_PATH',str(Path.home()/'.kaggle-qwen-client/settings.json'))).parent/'sessions'
        self.cache_file=directory/(user+'-'+notebook_slug+'.json')
        self.backend=None;self.lock=threading.Lock()

    def _call(self,service,method,body):
        try:
            response=self.http.post('https://api.kaggle.com/v1/'+service+'/'+method,json=body,timeout=(20,90))
        except requests.RequestException:
            raise RuntimeError('无法连接 Kaggle，请检查网络；没有自动重复提交。') from None
        if not response.ok:raise KaggleApiError(response.status_code)
        data=response.json()
        if isinstance(data,dict) and isinstance(data.get('code'),int) and data['code']>=400:raise KaggleApiError(data['code'])
        return data

    def _read(self):
        try:return json.loads(self.cache_file.read_text(encoding='utf-8'))
        except (OSError,ValueError):return {}

    def _save(self,data):
        self.cache_file.parent.mkdir(mode=0o700,parents=True,exist_ok=True)
        temporary=self.cache_file.with_suffix('.tmp')
        fd=os.open(temporary,os.O_CREAT|os.O_TRUNC|os.O_WRONLY,0o600)
        with os.fdopen(fd,'w',encoding='utf-8') as f:json.dump(data,f)
        temporary.replace(self.cache_file);os.chmod(self.cache_file,0o600)

    def _ensure_notebook(self):
        try:
            existing=self._call('kernels.KernelsApiService','GetKernel',{'userName':self.user,'kernelSlug':self.slug.split('/')[1]})
            if 'QWEN_STUDIO_MANAGED' not in existing.get('blob',{}).get('source',''):
                raise RuntimeError('同名 Notebook 已存在且不属于本程序。请先为它改名，避免覆盖你的内容。')
            try:
                notebook=json.loads(existing['blob']['source'])
                operation=notebook.get('metadata',{}).get('qwen_studio',{}).get('operation')
                if isinstance(operation,str) and operation.startswith('operations/create-kernel-session/'):
                    return operation
            except (ValueError,TypeError,AttributeError):pass
            return None
        except KaggleApiError as error:
            if error.status not in (403,404):raise
            # Kaggle can hide a missing private slug behind HTTP 403. Confirm
            # absence in the authenticated listing before creating anything.
            listing=self._call('kernels.KernelsApiService','ListKernels',{'user':self.user,'search':self.slug.split('/')[1],'group':'PUBLIC_AND_USERS_PRIVATE','pageSize':100})
            if listing.get('nextPageToken') or any(k.get('ref')==self.slug or k.get('slug')==self.slug.split('/')[1] for k in listing.get('kernels',[])):
                raise RuntimeError('无法安全确认专用 Notebook 不存在，没有覆盖任何内容。') from None
        notebook={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'}},'cells':[
            {'id':'studio-info','cell_type':'markdown','metadata':{},'source':['# Qwen Image Studio\nQWEN_STUDIO_MANAGED\n模型首次从 Hugging Face 自动下载；客户端连接后按需加载。']},
            {'id':'studio-ready','cell_type':'code','metadata':{},'execution_count':None,'outputs':[],'source':["print('Qwen Image Studio session ready')"]}]}
        data=self._call('kernels.KernelsApiService','SaveKernel',{'slug':self.slug,'newTitle':self.slug.split('/')[1],'text':json.dumps(notebook,ensure_ascii=False),'language':'python','kernelType':'notebook','isPrivate':True,'enableInternet':True,'enableGpu':False,'datasetDataSources':[],'kernelExecutionType':'QUICK_SAVE'})
        if data.get('error') or data.get('invalidDatasetSources'):raise RuntimeError('无法建立专用 Notebook，请检查 Kaggle 账号设置。')

    def _remember_operation(self,operation):
        # A task identifier is not a connection credential. Keep it in the
        # private, program-managed Notebook so a new client can authenticate
        # through GetOperation and recover the living session without a URL.
        existing=self._call('kernels.KernelsApiService','GetKernel',{'userName':self.user,'kernelSlug':self.slug.split('/')[1]})
        raw=existing.get('blob',{}).get('source','')
        if 'QWEN_STUDIO_MANAGED' not in raw:raise RuntimeError('专用 Notebook 标记已改变，没有修改内容。')
        notebook=json.loads(raw)
        notebook.setdefault('metadata',{})['qwen_studio']={'operation':operation}
        result=self._call('kernels.KernelsApiService','SaveKernel',{'slug':self.slug,
            'text':json.dumps(notebook,ensure_ascii=False),'language':'python','kernelType':'notebook',
            'isPrivate':True,'enableInternet':True,'enableGpu':False,'datasetDataSources':[],
            'kernelExecutionType':'QUICK_SAVE'})
        if result.get('error') or result.get('invalidDatasetSources'):raise RuntimeError('无法保存启动任务记录。')

    def _recover_operation(self,operation):
        try:result=self._call('common.OperationsService','GetOperation',{'name':operation})
        except KaggleApiError as error:
            if error.status==404:return None
            raise
        if not result.get('done'):return operation
        if result.get('error'):return None
        url=(result.get('response') or {}).get('jupyterUrl','')
        if not self._valid_connection(url):return None
        try:probe=requests.get(url.rstrip('/')+'/api/kernels',timeout=(10,20))
        except requests.RequestException:
            raise RuntimeError('无法检查 Notebook 中记录的会话，请稍后重试；没有重复申请 GPU。') from None
        if probe.ok:return operation
        if probe.status_code in (403,404,410):return None
        raise RuntimeError('现有 Kaggle 会话暂时无法连接；没有重复申请 GPU。')

    @staticmethod
    def _valid_connection(url):
        parsed=urlsplit(url)
        return parsed.scheme=='https' and parsed.hostname and parsed.hostname.endswith('.jupyter-proxy.kaggle.net') and parsed.path.rstrip('/').endswith('/proxy')

    def _ensure_session(self,progress):
        cached=self._read();url=cached.get('jupyter_url','');expired_operation=None
        if url and self._valid_connection(url):
            try:
                probe=requests.get(url.rstrip('/')+'/api/kernels',timeout=(10,20))
                if probe.ok:
                    progress({'phase':'connecting','message':'已找到运行中的 Kaggle 会话，正在接回模型。'})
                    return url
                if probe.status_code in (403,404,410):
                    expired_operation=cached.get('operation')
                    cached={}
                    self._save({})
                else:
                    raise RuntimeError('现有 Kaggle 会话暂时无法连接，请稍后重试；没有重复申请 GPU。')
            except requests.RequestException:
                raise RuntimeError('无法检查现有 Kaggle 会话，请检查网络后重试；没有重复申请 GPU。') from None
        progress({'phase':'checking','message':'正在准备你的专用 Kaggle Notebook。'})
        recorded_operation=self._ensure_notebook()
        operation=cached.get('operation')
        if not operation and isinstance(recorded_operation,str) and recorded_operation!=expired_operation:
            progress({'phase':'connecting','message':'正在找回专用 Notebook 中记录的运行会话。'})
            operation=self._recover_operation(recorded_operation)
            if operation:self._save({'operation':operation})
        if not operation:
            progress({'phase':'starting','message':'正在申请双 T4 会话，首次启动可能需要排队。'})
            result=self._call('kernels.KernelsApiService','CreateKernelSession',{'slug':self.slug,'language':'python','kernelType':'notebook','machineShape':'NvidiaTeslaT4','enableInternet':True})
            operation=result.get('name')
            if not operation:raise RuntimeError('Kaggle 未返回启动任务编号。')
            self._save({'operation':operation})
        deadline=time.monotonic()+1200
        while time.monotonic()<deadline:
            result=self._call('common.OperationsService','GetOperation',{'name':operation})
            if result.get('done'):
                if result.get('error'):
                    self._save({})
                    raise RuntimeError('Kaggle 会话启动失败，请检查 GPU 额度和账号验证。')
                payload=result.get('response') or {}
                # jupyterUrl is the direct Jupyter API/proxy URL. The separate
                # tokenizedJupyterUrl is a frontend handoff URL, not an API base.
                url=payload.get('jupyterUrl') or ''
                if not self._valid_connection(url):raise RuntimeError('Kaggle 未返回有效的会话连接。')
                self._save({'jupyter_url':url,'operation':operation})
                if recorded_operation!=operation:
                    try:self._remember_operation(operation)
                    except Exception:
                        progress({'phase':'session-warning','message':'启动记录已保存在本机；远端记录暂未保存，请保留本机配置。'})
                progress({'phase':'connected','message':'会话已启动，正在检查模型文件。'})
                return url
            progress({'phase':'queued','message':'Kaggle 正在排队或准备 GPU 环境。'})
            time.sleep(10)
        raise TimeoutError('Kaggle 启动等待超时。再次请求会继续检查同一个启动任务，不会重复申请。')

    def generate(self,prompt,seed=42,steps=20):
        return self.generate_with_progress(prompt,seed,steps)

    def generate_with_progress(self,prompt,seed=42,steps=20,progress=None):
        notify=progress or (lambda update:None)
        with self.lock:
            started=time.monotonic()
            if self.backend is None or self.backend.session.transport is None:
                url=self._ensure_session(notify)
                self.backend=SessionBackend(self.user,self.dataset,url,asset_mode='auto',
                    cache_handle=self.user+'/qwen-image-studio-assets',api_token=self.token)
            image,stats=self.backend.generate_with_progress(prompt,seed,steps,notify)
            stats.update(backend='account',request_seconds=round(time.monotonic()-started,2))
            return image,stats

    def close(self):
        with self.lock:
            if self.backend:self.backend.close()
            self.http.close()
