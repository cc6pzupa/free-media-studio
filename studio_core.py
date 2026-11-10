"""Durable, single-consumer job queue shared by all local entry points."""
import hashlib
import json
import math
import os
import re
import secrets
import sqlite3
import struct
import threading
import time
from pathlib import Path


def config_path():
    return Path(os.environ.get('EASY_CONFIG_PATH', str(Path.home()/'.kaggle-qwen-client/settings.json'))).expanduser().resolve()


def state_directory():
    return config_path().parent/'studio'


def private_write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_name(path.name+'.'+secrets.token_hex(6)+'.tmp')
    try:
        fd = os.open(temporary, os.O_CREAT|os.O_EXCL|os.O_WRONLY, 0o600)
        with os.fdopen(fd, 'wb') as output:
            output.write(data)
        temporary.replace(path)
        os.chmod(path, 0o600)
    finally:
        temporary.unlink(missing_ok=True)


def keep_alive_value(value):
    if isinstance(value,bool) or str(value).strip() not in {str(n) for n in range(1441)}:
        raise ValueError('保活时间需要为 0–1440 分钟的整数。')
    return int(value)


def normalize_config(config):
    mode = config.get('mode', 'account')
    user = config.get('user', '').strip()
    if not re.fullmatch(r'[A-Za-z0-9_-]+', user):
        raise ValueError('请配置 Kaggle 用户名。')
    result = {'mode':mode, 'user':user, 'dataset':config.get('dataset', '').strip(),
              'keep_alive_minutes':keep_alive_value(config.get('keep_alive_minutes',os.environ.get('CC_MEDIA_KEEP_ALIVE_MINUTES','15')))}
    result['acceleration']=config.get('acceleration',os.environ.get('CC_MEDIA_ACCELERATION','auto'))
    if result['acceleration'] not in ('auto','split'):raise ValueError('加速模式请选择 auto 或 split。')
    if mode == 'session':
        result.update(jupyter_url=config.get('jupyter_url', '').strip(), kernel_id=config.get('kernel_id') or None)
        if __package__:
            from .session_runtime import JupyterTransport
        else:
            from session_runtime import JupyterTransport
        transport = JupyterTransport(result['jupyter_url'])
        transport.close()
    elif mode in ('account', 'batch'):
        token = config.get('token', '').strip()
        if not token:raise ValueError('请配置 Kaggle API Key。')
        result['token'] = token
    else:
        raise ValueError('不支持的运行模式。')
    if mode != 'account' and not re.fullmatch(r'[A-Za-z0-9_-]+/[A-Za-z0-9_-]+', result['dataset']):
        raise ValueError('请配置模型 Dataset。')
    if mode == 'account' and not result['dataset']:
        result['dataset'] = user+'/qwen-image-t4-assets'
    return result


def load_config():
    path = config_path()
    if path.is_file():
        try:
            saved=json.loads(path.read_text(encoding='utf-8'))
            saved.setdefault('mode','batch')
            return normalize_config(saved)
        except (OSError,ValueError,TypeError,AttributeError):return None
    user = os.environ.get('KAGGLE_USERNAME', '')
    token = os.environ.get('KAGGLE_API_TOKEN') or os.environ.get('KAGGLE_KEY', '')
    if user and token:
        return normalize_config({'mode':'account','user':user,'token':token})
    return None


def configuration_fingerprint(config):
    return hashlib.sha256(json.dumps({k:v for k,v in config.items() if k!='keep_alive_minutes'}, sort_keys=True).encode('utf-8')).hexdigest()


def make_backend(config):
    if config['mode'] == 'account':
        from account_backend import AccountBackend
        return AccountBackend(config['user'], config['token'], dataset=config['dataset'],acceleration=config['acceleration'])
    if config['mode'] == 'session':
        from session_backend import SessionBackend
        return SessionBackend(config['user'], config['dataset'], config['jupyter_url'], config['kernel_id'],acceleration=config['acceleration'])
    from batch_backend import BatchBackend
    return BatchBackend(config['user'], config['dataset'], config['token'],acceleration=config['acceleration'])


def validate_generation(prompt, seed, steps, model_id=None):
    if __package__:
        from .model_registry import model_profile,model_steps,DEFAULT_MODEL
    else:
        from model_registry import model_profile,model_steps,DEFAULT_MODEL
    model_id=model_profile(model_id)["id"];steps=model_steps(model_id,steps)
    if not isinstance(prompt,str) or not prompt.strip() or not 1 <= len(prompt) <= 4000:
        raise ValueError('提示词需要 1–4000 个字符。')
    if type(seed) is not int or not 0 <= seed <= 2147483647:
        raise ValueError('seed 需要为 0–2147483647 的整数。')
    if type(steps) is not int or not 1 <= steps <= 50:
        raise ValueError('steps 需要为 1–50 的整数。')
    result={'prompt':prompt,'seed':seed,'steps':steps}
    if model_id!=DEFAULT_MODEL:result['model_id']=model_id
    return result


# Explicit allow-lists keep transport addresses, credentials and exception text
# out of durable jobs and MCP tool results.
STAT_FIELDS = {'encoding_seconds','generation_seconds','request_seconds','width','height',
    'steps','seed','peak_gpu_gib','runtime_id','model_load_count','model_reused',
    'generation_number','asset_source','cache_status','backend','model_id','denoising_seconds','decode_seconds',
    'denoising_steps','milliseconds_per_step','attention_backend','parallel_strategy','total_generation_seconds','migration_seconds','replica_load_seconds',
    'dual_model_reused','acceleration_mode','acceleration_fallback','media_type','frames','fps','loading_seconds','cfg_pairs'}
PROGRESS_FIELDS = {'phase','message','downloaded_bytes','total_bytes','percent','file'}
PHASES = {'session','checking','connecting','starting','queued','connected','manifest','download',
    'verify','cached','dataset','copy','loading','generating','cache-check','cache-download',
    'cache-prepare','cache-upload','cache-processing','cache-saved','cache-warning','session-warning','encoding','exporting'}
MESSAGES = {
    'encoding':'正在编码视频提示词。','exporting':'正在保存 MP4 视频。',
    'session':'正在检查 Kaggle 会话。','checking':'正在检查专用 Notebook。',
    'connecting':'正在恢复已有会话。','starting':'正在申请 Kaggle GPU。',
    'queued':'正在等待 Kaggle 准备 GPU。','connected':'会话已连接。',
    'manifest':'正在获取模型清单。','download':'正在下载模型。','verify':'正在校验模型。',
    'cached':'模型文件缓存已就绪。','dataset':'正在拉取模型 Dataset。',
    'copy':'正在准备模型文件。','loading':'正在加载模型。','generating':'正在生成图片。',
    'cache-check':'正在检查私有缓存。','cache-download':'正在下载私有模型缓存。',
    'cache-prepare':'正在保存 INT8/Q4 缓存。','cache-upload':'正在上传私有缓存。',
    'cache-processing':'Kaggle 正在处理私有缓存。','cache-saved':'私有缓存已保存。',
    'cache-warning':'私有缓存暂未就绪；本次仍可继续生图。',
    'session-warning':'远端启动记录未保存；本机连接记录仍可使用。'}


class JobStore:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.outputs = self.directory/'images'
        self.outputs.mkdir(exist_ok=True, mode=0o700)
        self.lock = threading.RLock()
        self.db = sqlite3.connect(self.directory/'jobs.sqlite3', check_same_thread=False)
        os.chmod(self.directory/'jobs.sqlite3', 0o600)
        self.db.row_factory = sqlite3.Row
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('''CREATE TABLE IF NOT EXISTS jobs (
            id TEXT PRIMARY KEY, request_id TEXT UNIQUE NOT NULL, params TEXT NOT NULL,
            config_hash TEXT NOT NULL, state TEXT NOT NULL, created REAL NOT NULL,
            updated REAL NOT NULL, progress TEXT NOT NULL, stats TEXT)''')
        self.db.commit()

    def recover(self):
        with self.lock:
            # A lost worker may have left a remote execution in flight. Never
            # resend it. A cleanly disconnected MCP adapter does not stop this
            # worker, and therefore does not trigger this recovery path.
            self.db.execute("UPDATE jobs SET state='interrupted', updated=?, progress=? WHERE state='running'",
                (time.time(),json.dumps({'phase':'interrupted','message':'后台曾中断，远端可能仍在执行。该任务未重发，请先检查 Kaggle。'},ensure_ascii=False)))
            self.db.commit()

    def submit(self, params, config_hash, request_id):
        if not isinstance(request_id,str) or not re.fullmatch(r'[A-Za-z0-9_.:-]{1,128}',request_id):
            raise ValueError('request_id 需要为 1–128 个字母、数字或 _ . : -。')
        encoded = json.dumps(params,sort_keys=True,ensure_ascii=False)
        with self.lock:
            previous = self.db.execute('SELECT * FROM jobs WHERE request_id=?',(request_id,)).fetchone()
            if previous:
                if previous['params'] != encoded or previous['config_hash'] != config_hash:
                    raise ValueError('request_id 已用于不同参数或配置，请勿复用。')
                return self.get(previous['id']), True
            now = time.time();job_id = secrets.token_hex(16)
            progress = json.dumps({'phase':'queued','message':'任务已进入本机队列。'},ensure_ascii=False)
            self.db.execute('INSERT INTO jobs VALUES (?,?,?,?,?,?,?,?,NULL)',
                (job_id,request_id,encoded,config_hash,'queued',now,now,progress))
            self.db.commit()
            return self.get(job_id), False

    def get(self, job_id):
        if not isinstance(job_id,str) or not re.fullmatch(r'[a-f0-9]{32}',job_id):
            raise ValueError('任务 ID 格式不正确。')
        with self.lock:
            row = self.db.execute('SELECT * FROM jobs WHERE id=?',(job_id,)).fetchone()
            if row is None:raise KeyError('任务不存在。')
            position = self.db.execute("SELECT COUNT(*) FROM jobs WHERE state IN ('queued','running') AND rowid<=(SELECT rowid FROM jobs WHERE id=?)",(job_id,)).fetchone()[0]
            result = {'job_id':row['id'],'request_id':row['request_id'],'state':row['state'],
                'created':row['created'],'updated':row['updated'],**json.loads(row['progress']),
                'queue_position':position if row['state']=='queued' else 0}
            if row['stats']:result['usage']=json.loads(row['stats'])
            video=json.loads(row['params']).get('media_type')=='video'
            result['media_type']='video' if video else 'image'
            result['image_ready']=row['state']=='complete' and not video
            result['video_ready']=row['state']=='complete' and video
            return result

    def update(self, job_id, state=None, progress=None, stats=None):
        with self.lock:
            fields={'updated':time.time()}
            if state:fields['state']=state
            if progress is not None:fields['progress']=json.dumps(progress,ensure_ascii=False)
            if stats is not None:fields['stats']=json.dumps(stats,ensure_ascii=False)
            self.db.execute('UPDATE jobs SET '+','.join(name+'=?' for name in fields)+' WHERE id=?',[*fields.values(),job_id])
            self.db.commit()

    def claim(self):
        with self.lock:
            row=self.db.execute("SELECT * FROM jobs WHERE state='queued' ORDER BY rowid LIMIT 1").fetchone()
            if row is None:return None
            self.update(row['id'],state='running',progress={'phase':'session','message':MESSAGES['session']})
            return dict(row)

    def pending(self):
        with self.lock:
            return self.db.execute("SELECT COUNT(*) FROM jobs WHERE state IN ('queued','running')").fetchone()[0]

    def active_id(self):
        with self.lock:
            row=self.db.execute("SELECT id FROM jobs WHERE state IN ('running','queued') ORDER BY CASE state WHEN 'running' THEN 0 ELSE 1 END,rowid LIMIT 1").fetchone()
            return row[0] if row else None

    def last_usage(self,config_hash=None):
        with self.lock:
            row=self.db.execute("SELECT stats FROM jobs WHERE state='complete'"+
                (" AND config_hash=?" if config_hash else '')+" ORDER BY updated DESC LIMIT 1",
                (config_hash,) if config_hash else ()).fetchone()
            return json.loads(row[0]) if row else None

    def image(self, job_id):
        job=self.get(job_id)
        if job['media_type']!='image':raise ValueError('该任务是视频，请使用视频接口。')
        if job['state']!='complete':raise ValueError('图片尚未生成完成，请先查询 get_job。')
        return (self.outputs/(job_id+'.png')).read_bytes()

    def video(self,job_id):
        job=self.get(job_id)
        if not job['video_ready']:raise ValueError('视频尚未就绪。')
        return (self.outputs/(job_id+'.mp4')).read_bytes()

    def close(self):
        with self.lock:self.db.close()


class StudioCore:
    def __init__(self, directory=None, backend_factory=make_backend):
        self.store=JobStore(directory or state_directory())
        self.store.recover()
        self.factory=backend_factory
        self.lock=threading.RLock()
        self.config=None;self.backend=None;self.fingerprint=None
        self.wake=threading.Event();self.stopping=threading.Event()
        self.worker=None
        self.idle_since=None;self.session_state='not_started';self.releasing=False;self.release_retry_at=0
        self.quota_cache={'available':False,'reason':'not_checked'};self.quota_checked_at=None
        self.quota_lock=threading.Lock()
        initial=load_config()
        if initial:self.configure(initial)

    def configure(self, config, remember=None, api_key=None):
        normalized=normalize_config(config)
        fingerprint=configuration_fingerprint(normalized)
        with self.lock:
            if self.releasing:raise ValueError('正在释放空闲会话，请稍后重试。')
            changed=fingerprint!=self.fingerprint
            if not changed and 'keep_alive_minutes' not in config:normalized['keep_alive_minutes']=self.config['keep_alive_minutes']
            if changed and self.backend is not None and self.store.pending():raise ValueError('队列尚有任务，请完成后再切换账号或连接配置。')
            candidate=self.factory(normalized) if changed else self.backend
            try:
                if remember is True:
                    if not isinstance(api_key,str) or not api_key:raise ValueError('缺少本机 API 密钥。')
                    private_write(config_path(),json.dumps(dict(normalized,api_key=api_key),ensure_ascii=False).encode('utf-8'))
                elif remember is False:
                    config_path().unlink(missing_ok=True)
            except Exception:
                if changed and candidate:
                    try:candidate.close()
                    except Exception:pass
                raise
            if not changed:
                self.config=normalized
                return
            previous=self.backend
            self.backend=candidate;self.config=normalized;self.fingerprint=fingerprint
            self.idle_since=None;self.session_state='not_started'
            self.quota_cache={'available':False,'reason':'not_checked'};self.quota_checked_at=None
            if previous:
                try:previous.close()
                except Exception:pass
        self.wake.set()

    def start(self):
        self.worker=threading.Thread(target=self._work,daemon=True)
        self.worker.start()

    def status(self):
        with self.lock:
            config=self.config or {}
            if __package__:
                from .model_registry import model_catalog
            else:
                from model_registry import model_catalog
            last_usage=self.store.last_usage(self.fingerprint)
            from video_registry import video_catalog
            return {'configured':self.backend is not None,'mode':config.get('mode','account'),'video_models':video_catalog(),
                'user':config.get('user',''),'dataset':config.get('dataset',''),
                'pending_jobs':self.store.pending(),'active_job_id':self.store.active_id(),
                'last_usage':last_usage,'models':model_catalog(),
                'keep_alive_minutes':config.get('keep_alive_minutes',15),'acceleration':config.get('acceleration','auto'),
                'idle_remaining_seconds':self.idle_remaining(),
                'session_state':self.session_state,
                'quota':dict(self.quota_cache),
                'model_state':'released' if self.session_state=='released' else ('last_request_ready' if last_usage else 'not_checked'),
                'message':'模型状态以最近一次生图为准；查询状态不会申请 GPU。'}

    def submit(self,prompt,seed=42,steps=None,request_id=None,model_id=None):
        params=validate_generation(prompt,seed,steps,model_id)
        with self.lock:
            if model_id not in (None,'qwen-image-2.1') and (self.config or {}).get('mode')!='account':raise ValueError('新增模型请使用自动账号模式。')
            if self.releasing:raise ValueError('正在释放空闲会话，请稍后重试。')
            if not self.backend:raise ValueError('先在 GUI 保存账号配置，或设置 KAGGLE_USERNAME 与 KAGGLE_API_TOKEN。')
            job,reused=self.store.submit(params,self.fingerprint,request_id or secrets.token_hex(16))
        self.wake.set()
        return dict(job,deduplicated=reused)

    def idle_remaining(self):
        minutes=(self.config or {}).get('keep_alive_minutes',15)
        if not minutes or self.idle_since is None or self.store.pending() or self.session_state not in ('active','release_failed'):
            return None
        return max(0,math.ceil(minutes*60-(time.monotonic()-self.idle_since)))

    def submit_video(self,prompt,seed=42,steps=None,request_id=None,model_id='wan21-t2v-1.3b',frames=None):
        from video_registry import validate_video
        params=dict(validate_video(prompt,seed,steps,model_id,frames),media_type='video')
        with self.lock:
            if not self.backend or (self.config or {}).get('mode')!='account':
                raise ValueError('视频需要自动账号模式。')
            if self.releasing:raise ValueError('正在释放会话，请稍后重试。')
            job,reused=self.store.submit(params,self.fingerprint,request_id or secrets.token_hex(16))
        self.wake.set()
        return dict(job,deduplicated=reused)

    def expire_idle(self):
        with self.lock:
            remaining=self.idle_remaining()
            if remaining is None or remaining>0 or self.releasing or time.monotonic()<self.release_retry_at:return
            backend=self.backend
            release=getattr(backend,'release_session',None)
            if not callable(release):
                self.session_state='externally_managed';return
            self.releasing=True;self.session_state='releasing'
        try:
            release()
            with self.lock:self.session_state='released';self.idle_since=None
        except Exception:
            with self.lock:
                self.session_state='release_failed';self.release_retry_at=time.monotonic()+60
        finally:
            with self.lock:self.releasing=False

    def quota(self):
        # This read never provisions a session and is rate-limited across all clients.
        with self.quota_lock:
            with self.lock:
                backend=self.backend;fingerprint=self.fingerprint
                if self.quota_checked_at is not None and time.monotonic()-self.quota_checked_at<60:return dict(self.quota_cache)
            getter=getattr(backend,'quota',None)
            try:result=getter() if callable(getter) else {'available':False,'reason':'unsupported_mode'}
            except Exception:result={'available':False,'reason':'api_unavailable'}
            with self.lock:
                if fingerprint!=self.fingerprint:return {'available':False,'reason':'configuration_changed'}
                self.quota_cache=dict(result,checked_at=time.time());self.quota_checked_at=time.monotonic()
                return dict(self.quota_cache)

    def _progress(self, job_id, event):
        phase=event.get('phase')
        if phase not in PHASES:return
        clean={'phase':phase,'message':MESSAGES[phase]}
        for name in ('downloaded_bytes','total_bytes','percent','step','steps'):
            value=event.get(name)
            if isinstance(value,(int,float)) and not isinstance(value,bool) and value>=0:
                clean[name]=value
        # Filenames can be supplied by remote logs: only display a safe basename.
        filename=event.get('file')
        if isinstance(filename,str) and re.fullmatch(r'[A-Za-z0-9_. /-]{1,200}',filename):
            clean['file']=filename.rsplit('/',1)[-1]
        self.store.update(job_id,progress=clean)

    def _work(self):
        while not self.stopping.is_set():
            with self.lock:
                if self.backend is None:
                    row=None
                else:
                    row=self.store.claim()
                backend=self.backend
            if row is None:
                self.expire_idle()
                self.wake.wait(.5);self.wake.clear();continue
            job_id=row['id']
            if row['config_hash']!=self.fingerprint:
                self.store.update(job_id,state='error',progress={'phase':'error','message':'账号配置已变化，该排队任务未执行。'})
                continue
            try:
                params=json.loads(row['params'])
                video=params.pop('media_type',None)=='video'
                if video:
                    raw,stats=backend.generate_video_with_progress(**params,progress=lambda event:self._progress(job_id,event))
                elif callable(getattr(backend,'generate_with_progress',None)):
                    raw,stats=backend.generate_with_progress(**params,progress=lambda event:self._progress(job_id,event))
                else:
                    raw,stats=backend.generate(**params)
                if video and (len(raw)<12 or raw[4:8]!=b'ftyp'):raise ValueError('Invalid MP4 result')
                if not video and (not raw.startswith(b'\x89PNG\r\n\x1a\n') or len(raw)<24 or struct.unpack('>II',raw[16:24])!=(512,512)):
                    raise ValueError('Invalid image result')
                clean_stats={key:value for key,value in stats.items() if key in STAT_FIELDS}
                private_write(self.store.outputs/(job_id+('.mp4' if video else '.png')),raw)
                self.store.update(job_id,state='complete',progress={'phase':'complete','message':'视频已生成并保存。' if video else '图片已生成并保存。'},stats=clean_stats)
            except Exception:
                self.store.update(job_id,state='error',progress={'phase':'error','message':'生成失败；该任务没有自动重发。请检查配置、网络或 Kaggle 会话后再决定是否提交新任务。'})

            finally:
                with self.lock:
                    self.idle_since=time.monotonic();self.session_state='active';self.release_retry_at=0

    def close(self):
        self.stopping.set();self.wake.set()
        if self.worker:self.worker.join(timeout=2)
        # Never close an active transport underneath a generation. Process exit
        # leaves running rows for recover() to mark interrupted on the next boot.
        if self.worker and self.worker.is_alive():return
        if self.backend:
            try:self.backend.close()
            except Exception:pass
        self.store.close()
