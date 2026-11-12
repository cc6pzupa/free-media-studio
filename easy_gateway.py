"""Local setup form and fixed image API. No Jupyter URL is needed."""
import base64
import hmac
import json
import os
import secrets
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler
from local_http import LocalHTTPServer
from pathlib import Path
from urllib.parse import urlsplit
from batch_backend import BatchBackend
from app_entry import VERSION


def validate(data):
    if not isinstance(data,dict):raise ValueError('请求必须是 JSON 对象')
    from model_registry import model_profile,model_steps
    profile=model_profile(data.get('model_id'))
    prompt=data.get('prompt','');seed=data.get('seed',42);steps=model_steps(profile['id'],data.get('steps'))
    if not isinstance(prompt,str) or not 1<=len(prompt.strip())<=4000:raise ValueError('请输入 1–4000 字的提示词')
    if type(seed) is not int or not 0<=seed<=2147483647:raise ValueError('seed 应为 0–2147483647 的整数')
    if type(steps) is not int or not 1<=steps<=50:raise ValueError('steps 应为 1–50 的整数')
    if data.get('size','512x512')!='512x512' or data.get('n',1)!=1 or data.get('response_format','b64_json')!='b64_json':raise ValueError('当前只支持单张 512×512、b64_json 返回')
    return prompt,seed,steps


def create_server(port=8787, backend_factory=BatchBackend, keep_alive_minutes=None, acceleration=None):
    from studio_client import SharedBackend, StudioClient, StudioHTTPError
    shared_mode=backend_factory is BatchBackend
    nonce=secrets.token_urlsafe(32); api_key=secrets.token_urlsafe(32)
    backend=None; state_lock=threading.Lock(); generation_lock=threading.Lock(); jobs={}
    config_path=Path(os.environ.get('EASY_CONFIG_PATH',str(Path.home()/'.kaggle-qwen-client/settings.json')))
    saved_user=saved_dataset=''
    def make_backend(data):
        if keep_alive_minutes is not None:data=dict(data,keep_alive_minutes=keep_alive_minutes)
        if acceleration is not None:data=dict(data,acceleration=acceleration)
        if shared_mode:
            return SharedBackend(dict(data,mode=data.get('mode','batch')),activate=False)
        user=data.get('user','').strip()
        dataset=data.get('dataset','').strip()
        mode=data.get('mode','batch')
        if mode=='account':
            from account_backend import AccountBackend
            return AccountBackend(user,data.get('token','').strip(),dataset=dataset or None,**({'acceleration':data['acceleration']} if 'acceleration' in data else {}))
        if mode=='session':
            from session_backend import SessionBackend
            return SessionBackend(user,dataset,data.get('jupyter_url','').strip(),data.get('kernel_id') or None,**({'acceleration':data['acceleration']} if 'acceleration' in data else {}))
        if mode!='batch':
            raise ValueError('请选择自动连接、常驻复用或批处理模式')
        return backend_factory(user,dataset,data.get('token','').strip())
    if config_path.is_file():
        try:
            saved=json.loads(config_path.read_text(encoding='utf-8'))
            backend=make_backend(saved)
            if isinstance(backend,SharedBackend):backend.activate()
            api_key=saved['api_key'];saved_user=saved['user'];saved_dataset=saved['dataset']
        except (OSError,ValueError,KeyError,RuntimeError):backend=None

    def sync_shared_backend():
        nonlocal backend,saved_user,saved_dataset
        if not shared_mode or backend is not None:return
        client=StudioClient(autostart=False)
        try:
            status=client.status()
            if status['configured']:
                backend=SharedBackend.from_running(client,status)
                saved_user=status['user'];saved_dataset=status['dataset']
            else:client.close()
        except RuntimeError:client.close()
    page=Path(__file__).with_name('easy_ui.html').read_text(encoding='utf-8').replace('__NONCE__',nonce)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def send(self,status,body,kind='application/json; charset=utf-8'):
            raw=json.dumps(body,ensure_ascii=False).encode() if not isinstance(body,bytes) else body
            self.send_response(status);self.send_header('Content-Type',kind);self.send_header('Content-Length',str(len(raw)))
            self.send_header('Cache-Control','no-store');self.send_header('X-Content-Type-Options','nosniff');self.send_header('Referrer-Policy','no-referrer');self.end_headers();self.wfile.write(raw)
        def local(self):
            # Reject cross-site browser requests, including DNS rebinding host names.
            host=self.headers.get('Host','')
            allowed={f'127.0.0.1:{self.server.server_port}',f'localhost:{self.server.server_port}'}
            origin=self.headers.get('Origin')
            return host in allowed and (origin is None or origin in {'http://'+h for h in allowed})
        def ui_auth(self):return hmac.compare_digest(self.headers.get('X-Local-Session',''),nonce)
        def api_auth(self):return hmac.compare_digest(self.headers.get('Authorization',''),'Bearer '+api_key)
        def do_GET(self):
            if not self.local():return self.send(403,{'error':{'message':'仅允许本机访问'}})
            if self.path=='/models':
                from model_registry import model_catalog
                return self.send(200,{'models':model_catalog()})
            if self.path=='/health':return self.send(200,{'app':'qwen-studio','version':VERSION})
            if self.path=='/gallery':
                gallery=Path(__file__).with_name('gallery_ui.html').read_text(encoding='utf-8').replace('__NONCE__',nonce)
                return self.send(200,gallery.encode(),'text/html; charset=utf-8')
            if urlsplit(self.path).path=='/gallery/items' or self.path.startswith('/gallery/media/'):
                if not self.ui_auth():return self.send(401,{'error':{'message':'访问密钥无效'}})
                client=StudioClient()
                try:
                    if self.path.startswith('/gallery/media/'):
                        job_id=self.path[15:];job=client.get_job(job_id)
                        video=job.get('media_type')=='video'
                        raw=client.get_video(job_id) if video else client.get_image(job_id)
                        return self.send(200,raw,'video/mp4' if video else 'image/png')
                    return self.send(200,client.history(urlsplit(self.path).query))
                except ValueError:return self.send(400,{'error':{'message':'无效的画廊请求。'}})
                except StudioHTTPError as error:return self.send(error.status_code,{'error':{'message':str(error)}})
                except RuntimeError:return self.send(503,{'error':{'message':'历史后台暂不可用，请重试。'}})
                finally:client.close()
            if self.path=='/video':
                video_page=Path(__file__).with_name('video_ui.html').read_text(encoding='utf-8').replace('__NONCE__',nonce)
                return self.send(200,video_page.encode(),'text/html; charset=utf-8')
            if self.path=='/video/models':
                from video_registry import video_catalog
                return self.send(200,{'models':video_catalog()})
            if self.path.startswith(('/video/jobs/','/v1/videos/jobs/')):
                api=self.path.startswith('/v1/')
                if not (self.api_auth() if api else self.ui_auth()):return self.send(401,{'error':{'message':'访问密钥无效'}})
                sync_shared_backend()
                if not isinstance(backend,SharedBackend):return self.send(409,{'error':{'message':'先在主界面配置自动账号模式。'}})
                parts=self.path.split('/');job_id=parts[4] if api and len(parts)>=5 else parts[3] if len(parts)>=4 else ''
                content=parts[-1]=='content'
                try:
                    job=backend.client.get_job(job_id)
                    if job.get('media_type')!='video':raise ValueError('不是视频任务。')
                    return self.send(200,backend.client.get_video(job_id),'video/mp4') if content else self.send(200,job)
                except ValueError:return self.send(404,{'error':{'message':'视频任务不存在。'}})
                except StudioHTTPError as error:
                    status=404 if error.status_code==404 else 409 if error.status_code==409 else 503
                    return self.send(status,{'error':{'message':str(error)}})
                except RuntimeError:return self.send(503,{'error':{'message':'本机后台暂不可用，正在保留原视频任务。'}})
            if self.path in ('/usage','/v1/studio/usage'):
                if not (self.ui_auth() if self.path=='/usage' else self.api_auth()):return self.send(401,{'error':{'message':'访问密钥无效'}})
                sync_shared_backend()
                if isinstance(backend,SharedBackend):
                    try:
                        quota=backend.client.quota();status=backend.client.status()
                        return self.send(200,{'quota':quota,**{k:status.get(k) for k in ('keep_alive_minutes','idle_remaining_seconds','session_state')}})
                    except RuntimeError:return self.send(200,{'quota':{'available':False,'reason':'worker_unavailable'}})
                return self.send(200,{'quota':{'available':False,'reason':'not_configured'}})
            if self.path=='/configuration' and self.ui_auth():
                sync_shared_backend()
                with state_lock:
                    active_job_id=next((key for key,job in jobs.items() if job['state']=='running'),None)
                if isinstance(backend,SharedBackend):
                    try:
                        active_job_id=backend.client.status()['active_job_id']
                        if active_job_id and backend.client.get_job(active_job_id).get('media_type')=='video':active_job_id=None
                    except RuntimeError:pass
                return self.send(200,{'configured':backend is not None,'user':saved_user,'dataset':saved_dataset,'mode':getattr(backend,'mode','batch'),'remembered':config_path.is_file(),'acceleration':getattr(backend,'acceleration',acceleration or 'auto'),'keep_alive_minutes':getattr(backend,'keep_alive_minutes',keep_alive_minutes if keep_alive_minutes is not None else 15),'endpoint':f'http://127.0.0.1:{self.server.server_port}/v1/images/generations','api_key':api_key if backend else '', 'active_job_id':active_job_id})
            if self.path=='/':return self.send(200,page.encode(),'text/html; charset=utf-8')
            if self.path in ('/kaggle_standalone.ipynb','/kaggle_video.ipynb'):
                notebook=Path(__file__).with_name(self.path[1:])
                if notebook.is_file():return self.send(200,notebook.read_bytes(),'application/x-ipynb+json')
            if self.path.startswith('/jobs/') and self.ui_auth():
                sync_shared_backend()
                if isinstance(backend,SharedBackend):
                    try:
                        job_id=self.path[6:];job=backend.client.get_job(job_id)
                        if job['state']=='interrupted':job['state']='error'
                        if job['state']=='complete':
                            job['result']={'data':[{'b64_json':base64.b64encode(backend.client.get_image(job_id)).decode()}],'usage':job['usage']}
                        return self.send(200,job)
                    except (RuntimeError,ValueError):return self.send(404,{'error':{'message':'任务不存在或后台暂不可用；没有重新提交。'}})
                with state_lock:job=jobs.get(self.path[6:])
                return self.send(200 if job else 404,job or {'error':{'message':'任务不存在'}})
            return self.send(404,{'error':{'message':'地址不存在'}})
        def do_POST(self):
            nonlocal backend, saved_user, saved_dataset
            if not self.local():return self.send(403,{'error':{'message':'仅允许本机访问'}})
            if self.path.startswith('/gallery/items/'):
                if not self.ui_auth():return self.send(401,{'error':{'message':'访问密钥无效'}})
                client=StudioClient()
                try:
                    length=int(self.headers.get('Content-Length','0'))
                    if not 0<length<=1024:raise ValueError()
                    body=json.loads(self.rfile.read(length))
                    if not isinstance(body,dict):raise ValueError()
                    return self.send(200,client.edit_history(self.path[15:],body))
                except (ValueError,TypeError):return self.send(400,{'error':{'message':'无效的画廊操作。'}})
                except StudioHTTPError as error:return self.send(error.status_code,{'error':{'message':str(error)}})
                except RuntimeError:return self.send(503,{'error':{'message':'历史后台暂不可用，请重试。'}})
                finally:client.close()
            if self.path in ('/video/jobs','/v1/videos/generations'):
                if not (self.api_auth() if self.path.startswith('/v1/') else self.ui_auth()):return self.send(401,{'error':{'message':'访问密钥无效'}})
                try:
                    length=int(self.headers.get('Content-Length','0'))
                    if not 0<length<=32768:raise ValueError('请求过大或为空')
                    data=json.loads(self.rfile.read(length))
                    if not isinstance(data,dict):raise ValueError('请求必须为对象')
                    sync_shared_backend()
                    if not isinstance(backend,SharedBackend):raise ValueError('先在主界面配置自动账号模式。')
                    job=backend.client.submit_video(data.get('prompt'),data.get('seed',42),data.get('steps'),
                        data.get('request_id'),data.get('model_id','wan21-t2v-1.3b'),data.get('frames'))
                    return self.send(202,job)
                except (ValueError,TypeError,AttributeError):return self.send(400,{'error':{'message':'检查视频模型、提示词、种子、步数与帧数。'}})
                except RuntimeError as error:return self.send(409,{'error':{'message':str(error)}})
            is_config=self.path=='/configure';is_job=self.path=='/jobs';is_api=self.path=='/v1/images/generations'
            if not (is_config or is_job or is_api):return self.send(404,{'error':{'message':'地址不存在'}})
            if not (self.api_auth() if is_api else self.ui_auth()):return self.send(401,{'error':{'message':'访问密钥无效'}})
            try:
                length=int(self.headers.get('Content-Length','0'))
                if not 0<length<=32768:raise ValueError('请求过大或为空')
                data=json.loads(self.rfile.read(length))
                if is_config:
                    if not isinstance(data,dict):raise ValueError('配置格式有误')
                    if keep_alive_minutes is not None and 'keep_alive_minutes' not in data:data['keep_alive_minutes']=keep_alive_minutes
                    if acceleration is not None and 'acceleration' not in data:data['acceleration']=acceleration
                    candidate=SharedBackend(dict(data,mode=data.get('mode','batch')),activate=False) if shared_mode else make_backend(data)
                    if not generation_lock.acquire(blocking=False):return self.send(409,{'error':{'message':'正在生成，请结束后再修改配置'}})
                    try:
                        if isinstance(candidate,SharedBackend):
                            candidate.activate(remember=data.get('remember') is True,api_key=api_key)
                        elif data.get('remember') is True:
                            config_path.parent.mkdir(mode=0o700,parents=True,exist_ok=True)
                            saved_config={'user':candidate.user,'dataset':candidate.dataset,'mode':getattr(candidate,'mode','batch'),'api_key':api_key}
                            if saved_config['mode']=='session':
                                saved_config.update(jupyter_url=candidate.jupyter_url,kernel_id=candidate.kernel_id)
                            else:
                                saved_config['token']=candidate.token
                            raw=json.dumps(saved_config)
                            fd=os.open(str(config_path),os.O_CREAT|os.O_TRUNC|os.O_WRONLY,0o600)
                            with os.fdopen(fd,'w',encoding='utf-8') as f:f.write(raw)
                            os.chmod(config_path,0o600)
                        elif config_path.exists():
                            config_path.unlink()
                        old_backend=backend
                        backend=candidate;saved_user=candidate.user;saved_dataset=candidate.dataset
                        if old_backend is not None and callable(getattr(old_backend,'close',None)):
                            try:old_backend.close()
                            except Exception:pass
                    finally:generation_lock.release()
                    return self.send(200,{'ok':True,'endpoint':f'http://127.0.0.1:{self.server.server_port}/v1/images/generations','api_key':api_key})
                prompt,seed,steps=validate(data)
            except OSError:return self.send(500,{'error':{'message':'无法保存本机配置。请检查文件权限，或取消记住配置。'}})
            except RuntimeError as error:return self.send(409,{'error':{'message':str(error)}})
            except (ValueError,TypeError,AttributeError):return self.send(400,{'error':{'message':'检查用户名、Dataset，以及所选模式的连接地址或 Token；提示词不能为空，steps 为 1–50。'}})
            sync_shared_backend()
            if backend is None:return self.send(409,{'error':{'message':'请先完成页面上方的首次配置'}})
            if is_job and isinstance(backend,SharedBackend):
                try:
                    job=backend.submit(prompt,seed,steps,data.get('request_id'),data.get('model_id'))
                    return self.send(202,{'job_id':job['job_id']})
                except (RuntimeError,ValueError) as error:return self.send(409,{'error':{'message':str(error)}})
            # Only accept a second task after the first finishes. No silent resubmission.
            if not generation_lock.acquire(blocking=False):return self.send(409,{'error':{'message':'已有任务正在运行，请等待完成'}})
            current=backend
            def generate(progress=None):
                try:
                    generate_with_progress=getattr(current,'generate_with_progress',None)
                    extra={'model_id':data['model_id']} if isinstance(current,SharedBackend) and data.get('model_id') else {}
                    if data.get('model_id') not in (None,'qwen-image-2.1') and not isinstance(current,SharedBackend):raise ValueError('新增模型请使用自动账号模式。')
                    if progress is not None and callable(generate_with_progress):
                        image,stats=generate_with_progress(prompt,seed,steps,progress=progress,**extra)
                    else:
                        image,stats=current.generate(prompt,seed,steps,**extra)
                    return {'created':int(time.time()),'data':[{'b64_json':base64.b64encode(image).decode()}],'usage':stats}
                finally:generation_lock.release()
            if is_api:
                try:return self.send(200,generate())
                except Exception as e:return self.send(502,{'error':{'message':safe_error(e)}})
            job_id=secrets.token_hex(16)
            with state_lock:
                # Keep a bounded number of images in process memory.
                for old in list(jobs)[:-4]:jobs.pop(old,None)
                jobs[job_id]={'state':'running','phase':'submitting','message':'正在提交新任务。'}
            def update_progress(update):
                with state_lock:
                    jobs[job_id].update(update)
            def work():
                try:result={'state':'complete','phase':'complete','result':generate(update_progress)}
                except Exception as e:result={'state':'error','message':safe_error(e)}
                with state_lock:jobs[job_id].update(result)
            threading.Thread(target=work,daemon=True).start()
            self.send(202,{'job_id':job_id})
    return LocalHTTPServer(('127.0.0.1',port),Handler)


def safe_error(error):
    if isinstance(error,(RuntimeError,TimeoutError)):return str(error)
    return '生成失败。请检查本机网络和 Kaggle 任务日志；确认旧任务状态后再重试。'


def main():
    port=int(os.environ.get('IMAGE_API_PORT','8787'))
    try:server=create_server(port)
    except OSError:raise SystemExit('8787 端口已被占用。请先检查是否已经启动本程序；不要重复打开。')
    print(f'请在浏览器打开 http://127.0.0.1:{port} 。关闭此窗口会停止本机服务。',flush=True)
    threading.Timer(.5,lambda:webbrowser.open(f'http://127.0.0.1:{port}')).start()
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close()

if __name__=='__main__':main()
