"""Private localhost worker, independent from GUI and MCP adapter lifetimes."""
import argparse
import base64
import hmac
import json
import os
import secrets
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from studio_core import StudioCore, private_write, state_directory

VERSION='0.7.0'


class ProcessLock:
    """One byte-range lock on Windows; flock on POSIX. Released on process death."""
    def __init__(self,path):
        self.path=Path(path);self.file=None
    def acquire(self):
        self.path.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
        self.file=self.path.open('a+b');os.chmod(self.path,0o600)
        self.file.seek(0)
        if self.file.read(1)==b'':self.file.write(b'0');self.file.flush()
        self.file.seek(0)
        try:
            if os.name=='nt':
                import msvcrt
                msvcrt.locking(self.file.fileno(),msvcrt.LK_NBLCK,1)
            else:
                import fcntl
                fcntl.flock(self.file.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
            return True
        except OSError:
            self.file.close();self.file=None;return False
    def close(self):
        if self.file:
            if os.name=='nt':
                import msvcrt
                self.file.seek(0);msvcrt.locking(self.file.fileno(),msvcrt.LK_UNLCK,1)
            else:
                import fcntl
                fcntl.flock(self.file.fileno(),fcntl.LOCK_UN)
            self.file.close();self.file=None


def create_worker_server(core, token, port=0):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def send(self,status,body,kind='application/json; charset=utf-8'):
            raw=body if isinstance(body,bytes) else json.dumps(body,ensure_ascii=False).encode('utf-8')
            self.send_response(status);self.send_header('Content-Type',kind)
            self.send_header('Content-Length',str(len(raw)));self.send_header('Cache-Control','no-store')
            self.send_header('X-Content-Type-Options','nosniff');self.end_headers();self.wfile.write(raw)
        def authorized(self):
            allowed={f'127.0.0.1:{self.server.server_port}',f'localhost:{self.server.server_port}'}
            host=self.headers.get('Host','');origin=self.headers.get('Origin')
            if host not in allowed or (origin is not None and origin not in {'http://'+h for h in allowed}):
                self.send(403,{'error':'仅允许本机访问。'});return False
            if not hmac.compare_digest(self.headers.get('Authorization',''),'Bearer '+token):
                self.send(401,{'error':'本机后台认证失败。'});return False
            return True
        def do_GET(self):
            if not self.authorized():return
            try:
                if self.path=='/health':return self.send(200,{'app':'qwen-studio-worker','version':VERSION,'protocol':1,'model_selection':True,'dual_cfg':True})
                if self.path=='/quota':return self.send(200,core.quota())
                if self.path=='/status':return self.send(200,core.status())
                if self.path.startswith('/jobs/'):
                    pieces=self.path.split('/')
                    if len(pieces)==3:return self.send(200,core.store.get(pieces[2]))
                    if len(pieces)==4 and pieces[3]=='image':return self.send(200,core.store.image(pieces[2]),'image/png')
                return self.send(404,{'error':'接口不存在。'})
            except KeyError:return self.send(404,{'error':'任务不存在。'})
            except ValueError:return self.send(409,{'error':'任务 ID 无效或图片尚未就绪，请先查询任务。'})
            except Exception:return self.send(500,{'error':'无法读取本机任务。'})
        def do_POST(self):
            if not self.authorized():return
            try:
                length=int(self.headers.get('Content-Length','0'))
                if not 0<length<=65536:raise ValueError('请求过大或为空。')
                body=json.loads(self.rfile.read(length))
                if not isinstance(body,dict):raise ValueError('请求必须为 JSON 对象。')
                if self.path=='/configure':
                    if isinstance(body.get('config'),dict):
                        core.configure(body['config'],remember=body.get('remember'),api_key=body.get('api_key'))
                    else:core.configure(body)
                    return self.send(200,core.status())
                if self.path=='/jobs':
                    job=core.submit(body.get('prompt'),body.get('seed',42),body.get('steps'),body.get('request_id'),body.get('model_id'))
                    return self.send(202,job)
                if self.path=='/shutdown':
                    with core.lock:
                        if core.store.pending():return self.send(409,{'error':'队列仍有任务，完成后再停止后台。'})
                        core.stopping.set()
                    self.send(200,{'stopped':True})
                    threading.Thread(target=self.server.shutdown,daemon=True).start()
                    return
                return self.send(404,{'error':'接口不存在。'})
            except OSError:return self.send(500,{'error':'无法保存本机配置，请检查目录权限。'})
            except (ValueError,TypeError,AttributeError) as error:
                # Validation messages are authored locally and contain no values.
                return self.send(400,{'error':str(error) if type(error) is ValueError else '请求格式有误。'})
            except Exception:return self.send(500,{'error':'本机后台操作失败。'})
    return ThreadingHTTPServer(('127.0.0.1',port),Handler)


def serve(directory=None,backend_factory=None,keep_alive_minutes=None):
    directory=Path(directory or state_directory())
    lock=ProcessLock(directory/'worker.lock')
    if not lock.acquire():return
    core=server=None;info_path=directory/'connection.json'
    token=secrets.token_urlsafe(32)
    try:
        core=StudioCore(directory,**({'backend_factory':backend_factory} if backend_factory else {}))
        if keep_alive_minutes is not None and core.config:
            core.configure(dict(core.config,keep_alive_minutes=keep_alive_minutes))
        server=create_worker_server(core,token)
        private_write(info_path,json.dumps({'url':f'http://127.0.0.1:{server.server_port}',
            'token':token,'pid':os.getpid(),'protocol':1}).encode('utf-8'))
        core.start()
        try:server.serve_forever()
        except KeyboardInterrupt:pass
    finally:
        if server:server.server_close()
        if core:core.close()
        try:
            if json.loads(info_path.read_text(encoding='utf-8')).get('token')==token:info_path.unlink()
        except (OSError,ValueError):pass
        lock.close()


def main():
    parser=argparse.ArgumentParser(description='CC Media Studio shared local worker')
    parser.add_argument('--state-dir',type=Path)
    parser.add_argument('--keep-alive-minutes',type=int,help='Idle retention in minutes (default: 15; 0 disables auto-release)')
    parser.add_argument('--stop',action='store_true',help='Stop the local worker after all jobs finish; does not stop Kaggle')
    args=parser.parse_args()
    if args.keep_alive_minutes is not None:
        from studio_core import keep_alive_value
        try:os.environ['CC_MEDIA_KEEP_ALIVE_MINUTES']=str(keep_alive_value(args.keep_alive_minutes))
        except ValueError as error:parser.error(str(error))
    if args.stop:
        from studio_client import StudioClient
        client=StudioClient(args.state_dir,autostart=False)
        print(json.dumps(client.shutdown(),ensure_ascii=False))
    else:serve(args.state_dir,keep_alive_minutes=args.keep_alive_minutes)


if __name__=='__main__':main()
