"""Android entry point: the worker and the GUI run inside the app process.

Android has no separate Python executable, so the worker that desktop builds
spawn as a process runs here as a thread. The Java side copies the app files
to private storage, then calls start() from its foreground service.
"""
import os
import threading
from pathlib import Path

_lock=threading.Lock()
_state={}


def _start_worker(timeout=30):
    """Run studio_worker.serve() in a thread and wait until it accepts requests."""
    import studio_worker
    from studio_core import state_directory
    thread=_state.get('worker_thread')
    if thread is not None and thread.is_alive():return
    ready=threading.Event()
    def on_ready(core,server):
        _state['core']=core;_state['worker']=server;ready.set()
    thread=threading.Thread(target=studio_worker.serve,kwargs=dict(directory=state_directory(),on_ready=on_ready),
        name='cc-media-worker',daemon=True)
    _state['worker_thread']=thread;thread.start()
    if not ready.wait(timeout):raise RuntimeError('本机后台未能启动。 / The local worker did not start.')


def start(data_dir,port=8787):
    """Start the worker and the GUI server once; return the GUI address."""
    with _lock:
        if 'url' in _state:return _state['url']
        data=Path(data_dir);data.mkdir(parents=True,exist_ok=True)
        os.environ['HOME']=str(data)
        os.environ['EASY_CONFIG_PATH']=str(data/'config'/'settings.json')
        os.environ['PYTHONUTF8']='1'
        import studio_client
        import easy_gateway
        studio_client.worker_starter=_start_worker
        _start_worker()
        try:server=easy_gateway.create_server(port)
        except OSError:server=easy_gateway.create_server(0)
        threading.Thread(target=server.serve_forever,name='cc-media-gui',daemon=True).start()
        _state['gateway']=server
        _state['url']='http://127.0.0.1:%d'%server.server_port
        return _state['url']


def busy():
    """True while jobs are queued or a Kaggle session is held (keep the CPU awake)."""
    core=_state.get('core')
    if core is None:return False
    try:
        with core.lock:state=core.session_state
        return bool(core.store.pending()) or state in ('active','releasing','release_failed')
    except Exception:return False


def stop():
    """Release a held Kaggle session, then stop both servers. Returns a short status."""
    core=_state.get('core');message='stopped'
    if core is not None:
        with core.lock:backend=core.backend;state=core.session_state
        release=getattr(backend,'release_session',None)
        if state in ('active','release_failed') and callable(release):
            try:release();message='released'
            except Exception:message='release_failed'
    thread=_state.get('worker_thread')
    for key in ('gateway','worker'):
        server=_state.pop(key,None)
        if server is not None:
            try:server.shutdown();server.server_close()
            except Exception:pass
    # serve() closes the job database after serve_forever returns; wait for it.
    if thread is not None:thread.join(15)
    _state.clear()
    return message
