"""Single executable entry point for the desktop, independent worker and MCP."""
import os
import sys
from pathlib import Path

VERSION='0.9.0'


def main():
    if getattr(sys,'frozen',False):
        # Windowed bootloaders have no console; keep worker diagnostics usable.
        if sys.stdout is None or sys.stderr is None:
            folder=Path(os.environ.get('EASY_CONFIG_PATH',str(Path.home()/'.kaggle-qwen-client/settings.json'))).parent/'studio'
            folder.mkdir(parents=True,exist_ok=True,mode=0o700)
            fd=os.open(folder/'desktop.log',os.O_CREAT|os.O_APPEND|os.O_WRONLY,0o600)
            stream=os.fdopen(fd,'a',encoding='utf-8')
            if sys.stdout is None:sys.stdout=stream
            if sys.stderr is None:sys.stderr=stream
        # Frozen Python does not apply PYTHONUTF8 like the normal interpreter.
        # Redirected Windows streams may otherwise use cp1252 and fail on the
        # Chinese startup/progress messages before the HTTP service stays up.
        for stream in (sys.stdout,sys.stderr):
            if callable(getattr(stream,'reconfigure',None)):
                stream.reconfigure(encoding='utf-8',errors='backslashreplace')
    if '--version' in sys.argv:
        print('CC Media Studio '+VERSION);return 0
    if '--self-test' in sys.argv:
        import ast
        from video_session import video_bootstrap_code,VIDEO_FILES
        from video_registry import VIDEO_MODELS
        root=Path(__file__).resolve().parent
        for name in ('easy_ui.html','video_ui.html','gallery_ui.html','template.ipynb','kaggle_standalone.ipynb','kaggle_video.ipynb',*VIDEO_FILES):
            assert (root/name).is_file(),name
        for model in VIDEO_MODELS:ast.parse(video_bootstrap_code(model,install=False))
        import account_backend,session_backend,batch_backend,websocket,requests,kaggle
        print('CC Media Studio bundle check passed');return 0
    if '--worker' in sys.argv:
        sys.argv.remove('--worker')
        from studio_worker import main as run
    elif '--mcp' in sys.argv:
        sys.argv.remove('--mcp')
        from mcp_server import build_server
        return build_server().run(transport='stdio')
    else:
        if os.environ.get('CC_MEDIA_DIAGNOSTICS')=='1':print('Desktop entry: importing GUI',flush=True)
        from desktop import main as run
        if os.environ.get('CC_MEDIA_DIAGNOSTICS')=='1':print('Desktop entry: starting GUI',flush=True)
    return run()


if __name__=='__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    try:
        raise SystemExit(main())
    except SystemExit as error:
        if error.code:
            import traceback
            traceback.print_exc()
        raise
    except Exception:
        import traceback
        traceback.print_exc()
        raise SystemExit(1)
