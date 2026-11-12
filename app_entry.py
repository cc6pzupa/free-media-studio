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
            folder.mkdir(parents=True,exist_ok=True)
            stream=(folder/'desktop.log').open('a',encoding='utf-8')
            if sys.stdout is None:sys.stdout=stream
            if sys.stderr is None:sys.stderr=stream
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
        from desktop import main as run
    return run()


if __name__=='__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    raise SystemExit(main())
