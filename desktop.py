"""Cross-platform GUI entry point; image computation stays on Kaggle."""
import argparse
import json
import threading
import webbrowser
from urllib.request import urlopen

from easy_gateway import create_server


def main(argv=None):
    parser = argparse.ArgumentParser(description='CC Media Studio · Kaggle image client')
    parser.add_argument('--browser', action='store_true', help='Use the system browser instead of a native window')
    parser.add_argument('--no-open', action='store_true', help='Serve the GUI without opening a window (for remote/headless use)')
    parser.add_argument('--port', type=int, default=8787, help='Local port (default: 8787; 0 selects a free port)')
    parser.add_argument('--keep-alive-minutes',type=int,help='Idle retention in minutes (default: 15; 0 disables auto-release)')
    args = parser.parse_args(argv)
    if args.keep_alive_minutes is not None and not 0<=args.keep_alive_minutes<=1440:
        parser.error('--keep-alive-minutes must be between 0 and 1440')
    if not 0 <= args.port <= 65535:
        parser.error('--port must be between 0 and 65535')
    try:
        server = create_server(args.port,keep_alive_minutes=args.keep_alive_minutes)
    except OSError:
        address = f'http://127.0.0.1:{args.port}'
        try:
            with urlopen(address+'/health', timeout=3) as response:
                health = json.load(response)
            if health.get('app') != 'qwen-studio':
                raise ValueError()
        except Exception:
            raise SystemExit(f'{args.port} 端口已被其他程序占用。请关闭旧程序，或使用 --port 指定其他端口。')
        print('CC Media Studio 已在运行：'+address, flush=True)
        if not args.no_open:
            webbrowser.open(address)
        return
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    address = f'http://127.0.0.1:{server.server_port}'
    print('CC Media Studio: '+address, flush=True)
    try:
        use_browser = args.browser or args.no_open
        if not use_browser:
            try:
                import webview
                webview.create_window('CC Media Studio', address, width=1300, height=880,
                    min_size=(980, 700), background_color='#f5f6fa')
                webview.start()
            except Exception:
                print('原生窗口暂不可用，改用本机浏览器界面。', flush=True)
                use_browser = True
        if use_browser:
            if not args.no_open:
                webbrowser.open(address)
            print('保持此进程运行；退出仅关闭本机服务，不会停止 Kaggle GPU 会话。', flush=True)
            worker.join()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        server.server_close()


if __name__ == '__main__':
    main()
