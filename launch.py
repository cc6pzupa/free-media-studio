"""Create/update a private environment, then run the desktop entry point."""
import hashlib
import os
import subprocess
import sys
import venv
from pathlib import Path


def main():
    if sys.version_info < (3, 11):
        raise SystemExit('请先安装 Python 3.11 或更新版本，再启动 CC Media Studio。')
    root = Path(__file__).resolve().parent
    folder = root/'.easy-venv'
    python = folder/('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
    requirements = root/'requirements.txt'
    desktop_requirements = root/'requirements-desktop.txt'
    fingerprint = hashlib.sha256(requirements.read_bytes()+desktop_requirements.read_bytes()).hexdigest()
    marker = folder/'requirements.sha256'
    try:
        if not python.exists():
            print('首次启动：正在准备运行环境……', flush=True)
            venv.EnvBuilder(with_pip=True).create(folder)
        installed = marker.read_text(encoding='utf-8').strip() if marker.exists() else ''
        if installed != fingerprint:
            print('正在安装或更新客户端依赖……', flush=True)
            subprocess.run([str(python), '-m', 'pip', 'install', '-r', str(requirements)], check=True)
            if sys.platform in ('win32', 'darwin'):
                try:
                    subprocess.run([str(python), '-m', 'pip', 'install', '-r', str(desktop_requirements)], check=True)
                except subprocess.CalledProcessError:
                    print('原生窗口依赖暂不可用，将使用本机浏览器界面。', flush=True)
            marker.write_text(fingerprint, encoding='utf-8')
        env = dict(os.environ,PYTHONUTF8='1')
        env['PATH'] = str(python.parent)+os.pathsep+env.get('PATH', '')
        result = subprocess.run([str(python), str(root/'desktop.py'), *sys.argv[1:]], env=env)
        return result.returncode
    except (OSError, subprocess.CalledProcessError) as error:
        print('启动失败。请检查 Python、venv 和网络。错误类型：'+type(error).__name__, flush=True)
        return 1
    except KeyboardInterrupt:
        return 0


if __name__ == '__main__':
    raise SystemExit(main())
