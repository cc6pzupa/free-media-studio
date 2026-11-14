"""Exercise installed/frozen assets, GUI routes and an independent worker, offline."""
import asyncio
import json
import os
import re
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from urllib.request import Request,build_opener,ProxyHandler

urlopen=build_opener(ProxyHandler({})).open


def wait_for_released_logs(folder,timeout=20):
    """Wait until no process holds a log open.

    The worker removes connection.json before its process exits. Windows
    refuses to rename (or delete) a file another process still has open, so
    the temporary folder cannot be removed until the worker has exited."""
    deadline=time.monotonic()+timeout
    for log in Path(folder).rglob('*.log'):
        moved=log.with_name(log.name+'.released')
        while True:
            try:
                log.rename(moved);moved.rename(log);break
            except OSError:
                if time.monotonic()>=deadline:raise RuntimeError('Worker still holds '+log.name) from None
                time.sleep(.2)


def check(executable):
    executable=str(Path(executable).resolve())
    with tempfile.TemporaryDirectory() as folder:
        root=Path(folder);state=root/'studio'
        env=dict(os.environ,EASY_CONFIG_PATH=str(root/'settings.json'),PYINSTALLER_RESET_ENVIRONMENT='1',CC_MEDIA_DIAGNOSTICS='1')
        subprocess.run([executable,'--self-test'],env=env,check=True,timeout=60)
        with socket.socket() as reservation:
            reservation.bind(('127.0.0.1',0))
            port=reservation.getsockname()[1]
        address=f'http://127.0.0.1:{port}'
        # A GUI gallery read must start its own durable worker without a GPU.
        gui=None;worker=None
        async def mcp_check():
            from mcp import Client
            from mcp.client.stdio import StdioServerParameters
            cli=Path(executable).with_name('cc-media-cli'+('.exe' if os.name=='nt' else ''))
            async with Client(StdioServerParameters(command=str(cli),args=['--mcp'],env=env)) as client:
                tools=await client.list_tools()
                assert len(tools.tools)==7
        try:
            asyncio.run(asyncio.wait_for(mcp_check(),timeout=30))
            with (root/'gui-launch.log').open('wb') as output:
                gui=subprocess.Popen([executable,'--no-open','--port',str(port)],env=env,stdout=output,stderr=output)
            deadline=time.monotonic()+45
            while time.monotonic()<deadline:
                try:
                    with urlopen(address+'/gallery',timeout=1) as r:page=r.read().decode()
                    break
                except OSError:
                    if gui.poll() is not None:raise RuntimeError('Frozen GUI exited early: '+str(gui.returncode))
                    time.sleep(.2)
            else:raise TimeoutError('Frozen GUI did not start')
            nonce=re.search("const nonce='([^']+)'",page).group(1)
            with urlopen(Request(address+'/gallery/items',headers={'X-Local-Session':nonce}),timeout=30) as r:
                assert json.load(r)['items']==[]
            for route in ('/','/video','/kaggle_video.ipynb'):
                with urlopen(address+route,timeout=5) as r:assert r.status==200
            worker=json.loads((state/'connection.json').read_text())
            gui.terminate();gui.wait(timeout=10)
            # Worker survives parent GUI exit, and accepts authenticated reads.
            with urlopen(Request(worker['url']+'/status',headers={'Authorization':'Bearer '+worker['token']}),timeout=5) as r:
                assert json.load(r)['configured'] is False
        except Exception:
            for path in root.rglob('*.log'):
                print(path.name+':\n'+path.read_text(encoding='utf-8',errors='replace'),flush=True)
            raise
        finally:
            if gui is not None and gui.poll() is None:gui.terminate();gui.wait(timeout=10)
            if worker is None and (state/'connection.json').exists():worker=json.loads((state/'connection.json').read_text())
            if worker:
                try:
                    req=Request(worker['url']+'/shutdown',data=b'{}',headers={'Authorization':'Bearer '+worker['token'],'Content-Type':'application/json'})
                    with urlopen(req,timeout=5) as r:assert json.load(r)['stopped']
                    deadline=time.monotonic()+10
                    while (state/'connection.json').exists() and time.monotonic()<deadline:time.sleep(.1)
                    assert not (state/'connection.json').exists(),'Worker did not stop'
                    wait_for_released_logs(state)
                except OSError:raise RuntimeError('Frozen worker cleanup failed') from None
    print('Frozen GUI, gallery, resources and independent worker passed')

if __name__=='__main__':check(sys.argv[1])
