"""Exercise installed/frozen assets, GUI routes and an independent worker, offline."""
import asyncio
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from urllib.request import Request,build_opener,ProxyHandler

urlopen=build_opener(ProxyHandler({})).open


def check(executable):
    executable=str(Path(executable).resolve())
    with tempfile.TemporaryDirectory() as folder:
        root=Path(folder);state=root/'studio'
        env=dict(os.environ,EASY_CONFIG_PATH=str(root/'settings.json'),PYINSTALLER_RESET_ENVIRONMENT='1')
        subprocess.run([executable,'--self-test'],env=env,check=True,timeout=60)
        # A GUI gallery read must start its own durable worker without a GPU.
        gui=subprocess.Popen([executable,'--no-open','--port','18787'],env=env,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        worker=None
        async def mcp_check():
            from mcp import Client
            from mcp.client.stdio import StdioServerParameters
            cli=Path(executable).with_name('cc-media-cli'+('.exe' if os.name=='nt' else ''))
            async with Client(StdioServerParameters(command=str(cli),args=['--mcp'],env=env)) as client:
                tools=await client.list_tools()
                assert len(tools.tools)==6
        try:
            asyncio.run(asyncio.wait_for(mcp_check(),timeout=30))
            deadline=time.monotonic()+45
            while time.monotonic()<deadline:
                try:
                    with urlopen('http://127.0.0.1:18787/gallery',timeout=1) as r:page=r.read().decode()
                    break
                except OSError:
                    if gui.poll() is not None:raise RuntimeError('Frozen GUI exited early')
                    time.sleep(.2)
            else:raise TimeoutError('Frozen GUI did not start')
            nonce=re.search("const nonce='([^']+)'",page).group(1)
            with urlopen(Request('http://127.0.0.1:18787/gallery/items',headers={'X-Local-Session':nonce}),timeout=30) as r:
                assert json.load(r)['items']==[]
            for route in ('/','/video','/kaggle_video.ipynb'):
                with urlopen('http://127.0.0.1:18787'+route,timeout=5) as r:assert r.status==200
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
            if gui.poll() is None:gui.terminate();gui.wait(timeout=10)
            if worker is None and (state/'connection.json').exists():worker=json.loads((state/'connection.json').read_text())
            if worker:
                try:
                    req=Request(worker['url']+'/shutdown',data=b'{}',headers={'Authorization':'Bearer '+worker['token'],'Content-Type':'application/json'})
                    with urlopen(req,timeout=5) as r:assert json.load(r)['stopped']
                    deadline=time.monotonic()+10
                    while (state/'connection.json').exists() and time.monotonic()<deadline:time.sleep(.1)
                    assert not (state/'connection.json').exists(),'Worker did not stop'
                except OSError:raise RuntimeError('Frozen worker cleanup failed') from None
    print('Frozen GUI, gallery, resources and independent worker passed')

if __name__=='__main__':check(sys.argv[1])
