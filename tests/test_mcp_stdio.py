"""Real SDK stdio clients, real worker subprocess, SQLite and HTTP GUI."""
import asyncio
import base64
import importlib.util
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import requests
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from studio_client import StudioClient,SharedBackend
from easy_gateway import create_server

HAS_MCP=importlib.util.find_spec('mcp') is not None


def wait_worker(client,deadline=10):
    end=time.monotonic()+deadline
    while time.monotonic()<end:
        try:client.ensure(force=True);return
        except RuntimeError:time.sleep(.05)
    raise AssertionError('Worker failed to start')


def stop_worker(client,process=None):
    try:client.shutdown()
    except RuntimeError:pass
    if process:
        try:process.wait(timeout=8)
        except subprocess.TimeoutExpired:process.kill();process.wait(timeout=5)
    else:
        end=time.monotonic()+8
        while (client.directory/'connection.json').exists() and time.monotonic()<end:time.sleep(.05)
        # Detached worker may be between releasing its lock and closing the log.
        time.sleep(.1)
    client.close()


@unittest.skipUnless(HAS_MCP,'Install requirements-mcp.txt to test the MCP transport')
class MCPStdioTest(unittest.TestCase):
    def parameters(self,config_path):
        from mcp.client.stdio import StdioServerParameters
        return StdioServerParameters(command=sys.executable,args=[str(ROOT/'mcp_server.py')],
            env={'EASY_CONFIG_PATH':str(config_path),'PYTHONUTF8':'1'})

    def test_stdio_gui_and_backend_share_queue_and_results_survive_reconnect(self):
        from mcp import Client
        with tempfile.TemporaryDirectory() as d,patch.dict(os.environ,{'EASY_CONFIG_PATH':str(Path(d)/'settings.json')}):
            directory=Path(d)/'studio';config_path=Path(d)/'settings.json'
            env=dict(os.environ,KAGGLE_USERNAME='owner',KAGGLE_API_TOKEN='KGAT_fixture_secret',PYTHONUTF8='1')
            process=subprocess.Popen([sys.executable,str(ROOT/'tests/fixtures/fake_worker.py'),str(directory)],
                env=env,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            service=StudioClient(directory,autostart=False);wait_worker(service)
            gui=create_server(0);thread=threading.Thread(target=gui.serve_forever);thread.start()
            address=f'http://127.0.0.1:{gui.server_port}'
            headers={'X-Local-Session':re.search("const nonce='([^']+)'",requests.get(address,timeout=5).text).group(1)}
            configured=requests.post(address+'/configure',headers=headers,
                json={'mode':'account','user':'owner','token':'KGAT_fixture_secret','remember':True},timeout=5)
            self.assertEqual(configured.status_code,200,configured.text)
            self.assertEqual(json.loads(config_path.read_text())['token'],'KGAT_fixture_secret')
            async def exercise():
                async with Client(self.parameters(config_path)) as client:
                    tools=await client.list_tools()
                    self.assertEqual({t.name for t in tools.tools},{'studio_status','generate_image','get_job','get_image'})
                    for tool in tools.tools:
                        self.assertNotIn('token',json.dumps(tool.input_schema).lower())
                    status=await client.call_tool('studio_status',{})
                    self.assertTrue(status.structured_content['configured'])
                    submitted=await client.call_tool('generate_image',{'prompt':'fox','request_id':'mcp-one'})
                    self.assertFalse(submitted.is_error)
                    job_id=submitted.structured_content['job_id']
                    duplicate=await client.call_tool('generate_image',{'prompt':'fox','request_id':'mcp-one'})
                    self.assertEqual(duplicate.structured_content['job_id'],job_id)
                    self.assertTrue(duplicate.structured_content['deduplicated'])
                    second=requests.post(address+'/jobs',headers=headers,json={'prompt':'teapot','request_id':'gui-two'},timeout=5)
                    self.assertEqual(second.status_code,202,second.text)
                    second_id=second.json()['job_id']
                # A new MCP process can query the exact same job. Neither the
                # worker nor the submitted remote operation is restarted.
                async with Client(self.parameters(config_path),mode='legacy') as client:
                    deadline=time.monotonic()+12
                    while time.monotonic()<deadline:
                        polled=await client.call_tool('get_job',{'job_id':job_id})
                        self.assertNotIn('KGAT_fixture_secret',json.dumps(polled.model_dump()))
                        if polled.structured_content['state']=='complete':break
                        await asyncio.sleep(.1)
                    self.assertEqual(polled.structured_content['state'],'complete')
                    image=await client.call_tool('get_image',{'job_id':job_id})
                    self.assertFalse(image.is_error)
                    png=base64.b64decode(next(c.data for c in image.content if c.type=='image'))
                    self.assertTrue(png.startswith(b'\x89PNG\r\n\x1a\n'))
                    self.assertEqual(image.structured_content['usage']['model_load_count'],1)
                    bad=await client.call_tool('get_job',{'job_id':'../../settings.json'})
                    self.assertTrue(bad.is_error)
                result=service.wait(second_id,timeout=10)[1]
                self.assertEqual(result['generation_number'],2)
                self.assertTrue(result['model_reused'])
                # GUI restart must still find the old job, not submit another.
                return job_id
            try:
                job_id=asyncio.run(exercise())
                connection=json.loads((directory/'connection.json').read_text())
                self.assertEqual(requests.get(connection['url']+'/status',timeout=5).status_code,401)
                auth={'Authorization':'Bearer '+connection['token'],'Origin':'https://evil.test'}
                self.assertEqual(requests.get(connection['url']+'/status',headers=auth,timeout=5).status_code,403)
                gui.shutdown();thread.join();gui.server_close()
                replacement=create_server(0);new_thread=threading.Thread(target=replacement.serve_forever);new_thread.start()
                try:
                    base=f'http://127.0.0.1:{replacement.server_port}'
                    nonce=re.search("const nonce='([^']+)'",requests.get(base,timeout=5).text).group(1)
                    recovered=requests.get(base+'/jobs/'+job_id,headers={'X-Local-Session':nonce},timeout=5)
                    self.assertEqual(recovered.json()['state'],'complete')
                    self.assertEqual(recovered.json()['result']['usage']['generation_number'],1)
                finally:replacement.shutdown();new_thread.join();replacement.server_close()
                # This is the same adapter now used by ComfyUI; no GUI remains.
                backend=SharedBackend({'mode':'account','user':'owner','token':'KGAT_fixture_secret'},client=service)
                raw,usage=backend.generate('third',42,20)
                self.assertEqual(usage['generation_number'],3);self.assertTrue(usage['model_reused'])
                async def model_choice():
                    async with Client(self.parameters(config_path)) as client:
                        submitted=await client.call_tool('generate_image',{'prompt':'catgirl','request_id':'mcp-lightning','model_id':'sdxl-lightning'})
                        self.assertFalse(submitted.is_error)
                        selected=submitted.structured_content['job_id']
                        for _ in range(60):
                            result=await client.call_tool('get_job',{'job_id':selected})
                            if result.structured_content['state']=='complete':break
                            await asyncio.sleep(.1)
                        self.assertEqual(result.structured_content['usage']['model_id'],'sdxl-lightning')
                        self.assertEqual(result.structured_content['usage']['steps'],4)
                        invalid=await client.call_tool('generate_image',{'prompt':'catgirl','request_id':'bad-model','model_id':'unknown'})
                        self.assertTrue(invalid.is_error)
                asyncio.run(model_choice())
            finally:
                if thread.is_alive():gui.shutdown();thread.join();gui.server_close()
                stop_worker(service,process)

    def test_auto_started_worker_survives_stdio_process_exit(self):
        from mcp import Client
        with tempfile.TemporaryDirectory() as d:
            config=Path(d)/'settings.json'
            async def exercise():
                async with Client(self.parameters(config)) as client:
                    result=await client.call_tool('studio_status',{})
                    self.assertFalse(result.is_error)
                    self.assertFalse(result.structured_content['configured'])
            service=StudioClient(Path(d)/'studio',autostart=False)
            try:
                asyncio.run(exercise())
                self.assertFalse(service.status()['configured'])
            finally:stop_worker(service)


if __name__=='__main__':unittest.main()
