import contextlib
import io
import json
import os
import re
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import requests
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from batch_backend import kernel_status
from easy_gateway import create_server
from session_backend import bootstrap_code
from session_runtime import JupyterTransport, ModelSession
import session_backend


class ReuseTest(unittest.TestCase):
    def test_bootstrap_loads_once_even_when_client_reconnects(self):
        source = "_loads += 1\npipe = object()\nencoder = object()\ndef generate_image(*args, **kwargs): pass\n"
        scope = {'_loads':0}
        with contextlib.redirect_stdout(io.StringIO()):
            exec(bootstrap_code('owner/assets',source,install=False),scope)
            identity=scope['_COMFY_QWEN_STATE']['runtime_id']
            exec(bootstrap_code('owner/assets',"raise AssertionError('must not reload')",install=False),scope)
        self.assertEqual(scope['_loads'],1)
        self.assertEqual(scope['_COMFY_QWEN_STATE']['runtime_id'],identity)
        self.assertEqual(scope['_COMFY_QWEN_STATE']['load_count'],1)
        self.assertFalse(scope['_COMFY_QWEN_STATE']['loaded_this_request'])

    def test_adopts_already_loaded_notebook_without_loading_again(self):
        scope={'pipe':object(),'encoder':object(),'generate_image':lambda *a:None,
               'source':Path('/kaggle/input/datasets/owner/assets/versions/1')}
        with contextlib.redirect_stdout(io.StringIO()):
            exec(bootstrap_code('owner/assets',"raise AssertionError('must not load')",install=False),scope)
        self.assertTrue(scope['_COMFY_QWEN_STATE']['adopted'])
        self.assertFalse(scope['_COMFY_QWEN_STATE']['loaded_this_request'])

    def test_different_dataset_does_not_load_second_model(self):
        scope={'pipe':object(),'encoder':object(),'generate_image':lambda *a:None,
               '_COMFY_QWEN_STATE':{'dataset':'owner/other'}}
        with self.assertRaisesRegex(RuntimeError,'different dataset'):
            exec(bootstrap_code('owner/assets',"raise AssertionError('must not load')",install=False),scope)

    def test_attaches_notebook_and_close_preserves_remote_kernel(self):
        transport=JupyterTransport('https://example.test/proxy')
        transport.http.close()
        transport.http=Mock()
        transport.http.get.return_value.json.return_value=[{'type':'notebook','kernel':{'id':'existing-kernel'}}]
        transport.start()
        self.assertEqual(transport.kernel_id,'existing-kernel')
        transport.http.post.assert_not_called()
        transport.close()
        transport.http.delete.assert_not_called()
        transport.http.close.assert_called_once()

    def test_no_fixed_reload_deadline_for_live_notebook(self):
        now=[0];created=[]
        class Transport:
            def start(self):pass
            def execute(self,code,timeout):return code
            def close(self):pass
        def factory():
            t=Transport();created.append(t);return t
        session=ModelSession(factory,'load',lifetime_seconds=None,clock=lambda:now[0])
        try:
            self.assertEqual(session.execute('one'),'one')
            now[0]=10**9
            self.assertEqual(session.execute('two'),'two')
            self.assertEqual(len(created),1)
        finally:session.close()

    def test_status_uses_status_field_not_account_name(self):
        self.assertEqual(kernel_status('error-user/job has status "KernelWorkerStatus.RUNNING"'),'RUNNING')
        self.assertEqual(kernel_status('complete-user/job has status "KernelWorkerStatus.QUEUED"'),'QUEUED')
        self.assertEqual(kernel_status('owner/job has status "KernelWorkerStatus.CANCEL_ACKNOWLEDGED"'),'CANCEL_ACKNOWLEDGED')

    def test_session_mode_persists_without_exposing_connection_or_falling_back(self):
        class Session:
            mode='session'
            def __init__(self,user,dataset,jupyter_url,kernel_id=None):
                self.user,self.dataset,self.jupyter_url,self.kernel_id=user,dataset,jupyter_url,kernel_id
            def generate(self,*args):return b'png',{'backend':'session','model_reused':True,'request_seconds':1}
            def close(self):pass
        def reject_batch(*args):raise AssertionError('must not use batch backend')
        with tempfile.TemporaryDirectory() as d, patch.dict(os.environ,{'EASY_CONFIG_PATH':d+'/settings.json'}),patch.object(session_backend,'SessionBackend',Session):
            server=create_server(0,reject_batch);thread=threading.Thread(target=server.serve_forever);thread.start()
            url=f'http://127.0.0.1:{server.server_port}'
            try:
                nonce=re.search("const nonce='([^']+)'",requests.get(url,timeout=5).text).group(1)
                headers={'X-Local-Session':nonce}
                r=requests.post(url+'/configure',headers=headers,json={'mode':'session','user':'owner','dataset':'owner/assets','jupyter_url':'https://example.test/private-connection','remember':True},timeout=5)
                self.assertEqual(r.status_code,200)
                key=r.json()['api_key']
                cfg=requests.get(url+'/configuration',headers=headers,timeout=5)
                self.assertNotIn('private-connection',cfg.text)
                self.assertEqual(cfg.json()['mode'],'session')
                for _ in range(2):
                    r=requests.post(url+'/v1/images/generations',headers={'Authorization':'Bearer '+key},json={'prompt':'cat'},timeout=5)
                    self.assertEqual(r.status_code,200)
                    self.assertTrue(r.json()['usage']['model_reused'])
            finally:server.shutdown();thread.join();server.server_close()
            second=create_server(0,reject_batch);worker=threading.Thread(target=second.serve_forever);worker.start()
            try:
                url=f'http://127.0.0.1:{second.server_port}'
                nonce=re.search("const nonce='([^']+)'",requests.get(url,timeout=5).text).group(1)
                self.assertEqual(requests.get(url+'/configuration',headers={'X-Local-Session':nonce},timeout=5).json()['mode'],'session')
            finally:second.shutdown();worker.join();second.server_close()

if __name__=='__main__':unittest.main()
