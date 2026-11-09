import hashlib,json,sys,tempfile,threading,unittest
from pathlib import Path
from unittest.mock import patch
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import requests
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from account_backend import AccountBackend,KaggleApiError
from asset_manager import download_file,safe_relative

class AutomaticTest(unittest.TestCase):
    def test_expired_connection_discards_old_operation_and_provisions_again(self):
        with tempfile.TemporaryDirectory() as d:
            b=AccountBackend('owner','KGAT_test',cache_dir=d)
            old='https://kkb-production.jupyter-proxy.kaggle.net/k/1/expired/proxy'
            new='https://kkb-production.jupyter-proxy.kaggle.net/k/2/current/proxy'
            b._save({'jupyter_url':old,'operation':'operations/old'})
            from unittest.mock import Mock
            response=Mock(ok=False,status_code=404)
            calls=[]
            def api(service,method,body):
                calls.append((method,body))
                if method=='CreateKernelSession':return {'name':'operations/new'}
                return {'done':True,'response':{'jupyterUrl':new}}
            with patch('account_backend.requests.get',return_value=response),patch.object(b,'_ensure_notebook'),patch.object(b,'_remember_operation'),patch.object(b,'_call',side_effect=api):
                self.assertEqual(b._ensure_session(lambda e:None),new)
            self.assertEqual([v[0] for v in calls],['CreateKernelSession','GetOperation'])
            self.assertEqual(calls[-1][1],{'name':'operations/new'});b.close()

    def test_uses_api_base_not_frontend_handoff_and_retains_operation(self):
        with tempfile.TemporaryDirectory() as d:
            b=AccountBackend('owner','KGAT_test',cache_dir=d)
            url='https://kkb-production.jupyter-proxy.kaggle.net/k/42/secret/proxy'
            def api(service,method,body):
                if method=='CreateKernelSession':return {'name':'operations/existing'}
                return {'done':True,'response':{'jupyterUrl':url,'tokenizedJupyterUrl':'https://kkb-production.jupyter-proxy.kaggle.net/?token=bad-base'}}
            with patch.object(b,'_ensure_notebook'),patch.object(b,'_call',side_effect=api):
                self.assertEqual(b._ensure_session(lambda e:None),url)
            saved=json.loads(b.cache_file.read_text());self.assertEqual(saved['operation'],'operations/existing')
            self.assertNotIn('KGAT_test',b.cache_file.read_text());b.close()

    def test_restart_resumes_operation_without_duplicate_session(self):
        with tempfile.TemporaryDirectory() as d:
            b=AccountBackend('owner','KGAT_test',cache_dir=d);b._save({'operation':'operations/pending'})
            url='https://kkb-production.jupyter-proxy.kaggle.net/k/42/secret/proxy'
            with patch.object(b,'_ensure_notebook'),patch.object(b,'_remember_operation'),patch.object(b,'_call',return_value={'done':True,'response':{'jupyterUrl':url}}) as api:
                self.assertEqual(b._ensure_session(lambda e:None),url)
                self.assertEqual(api.call_args.args[1],'GetOperation');self.assertEqual(api.call_count,1)
            b.close()

    def test_new_private_notebook_has_no_dataset_dependency_or_token(self):
        with tempfile.TemporaryDirectory() as d:
            b=AccountBackend('owner','KGAT_test',cache_dir=d);created=[]
            def api(service,method,body):
                if method=='GetKernel':raise KaggleApiError(403)
                if method=='ListKernels':return {'kernels':[]}
                created.append(body);return {'ref':b.slug}
            with patch.object(b,'_call',side_effect=api):b._ensure_notebook()
            self.assertTrue(created[0]['isPrivate']);self.assertEqual(created[0]['datasetDataSources'],[])
            self.assertEqual(created[0]['kernelExecutionType'],'QUICK_SAVE');self.assertNotIn(b.token,created[0]['text']);b.close()

    def test_never_overwrites_an_unmanaged_notebook(self):
        with tempfile.TemporaryDirectory() as d:
            b=AccountBackend('owner','KGAT_test',cache_dir=d)
            with patch.object(b,'_call',return_value={'blob':{'source':'valuable notebook'}}) as api:
                with self.assertRaisesRegex(RuntimeError,'不属于本程序'):b._ensure_notebook()
                self.assertEqual(api.call_count,1)
            b.close()

class DownloadTest(unittest.TestCase):
    def download(self,ignore=False):
        content=b'weights-'*10000+b'end';offset=4000;seen=[]
        class Handler(BaseHTTPRequestHandler):
            def log_message(self,*a):pass
            def do_GET(self):
                value=self.headers.get('Range');seen.append(value)
                start=int(value.split('=')[1].split('-')[0]) if value and not ignore else 0
                self.send_response(206 if start else 200);self.send_header('Content-Length',str(len(content)-start))
                if start:self.send_header('Content-Range',f'bytes {start}-{len(content)-1}/{len(content)}')
                self.end_headers();self.wfile.write(content[start:])
        server=ThreadingHTTPServer(('127.0.0.1',0),Handler);worker=threading.Thread(target=server.serve_forever);worker.start()
        try:
            with tempfile.TemporaryDirectory() as d:
                file=Path(d)/'weight';file.with_name('weight.partial').write_bytes(content[:offset]);values=[]
                download_file(requests.Session(),f'http://127.0.0.1:{server.server_port}/weight',file,len(content),hashlib.sha256(content).hexdigest(),values.append)
                self.assertEqual(file.read_bytes(),content);self.assertEqual(seen,[f'bytes={offset}-'])
                self.assertEqual(values[-1],len(content));self.assertFalse(file.with_name('weight.partial').exists())
        finally:server.shutdown();worker.join();server.server_close()
    def test_verified_resume(self):self.download()
    def test_server_ignoring_range_restarts_cleanly(self):self.download(ignore=True)
    def test_path_traversal_is_rejected(self):
        for name in ['../secret','/absolute','a\\..\\b']:
            with self.subTest(name=name),self.assertRaises(ValueError):safe_relative(name)

if __name__=='__main__':unittest.main()
