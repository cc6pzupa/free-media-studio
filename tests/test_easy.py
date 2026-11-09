import ast, json, os, re, sys, tempfile, threading, time, unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import requests
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from batch_backend import BatchBackend
from easy_gateway import create_server

class BatchTest(unittest.TestCase):
    def test_submission_and_exact_output(self):
        observed={}; prompt="cat\\test\n'; print('still prompt') #"
        def run(args,**kw):
            self.assertEqual(kw['env']['KAGGLE_API_TOKEN'],'private-token')
            if args[2]=='push':
                folder=Path(args[4]);nb=json.loads((folder/'job.ipynb').read_text())
                code=''.join(nb['cells'][3]['source']);ast.parse(code)
                assignment=next(n for n in ast.parse(code).body if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','')=='PROMPT')
                self.assertEqual(ast.literal_eval(assignment.value),prompt)
                meta=json.loads((folder/'kernel-metadata.json').read_text())
                self.assertTrue(meta['is_private']);self.assertEqual(meta['dataset_sources'],['owner/assets'])
                observed['image']=re.search(r'result_[a-f0-9]+\.png',code).group()
                return SimpleNamespace(returncode=0,stdout='submitted')
            if args[2]=='status':return SimpleNamespace(returncode=0,stdout='complete')
            out=Path(args[5]);out.mkdir();(out/observed['image']).write_bytes(b'expected-image');(out/'result-old.png').write_bytes(b'stale')
            return SimpleNamespace(returncode=0,stdout='downloaded')
        image,stats=BatchBackend('me','owner/assets','private-token',runner=run,poll_seconds=0).generate(prompt)
        self.assertEqual(image,b'expected-image');self.assertIn('request_seconds',stats)
    def test_cli_failure_hides_output(self):
        def run(*a,**kw):return SimpleNamespace(returncode=1,stdout='private-token',stderr='private-token')
        with self.assertRaises(RuntimeError) as caught:BatchBackend('me','owner/assets','private-token',runner=run).generate('cat')
        self.assertNotIn('private-token',str(caught.exception))

class EasyHTTPTest(unittest.TestCase):
    def test_setup_generation_and_remember(self):
        class FakeBackend:
            def __init__(self,user,dataset,token):self.user,self.dataset,self.token=user,dataset,token
            def generate(self,prompt,seed,steps):return b'fake-png',{'request_seconds':1}
        with tempfile.TemporaryDirectory() as d,patch.dict(os.environ,{'EASY_CONFIG_PATH':d+'/settings.json'}):
            server=create_server(0,FakeBackend);worker=threading.Thread(target=server.serve_forever);worker.start()
            url=f'http://127.0.0.1:{server.server_port}'
            try:
                page=requests.get(url,timeout=5).text;nonce=re.search("const nonce='([^']+)'",page).group(1)
                headers={'X-Local-Session':nonce}
                bad=requests.post(url+'/configure',headers={'Origin':'https://evil.test'},json={},timeout=5)
                self.assertEqual(bad.status_code,403)
                cfg=requests.post(url+'/configure',headers=headers,json={'user':'me','dataset':'owner/assets','token':'private-token','remember':True},timeout=5)
                self.assertEqual(cfg.status_code,200);key=cfg.json()['api_key']
                public=requests.get(url+'/configuration',headers=headers,timeout=5).text
                self.assertNotIn('private-token',public)
                if os.name != 'nt':
                    self.assertEqual(os.stat(d+'/settings.json').st_mode&0o777,0o600)
                auth={'Authorization':'Bearer '+key}
                self.assertEqual(requests.post(url+'/v1/images/generations',json={'prompt':'cat'},timeout=5).status_code,401)
                result=requests.post(url+'/v1/images/generations',headers=auth,json={'prompt':'cat'},timeout=5)
                self.assertEqual(result.status_code,200);self.assertIn('b64_json',result.json()['data'][0])
                job=requests.post(url+'/jobs',headers=headers,json={'prompt':'cartoon cat'},timeout=5).json()
                for _ in range(20):
                    state=requests.get(url+'/jobs/'+job['job_id'],headers=headers,timeout=5).json()
                    if state['state']=='complete':break
                    time.sleep(.01)
                self.assertEqual(state['state'],'complete')
            finally:server.shutdown();worker.join();server.server_close()
            second=create_server(0,FakeBackend);w=threading.Thread(target=second.serve_forever);w.start()
            try:
                url=f'http://127.0.0.1:{second.server_port}'
                nonce=re.search("const nonce='([^']+)'",requests.get(url,timeout=5).text).group(1)
                cfg=requests.get(url+'/configuration',headers={'X-Local-Session':nonce},timeout=5).json()
                self.assertTrue(cfg['configured']);self.assertEqual(cfg['api_key'],key)
            finally:second.shutdown();w.join();second.server_close()

if __name__=='__main__':unittest.main()
