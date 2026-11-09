import base64, json, os, sys, threading, unittest
from pathlib import Path
from unittest.mock import patch
from http.server import ThreadingHTTPServer
import requests
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import gateway

class GatewayTest(unittest.TestCase):
    def test_http_contract(self):
        ready=threading.Event(); servers=[]; codes=[]
        class FakeSession:
            def __init__(self,*a,**kw):pass
            def execute(self,code):
                codes.append(code)
                return 'COMFY_RESULT:'+json.dumps({'png':base64.b64encode(b'test-image-bytes').decode(),'stats':{'generation_seconds':1}})+'\n'
            def close(self):pass
        def server_factory(addr,handler):
            server=ThreadingHTTPServer((addr[0],0),handler);servers.append(server);ready.set();return server
        env={'KAGGLE_JUPYTER_URL':'http://127.0.0.1:1/?token=test','IMAGE_API_KEY':'test-secret','KAGGLE_DATASET':'test-user/test-assets'}
        with patch.dict(os.environ,env),patch.object(gateway,'ModelSession',FakeSession),patch.object(gateway,'ThreadingHTTPServer',server_factory):
            worker=threading.Thread(target=gateway.main);worker.start();self.assertTrue(ready.wait(5))
            url=f'http://127.0.0.1:{servers[0].server_port}/v1/images/generations'
            try:
                self.assertEqual(requests.post(url,json={'prompt':'cat'},timeout=5).status_code,401)
                headers={'Authorization':'Bearer test-secret'}
                self.assertEqual(requests.post(url,headers=headers,json={'prompt':'cat','steps':0},timeout=5).status_code,400)
                prompt="Cat cartoon\n'; raise Exception('injection') #"
                r=requests.post(url,headers=headers,json={'prompt':prompt,'steps':20},timeout=5)
                self.assertEqual(r.status_code,200)
                self.assertEqual(base64.b64decode(r.json()['data'][0]['b64_json']),b'test-image-bytes')
                # Execute generated code against a harmless stub to verify quoting.
                class Image:
                    def save(self,buf,format):buf.write(b'png')
                seen=[]
                def generate(p,**kw):seen.append(p);return Image(),{}
                scope={'generate_image':generate, '_COMFY_QWEN_STATE':{'runtime_id':'test-runtime','load_count':1,'loaded_this_request':False,'generations':0}}
                exec(codes[0],scope)
                self.assertEqual(seen,[prompt])
            finally:servers[0].shutdown();worker.join(5)

if __name__=='__main__':unittest.main()
