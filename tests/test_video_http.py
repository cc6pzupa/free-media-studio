import json
import os
import re
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch
import requests
from easy_gateway import create_server
from studio_core import StudioCore,private_write
from studio_worker import create_worker_server
from tests.test_video import Backend


class VideoHTTPTest(unittest.TestCase):
    def test_authenticated_queue_status_and_mp4_download_share_the_worker(self):
        with tempfile.TemporaryDirectory() as d,patch.dict(os.environ,{'EASY_CONFIG_PATH':str(Path(d)/'settings.json')}):
            with patch('studio_core.load_config',return_value=None):core=StudioCore(Path(d)/'studio',backend_factory=Backend)
            core.configure({'mode':'account','user':'owner','token':'fixture'});core.start()
            worker=create_worker_server(core,'x'*32)
            private_write(Path(d)/'studio/connection.json',json.dumps({'url':f'http://127.0.0.1:{worker.server_port}',
                'token':'x'*32,'protocol':1}).encode())
            worker_thread=threading.Thread(target=worker.serve_forever);worker_thread.start()
            gui=create_server(0);thread=threading.Thread(target=gui.serve_forever);thread.start()
            base=f'http://127.0.0.1:{gui.server_port}'
            try:
                page=requests.get(base+'/video',timeout=5)
                nonce=re.search("const nonce='([^']+)'",page.text).group(1)
                headers={'X-Local-Session':nonce}
                self.assertEqual(len(requests.get(base+'/video/models',timeout=5).json()['models']),3)
                body={'prompt':'cat','request_id':'http-video'}
                self.assertEqual(requests.post(base+'/video/jobs',json=body,timeout=5).status_code,401)
                posted=requests.post(base+'/video/jobs',headers=headers,json=body,timeout=5)
                self.assertEqual(posted.status_code,202,posted.text);job=posted.json()['job_id']
                duplicate=requests.post(base+'/video/jobs',headers=headers,json=body,timeout=5)
                self.assertTrue(duplicate.json()['deduplicated'])
                deadline=time.monotonic()+5
                while time.monotonic()<deadline:
                    status=requests.get(base+'/video/jobs/'+job,headers=headers,timeout=5).json()
                    if status['state']=='complete':break
                    time.sleep(.02)
                self.assertTrue(status['video_ready'])
                self.assertEqual(requests.get(base+'/video/jobs/'+job+'/content',timeout=5).status_code,401)
                content=requests.get(base+'/video/jobs/'+job+'/content',headers=headers,timeout=5)
                self.assertEqual(content.headers['Content-Type'],'video/mp4');self.assertEqual(content.content[4:8],b'ftyp')
                api_key=requests.get(base+'/configuration',headers=headers,timeout=5).json()['api_key']
                api_headers={'Authorization':'Bearer '+api_key}
                self.assertEqual(requests.get(base+'/v1/videos/jobs/'+job,headers=api_headers,timeout=5).status_code,200)
                self.assertEqual(requests.get(base+'/v1/videos/jobs/'+job+'/content',headers=api_headers,timeout=5).content,content.content)
                self.assertEqual(requests.get(base+'/v1/videos/jobs/'+job,timeout=5).status_code,401)
                self.assertEqual(requests.get(base+'/video/jobs/../../settings.json',headers=headers,timeout=5).status_code,404)
            finally:
                gui.shutdown();thread.join();gui.server_close()
                worker.shutdown();worker_thread.join();worker.server_close();core.close()
