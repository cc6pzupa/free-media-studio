import json
import os
import re
import sqlite3
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import requests
from easy_gateway import create_server
from studio_core import JobStore,StudioCore,private_write
from studio_worker import create_worker_server


class GalleryTest(unittest.TestCase):
    def seed(self,store,prompt='sea turtle',video=False,request='one'):
        params={'prompt':prompt,'seed':42,'steps':30,'model_id':'wan22-ti2v-5b' if video else 'z-image-turbo'}
        if video:params['media_type']='video'
        job,_=store.submit(params,'private-config-hash',request)
        store.update(job['job_id'],state='complete',stats={'model_id':params['model_id']})
        private_write(store.outputs/(job['job_id']+('.mp4' if video else '.png')),b'fixture')
        return job['job_id'],params

    def test_migration_filters_pagination_and_reopen(self):
        with tempfile.TemporaryDirectory() as d:
            # Existing pre-gallery database, with the exact old schema.
            db=sqlite3.connect(Path(d)/'jobs.sqlite3')
            db.execute('CREATE TABLE jobs(id TEXT PRIMARY KEY,request_id TEXT UNIQUE NOT NULL,params TEXT NOT NULL,config_hash TEXT NOT NULL,state TEXT NOT NULL,created REAL NOT NULL,updated REAL NOT NULL,progress TEXT NOT NULL,stats TEXT)');db.close()
            store=JobStore(d)
            try:
                image,_=self.seed(store,request='image')
                video,_=self.seed(store,prompt='Sea whale shark',video=True,request='video')
                store.submit({'prompt':'pending'},'hash','pending')
                page=store.history(limit=1)
                self.assertEqual(page['total'],2);self.assertEqual(page['items'][0]['job_id'],video)
                self.assertEqual(store.history(limit=1,offset=page['next_offset'])['items'][0]['job_id'],image)
                self.assertEqual(store.history(media_type='video',query='SEA')['total'],1)
                self.assertEqual(store.history(query="' OR 1=1 --")['total'],0)
                self.assertEqual(store.history(query='%')['total'],0)
                store.edit_history(video,'favorite',True)
                self.assertEqual(store.history(favorites=True)['items'][0]['job_id'],video)
                self.assertNotIn('private-config-hash',json.dumps(store.history()))
                (store.outputs/(video+'.mp4')).unlink()
                self.assertFalse(store.history(media_type='video')['items'][0]['available'])
            finally:store.close()
            store=JobStore(d)
            try:self.assertEqual(store.history(favorites=True)['total'],1)
            finally:store.close()

    def test_delete_removes_media_but_deduplication_never_regenerates(self):
        with tempfile.TemporaryDirectory() as d:
            store=JobStore(d)
            try:
                job,params=self.seed(store)
                store.edit_history(job,'delete')
                self.assertEqual(store.history()['total'],0)
                self.assertFalse((store.outputs/(job+'.png')).exists())
                with self.assertRaises(KeyError):store.image(job)
                with self.assertRaises(ValueError):store.submit(params,'private-config-hash','one')
                self.assertEqual(store.pending(),0)
                for kwargs in ({'limit':True},{'limit':101},{'offset':-1},{'media_type':'bad'},{'query':None}):
                    with self.assertRaises(ValueError):store.history(**kwargs)
                pending,_=store.submit({'prompt':'pending'},'hash','pending')
                with self.assertRaises(ValueError):store.edit_history(pending['job_id'],'delete')
            finally:store.close()

    def test_gateway_gallery_works_without_account_and_requires_auth(self):
        with tempfile.TemporaryDirectory() as d,patch.dict(os.environ,{'EASY_CONFIG_PATH':str(Path(d)/'settings.json')}):
            with patch('studio_core.load_config',return_value=None):core=StudioCore(Path(d)/'studio')
            worker=create_worker_server(core,'x'*32);wt=threading.Thread(target=worker.serve_forever);wt.start()
            private_write(Path(d)/'studio/connection.json',json.dumps({'url':f'http://127.0.0.1:{worker.server_port}','token':'x'*32}).encode())
            job,_=self.seed(core.store,video=True)
            gui=create_server(0);gt=threading.Thread(target=gui.serve_forever);gt.start();base=f'http://127.0.0.1:{gui.server_port}'
            try:
                page=requests.get(base+'/gallery',timeout=5)
                nonce=re.search("const nonce='([^']+)'",page.text).group(1);headers={'X-Local-Session':nonce}
                self.assertEqual(requests.get(base+'/gallery/items',timeout=5).status_code,401)
                self.assertEqual(requests.get(base+'/gallery/items',headers={**headers,'Origin':'https://evil.test'},timeout=5).status_code,403)
                r=requests.get(base+'/gallery/items?media_type=video&query=sea',headers=headers,timeout=5)
                self.assertEqual(r.status_code,200,r.text);self.assertEqual(r.json()['items'][0]['job_id'],job)
                self.assertEqual(requests.get(base+'/gallery/media/'+job,timeout=5).status_code,401)
                media=requests.get(base+'/gallery/media/'+job,headers=headers,timeout=5)
                self.assertEqual(media.headers['Content-Type'],'video/mp4');self.assertEqual(media.content,b'fixture')
                self.assertEqual(requests.get(base+'/gallery/items?limit=no',headers=headers,timeout=5).status_code,409)
                url=base+'/gallery/items/'+job
                self.assertEqual(requests.post(url,json={'action':'delete'},timeout=5).status_code,401)
                self.assertEqual(requests.post(url,headers=headers,json={'action':'favorite','favorite':True},timeout=5).status_code,200)
                self.assertEqual(core.store.history(favorites=True)['total'],1)
                self.assertEqual(requests.post(url,headers=headers,json={'action':'delete'},timeout=5).status_code,200)
                self.assertEqual(requests.get(base+'/gallery/media/'+job,headers=headers,timeout=5).status_code,404)
                self.assertIsNone(core.backend)  # browsing never provisions a GPU
            finally:
                gui.shutdown();gt.join();gui.server_close();worker.shutdown();wt.join();worker.server_close();core.close()


if __name__=='__main__':unittest.main()
