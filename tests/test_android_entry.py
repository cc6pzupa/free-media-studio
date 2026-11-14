"""The Android entry runs the worker as a thread and never spawns a process."""
import json
import os
import re
import sys
import tempfile
import unittest
import urllib.request
from pathlib import Path
from unittest import mock

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import android_entry
import studio_client


class AndroidEntryTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        saved={key:os.environ.get(key) for key in ('HOME','EASY_CONFIG_PATH','PYTHONUTF8')}
        def restore():
            for key,value in saved.items():
                if value is None:os.environ.pop(key,None)
                else:os.environ[key]=value
            studio_client.worker_starter=None
        self.addCleanup(restore)
        self.addCleanup(self.temp.cleanup)
        self.addCleanup(android_entry.stop)

    def get(self,url,headers=None):
        request=urllib.request.Request(url,headers=headers or {})
        with urllib.request.urlopen(request,timeout=10) as response:return response.status,response.read()

    def test_gui_and_worker_run_in_process(self):
        def no_process(*args,**kwargs):raise AssertionError('Android must not spawn a worker process')
        with mock.patch.object(studio_client,'spawn_worker',no_process):
            url=android_entry.start(self.temp.name,0)
            self.assertRegex(url,r'^http://127\.0\.0\.1:\d+$')
            self.assertEqual(android_entry.start(self.temp.name,0),url)
            self.assertEqual(os.environ['EASY_CONFIG_PATH'],str(Path(self.temp.name)/'config'/'settings.json'))
            status,page=self.get(url+'/')
            self.assertEqual(status,200)
            nonce=re.search(r"const nonce='([^']+)'",page.decode()).group(1)
            # These go through StudioClient to the in-process worker.
            status,body=self.get(url+'/configuration',{'X-Local-Session':nonce})
            self.assertEqual(status,200)
            self.assertFalse(json.loads(body)['configured'])
            status,body=self.get(url+'/gallery/items',{'X-Local-Session':nonce})
            self.assertEqual(status,200)
            self.assertFalse(android_entry.busy())
            self.assertTrue((Path(self.temp.name)/'config'/'studio'/'connection.json').is_file())
        self.assertEqual(android_entry.stop(),'stopped')

    def test_worker_restarts_when_client_finds_it_gone(self):
        url=android_entry.start(self.temp.name,0)
        worker=android_entry._state['worker'];thread=android_entry._state['worker_thread']
        worker.shutdown();thread.join(10)
        self.assertFalse(thread.is_alive())
        client=studio_client.StudioClient()
        client.ensure()
        self.assertTrue(android_entry._state['worker_thread'].is_alive())
        self.assertIsNot(android_entry._state['worker_thread'],thread)
        self.assertTrue(url)


if __name__=='__main__':unittest.main()
