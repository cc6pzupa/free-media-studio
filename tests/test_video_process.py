import io
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from video_process import VideoProcess


class VideoProcessTest(unittest.TestCase):
    def test_reuses_worker_and_forwards_progress(self):
        # A real pipe worker exercises framing/select, repeated calls and graceful exit.
        source='''import sys,json
count=0
for line in sys.stdin:
 c=json.loads(line)
 if c['action']=='close':break
 count+=1
 for payload in [{'progress':{'phase':'generating'}},{'result':{'count':count}}]:
  print('CC_VIDEO_PROCESS:'+json.dumps(dict(id=c['id'],**payload)),flush=True)
'''
        process=subprocess.Popen([os.sys.executable,'-u','-c',source],stdin=subprocess.PIPE,
                                 stdout=subprocess.PIPE,start_new_session=True)
        actor=VideoProcess.__new__(VideoProcess);actor.process=process;actor.buffer=b'';events=[];actor.progress=events.append
        try:
            self.assertEqual(actor.request('one')['count'],1)
            self.assertEqual(actor.request('two')['count'],2)
            self.assertEqual(len(events),2);self.assertTrue(actor.alive())
        finally:actor.close()
        self.assertIsNotNone(process.poll());self.assertFalse(actor.alive())

    def test_rejects_response_from_another_request(self):
        source="import sys;sys.stdin.readline();print('CC_VIDEO_PROCESS:{\"id\":\"wrong\",\"result\":{}}',flush=True)"
        process=subprocess.Popen([os.sys.executable,'-u','-c',source],stdin=subprocess.PIPE,
                                 stdout=subprocess.PIPE,start_new_session=True)
        actor=VideoProcess.__new__(VideoProcess);actor.process=process;actor.buffer=b'';actor.progress=lambda event:None
        try:
            with self.assertRaisesRegex(RuntimeError,'mismatch'):actor.request('one')
        finally:actor.close()


if __name__=='__main__':unittest.main()
