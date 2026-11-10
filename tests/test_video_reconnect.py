import contextlib
import io
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from session_backend import model_switch_code
from video_session import video_bootstrap_code


class VideoReconnectTest(unittest.TestCase):
    def test_reconnect_reuses_actor_and_switching_back_to_images_closes_it(self):
        source='''class VideoProcess:
 def __init__(self,model_id,**kwargs):self.profile={'id':model_id};self.closed=False
 def alive(self):return not self.closed
 def close(self):self.closed=True
'''
        previous=sys.modules.get('video_process');paths=list(sys.path)
        try:
            with tempfile.TemporaryDirectory() as d,patch('video_session.VIDEO_FILES',()),patch.dict(sys.modules,{'torch':SimpleNamespace(cuda=SimpleNamespace(empty_cache=lambda:None))}),contextlib.redirect_stdout(io.StringIO()):
                Path(d,'video_process.py').write_text(source)
                def code(model):return video_bootstrap_code(model,install=False).replace("Path('/kaggle/temp/cc-media-video-runtime')",'Path('+repr(d)+')')
                state={'pipe':object(),'_CC_MANAGED_MODEL':True,'_CC_MODEL_ID':'qwen-image-2.1'}
                exec(code('wan21-t2v-1.3b'),state);actor=state['_CC_VIDEO_PROCESS']
                self.assertNotIn('pipe',state)
                exec(code('wan21-t2v-1.3b'),state)
                self.assertIs(actor,state['_CC_VIDEO_PROCESS']);self.assertFalse(actor.closed)
                exec(model_switch_code('qwen-image-2.1'),state)
                self.assertTrue(actor.closed);self.assertNotIn('_CC_VIDEO_PROCESS',state)
                self.assertEqual(state['_CC_MODEL_ID'],'qwen-image-2.1')
        finally:
            sys.path[:]=paths
            if previous is None:sys.modules.pop('video_process',None)
            else:sys.modules['video_process']=previous


if __name__=='__main__':unittest.main()
