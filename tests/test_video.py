import ast
import contextlib
import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from studio_core import StudioCore
from video_registry import VIDEO_MODELS,validate_video,video_catalog
from video_session import video_bootstrap_code,video_generation_code
from video_assets import video_asset_records


class Backend:
    def __init__(self,config):self.count=0
    def close(self):pass
    def generate_video_with_progress(self,progress=None,**params):
        self.count+=1
        if progress:progress({'phase':'exporting','token':'must-not-persist'})
        return b'\0\0\0\x18ftypisom'+b'fixture-mp4',dict(params,media_type='video',
            parallel_strategy='cfg_parallel',model_reused=self.count>1,token='must-not-persist')


class VideoTest(unittest.TestCase):
    def test_catalog_and_budget_bounds(self):
        self.assertEqual([p['tier'] for p in video_catalog()],['efficient','balanced','quality'])
        for model in VIDEO_MODELS:self.assertEqual(validate_video('cat',model_id=model)['frames'],33)
        for kwargs in [{'frames':True},{'frames':10},{'frames':89},{'steps':0},{'seed':-1},{'model_id':'unknown'},{'prompt':''}]:
            args=dict(prompt='cat');args.update(kwargs)
            with self.assertRaises(ValueError):validate_video(**args)
        with self.assertRaises(ValueError):validate_video('cat',model_id='ltx-video-2b-distilled',steps=20)

    def test_notebook_protocol_is_video_only_and_compiles_for_every_tier(self):
        for model in VIDEO_MODELS:
            ast.parse(video_bootstrap_code(model,install=False))
            code=video_generation_code('cat " \\ newline\n',model_id=model)
            ast.parse(code)
            self.assertIn('CC_VIDEO_RESULT:',code)
            self.assertNotIn('COMFY_RESULT:',code)
            with self.assertRaises(AssertionError):exec(code,{'_CC_MODEL_ID':'different'})

    def test_download_selection_excludes_other_ltx_checkpoints(self):
        p=VIDEO_MODELS['ltx-video-2b-distilled']
        names=['model_index.json','text_encoder/model-1.safetensors','vae/config.json',
            'vae/old.safetensors','transformer/old.safetensors',p['checkpoint'],
            'ltxv-13b-0.9.8-distilled.safetensors','media/movie.gif']
        manifest={'sha':p['revision'],'siblings':[{'rfilename':n} for n in names]}
        selected=[i['rfilename'] for i in video_asset_records(p,manifest)]
        self.assertIn(p['checkpoint'],selected);self.assertIn('text_encoder/model-1.safetensors',selected)
        self.assertNotIn('transformer/old.safetensors',selected);self.assertNotIn('vae/old.safetensors',selected)
        with self.assertRaises(RuntimeError):video_asset_records(p,dict(manifest,sha='wrong'))

    def test_video_jobs_share_queue_deduplication_and_survive_restart(self):
        with tempfile.TemporaryDirectory() as d:
            with patch('studio_core.load_config',return_value=None):core=StudioCore(d,backend_factory=Backend)
            try:
                core.configure({'mode':'account','user':'owner','token':'fixture'})
                first=core.submit_video('cat',request_id='one');duplicate=core.submit_video('cat',request_id='one')
                self.assertEqual(first['job_id'],duplicate['job_id']);self.assertTrue(duplicate['deduplicated'])
                with self.assertRaises(ValueError):core.submit_video('dog',request_id='one')
                core.start();deadline=time.monotonic()+5
                while time.monotonic()<deadline:
                    job=core.store.get(first['job_id'])
                    if job['state']=='complete':break
                    time.sleep(.02)
                self.assertTrue(job['video_ready']);self.assertFalse(job['image_ready'])
                raw=core.store.video(first['job_id']);self.assertEqual(raw[4:8],b'ftyp')
                self.assertNotIn('must-not-persist',json.dumps(job))
                with self.assertRaises(ValueError):core.store.image(first['job_id'])
            finally:core.close()
            with patch('studio_core.load_config',return_value=None):other=StudioCore(d,backend_factory=Backend)
            try:self.assertEqual(other.store.video(first['job_id']),raw)
            finally:other.close()

    def test_manual_connection_cannot_queue_a_video(self):
        with tempfile.TemporaryDirectory() as d:
            with patch('studio_core.load_config',return_value=None):core=StudioCore(d,backend_factory=Backend)
            try:
                with self.assertRaises(ValueError):core.submit_video('cat')
            finally:core.close()


if __name__=='__main__':unittest.main()
