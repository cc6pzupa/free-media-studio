import contextlib
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock,patch
from studio_core import normalize_config,configuration_fingerprint

class Tensor:
    def to(self,device):return self
class Event:
    def record(self,*a):pass
    def elapsed_time(self,other):return 1000
class Grid:
    def __getitem__(self,key):return self
    def tolist(self):return [1,.5]
class Replica:
    def __init__(self):self.device='cuda:1';self.moves=[]
    def to(self,device):self.moves.append(device);self.device=device;return self
    def parameters(self):return iter([self])
class Pipe:
    def __init__(self):self.encodes=[];self.samples=[];self.fail=False
    def register_modules(self,**kwargs):self.encoder=kwargs['text_encoder']
    def encode_prompt(self,prompt,device):self.encodes.append(prompt);return Tensor(),Tensor(),Tensor()
    def __call__(self,**kwargs):
        self.samples.append(kwargs)
        if self.fail:self.fail=False;raise RuntimeError('sample failed')
        kwargs['callback_on_step_end'](self,0,1,{})
        return SimpleNamespace(images=['image'])
class Parallel:
    def __init__(self,*a):self.calls=20
    def __enter__(self):return self
    def __exit__(self,*a):pass

class QwenRuntimeTest(unittest.TestCase):
    def runtime(self):
        fake=SimpleNamespace(inference_mode=lambda:lambda f:f,device=lambda d:d,
            cuda=SimpleNamespace(reset_peak_memory_stats=Mock(),synchronize=Mock(),empty_cache=Mock(),Event=lambda **k:Event(),current_stream=lambda i:None,max_memory_allocated=lambda i:1),
            linspace=lambda *a:Grid(),Generator=lambda **k:SimpleNamespace(manual_seed=lambda s:s))
        encoder=SimpleNamespace(device='cuda:1')
        pipe=Pipe();replica=Replica()
        scope={'transformer':object(),'encoder':encoder,'pipe':pipe,'ASSET_DIR':'assets',
            'move_int8_encoder':lambda e,d:setattr(e,'device',d),'load_replica':Mock(return_value=replica),
            'ParallelCFG':Parallel,'emit_progress':Mock()}
        source=Path(__file__).resolve().parents[1].joinpath('remote_qwen.py').read_text()
        with patch.dict(sys.modules,{'torch':fake}):exec(source,scope)
        return scope,pipe,replica,source,fake

    def test_live_code_upgrade_and_new_prompt_reuse_same_models_and_replica(self):
        s,p,r,source,t=self.runtime();generate=s['generate_image']
        _,first=generate('first')
        with patch.dict(sys.modules,{'torch':t}):exec(source,s)
        _,second=s['generate_image']('second')
        self.assertEqual(s['load_replica'].call_count,1)
        self.assertIs(s['_CC_DUAL_STATE']['replica'],r)
        self.assertEqual(p.encodes,['first',' ','second',' '])
        self.assertFalse(first['dual_model_reused']);self.assertTrue(second['dual_model_reused'])
        self.assertEqual(second['parallel_strategy'],'cfg_parallel')
        self.assertEqual(s['encoder'].device,'cpu');self.assertIsNone(p.encoder)

    def test_switch_to_original_mode_releases_replica_and_restores_encoder(self):
        s,p,r,*_=self.runtime();s['generate_image']('first');s['_CC_ACCELERATION']='split'
        _,stats=s['generate_image']('second')
        self.assertIsNone(s['_CC_DUAL_STATE']['replica'])
        self.assertEqual(s['encoder'].device,'cuda:1')
        self.assertEqual(stats['parallel_strategy'],'encoder_split')
        self.assertIs(p.encoder,s['encoder'])

    def test_replica_failure_falls_back_once_and_does_not_retry_on_next_request(self):
        s,p,r,*_=self.runtime();s['load_replica'].side_effect=RuntimeError('allocation failed')
        _,first=s['generate_image']('first');_,second=s['generate_image']('second')
        self.assertEqual(first['acceleration_fallback'],'RuntimeError')
        self.assertEqual(second['parallel_strategy'],'encoder_split')
        self.assertEqual(s['load_replica'].call_count,1)
        self.assertEqual(len(p.samples),2);self.assertEqual(s['encoder'].device,'cuda:1')

    def test_failed_parallel_sampling_preserves_seed_for_original_path(self):
        s,p,r,*_=self.runtime();p.fail=True
        _,stats=s['generate_image']('first',seed=123)
        self.assertEqual([v['generator'] for v in p.samples],[123,123])
        self.assertEqual(stats['parallel_strategy'],'encoder_split')
        self.assertIsNone(s['_CC_DUAL_STATE']['replica'])

    def test_acceleration_default_validation_and_job_configuration_identity(self):
        config={'user':'owner','token':'key','mode':'account'}
        auto=normalize_config(config);split=normalize_config(dict(config,acceleration='split'))
        self.assertEqual(auto['acceleration'],'auto')
        self.assertNotEqual(configuration_fingerprint(auto),configuration_fingerprint(split))
        for invalid in ['dual',True,None,{},'']:
            with self.assertRaises(ValueError):normalize_config(dict(config,acceleration=invalid))

    def test_additional_model_generation_upgrades_without_reloading_existing_model(self):
        from session_backend import bootstrap_code
        original=Path.read_text
        source=["_loads+=1\npipe=object()\nencoder=object()\n@torch.inference_mode()\ndef generate_image(*a,**k):return 'first'\n"]
        def read(path,*a,**k):return source[0] if path.name=='remote_models.py' else original(path,*a,**k)
        scope={'_loads':0}
        torch=SimpleNamespace(inference_mode=lambda:lambda f:f,cuda=SimpleNamespace(empty_cache=Mock()))
        with patch.dict(sys.modules,{'torch':torch}),patch.object(Path,'read_text',read),contextlib.redirect_stdout(__import__('io').StringIO()):
            exec(bootstrap_code('owner/assets',asset_mode='auto',install=False,model_id='z-image-turbo'),scope)
            pipe=scope['pipe']
            source[0]="raise AssertionError('must not reload')\n@torch.inference_mode()\ndef generate_image(*a,**k):return 'updated'\n"
            exec(bootstrap_code('owner/assets',asset_mode='auto',install=False,model_id='z-image-turbo'),scope)
        self.assertIs(scope['pipe'],pipe)
        self.assertEqual(scope['_loads'],1)
        self.assertEqual(scope['generate_image'](),'updated')

    def test_old_worker_rejects_new_generation_before_sending_a_request(self):
        from studio_client import StudioClient
        client=StudioClient(autostart=False)
        try:
            with patch.object(client,'ensure'),patch.object(client.http,'request') as request:
                with self.assertRaisesRegex(RuntimeError,'Restart'):
                    client._request('POST','/jobs',{})
                request.assert_not_called()
        finally:client.close()

if __name__=='__main__':unittest.main()
