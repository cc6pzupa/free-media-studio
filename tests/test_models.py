import ast
import contextlib
import io
import json
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path
from unittest.mock import Mock,patch
from model_registry import DEFAULT_MODEL,MODELS,MODEL_SOURCES,model_profile,model_steps
from session_backend import bootstrap_code,generation_code,model_switch_code
from studio_core import StudioCore,validate_generation
from benchmark import benchmark_plan,run_plan
import asset_manager,cache_manager,model_assets


class ModelTests(unittest.TestCase):
    def test_defaults_invalid_models_and_lightning_checkpoint_steps(self):
        self.assertEqual([model_steps(m) for m in MODELS],[20,9,4])
        self.assertEqual(validate_generation('fox',42,None,'z-image-turbo')['steps'],9)
        self.assertNotIn('model_id',validate_generation('fox',42,20))
        for value in ['../other',{},False,'unknown']:
            with self.assertRaises(ValueError):model_profile(value)
        for steps in [1,2,8,20]:
            with self.assertRaises(ValueError):model_steps('sdxl-lightning',steps)

    def test_switch_releases_managed_objects_then_reconnect_reuses_new_model(self):
        original=Path.read_text;loads=[]
        def read(path,*args,**kwargs):
            if path.name=='remote_models.py':return "_loads.append(_CC_MODEL_ID)\npipe=object()\nencoder=object()\ntransformer=object()\ndef generate_image(*a,**k):pass\n"
            return original(path,*args,**kwargs)
        cuda=Mock();torch=types.SimpleNamespace(cuda=cuda)
        scope={'_loads':loads}
        source="_loads.append('qwen-image-2.1')\npipe=object()\nencoder=object()\ntransformer=object()\ndef generate_image(*a,**k):pass\n"
        with patch.dict(sys.modules,{'torch':torch}),patch.object(Path,'read_text',read),contextlib.redirect_stdout(io.StringIO()):
            exec(bootstrap_code('owner/assets',source,install=False),scope)
            old=scope['pipe']
            exec(bootstrap_code('owner/assets',asset_mode='auto',install=False,model_id='z-image-turbo'),scope)
            new=scope['pipe'];self.assertIsNot(old,new)
            runtime=scope['_COMFY_QWEN_STATE']['runtime_id']
            exec(bootstrap_code('owner/assets',asset_mode='auto',install=False,model_id='z-image-turbo'),scope)
            self.assertIs(scope['pipe'],new);self.assertEqual(scope['_COMFY_QWEN_STATE']['runtime_id'],runtime)
            exec(bootstrap_code('owner/assets',source,install=False),scope)
        self.assertEqual(loads,[DEFAULT_MODEL,'z-image-turbo',DEFAULT_MODEL])
        self.assertEqual(scope['_CC_MODEL_ID'],DEFAULT_MODEL)
        self.assertGreater(cuda.empty_cache.call_count,0)

    def test_unmanaged_model_is_not_deleted_on_switch(self):
        obj=object();scope={'pipe':obj}
        with self.assertRaisesRegex(RuntimeError,'unmanaged'):exec(model_switch_code('z-image-turbo'),scope)
        self.assertIs(scope['pipe'],obj)

    def test_generated_bootstraps_compile_and_generation_checks_active_model(self):
        for model in MODELS:
            ast.parse(bootstrap_code('owner/assets',install=False,asset_mode='auto',model_id=model))
            text=generation_code('test',42,model_steps(model),model);ast.parse(text)
            with self.assertRaises(AssertionError):exec(text,{'_CC_MODEL_ID':'wrong'})

    def test_sdxl_downloads_only_fp16_base_components_and_selected_unet(self):
        manifest={'sha':'pinned','siblings':[{'rfilename':p,'size':1} for p in (
            'model_index.json','text_encoder/model.safetensors','text_encoder/model.fp16.safetensors',
            'unet/config.json','unet/diffusion_pytorch_model.fp16.safetensors',
            'vae/diffusion_pytorch_model.fp16.safetensors','tokenizer/vocab.json')]}
        lightning={'sha':'lightning','siblings':[{'rfilename':p,'size':1} for p in ('sdxl_lightning_4step_unet.safetensors','sdxl_lightning_8step_unet.safetensors')]}
        with patch.object(model_assets,'repository_manifest',side_effect=[manifest,lightning],create=True),patch.object(model_assets,'safe_relative',asset_manager.safe_relative,create=True):
            files=[row[3] for row in model_assets.model_asset_records('sdxl-lightning',[('base','pinned'),('fast','lightning')],object())]
        self.assertIn('base/text_encoder/model.fp16.safetensors',files)
        self.assertIn('lightning/sdxl_lightning_4step_unet.safetensors',files)
        self.assertNotIn('base/unet/diffusion_pytorch_model.fp16.safetensors',files)
        self.assertNotIn('base/text_encoder/model.safetensors',files)
        self.assertNotIn('lightning/sdxl_lightning_8step_unet.safetensors',files)

    def test_model_cache_identity_cannot_be_reused_by_another_model(self):
        for name in ('BASE_REVISION','GGUF_REVISION','safe_relative'):
            setattr(cache_manager,name,getattr(asset_manager,name))
        manifest={'studio_model_id':'z-image-turbo','studio_cache_version':2,'model_sources':[['repo','revision']],
                  'cache_files':{'transformer/model.safetensors':1}}
        with patch.object(cache_manager,'_CC_MODEL_ID','z-image-turbo',create=True),patch.object(cache_manager,'_CC_MODEL_SOURCES',[['repo','revision']],create=True):
            self.assertEqual(cache_manager.validate_cache_manifest(manifest),manifest)
            with self.assertRaises(RuntimeError):cache_manager.validate_cache_manifest(dict(manifest,studio_model_id='sdxl-lightning'))
            with self.assertRaises(RuntimeError):cache_manager.validate_cache_manifest(dict(manifest,model_sources=[['repo','wrong']]))

    def test_old_worker_cannot_silently_generate_with_the_wrong_model(self):
        from studio_client import StudioClient
        client=StudioClient(autostart=False)
        try:
            with patch.object(client,'status',return_value={'configured':True}),patch.object(client,'_request') as request:
                with self.assertRaisesRegex(RuntimeError,'Restart'):
                    client.submit('fox',model_id='z-image-turbo')
                request.assert_not_called()
                client.model_selection_supported=True
                client.submit('fox',model_id='z-image-turbo')
                self.assertEqual(request.call_args.args[2]['model_id'],'z-image-turbo')
        finally:client.close()

    def test_benchmark_separates_warmup_and_omits_private_ids(self):
        client=Mock();client.submit.return_value={'job_id':'private-job'}
        client.get_job.side_effect=[{'state':'complete','usage':{'generation_seconds':90,'runtime_id':'private','model_reused':False}},
            {'state':'complete','usage':{'generation_seconds':5,'runtime_id':'private','model_reused':True}},
            {'state':'complete','usage':{'generation_seconds':7,'runtime_id':'private','model_reused':True}}]
        client.get_image.return_value=b'png'
        with tempfile.TemporaryDirectory() as d,contextlib.redirect_stdout(io.StringIO()):
            result=run_plan(client,benchmark_plan(['sdxl-lightning']),'fox',2,d)
            report=Path(d,'timings.json').read_text()
        self.assertEqual(result[0]['median']['generation_seconds'],6)
        self.assertEqual(result[0]['warmup']['generation_seconds'],90)
        self.assertNotIn('private',report)
        ids=[call.args[3] for call in client.submit.call_args_list]
        self.assertEqual(len(ids),len(set(ids)))

    def test_model_is_part_of_durable_job_identity_and_worker_routing(self):
        class Backend:
            def __init__(self,config):self.models=[]
            def generate_with_progress(self,prompt,seed,steps,progress=None,model_id=DEFAULT_MODEL):
                from fixtures.fake_worker import png
                self.models.append(model_id);return png(),{'model_id':model_id,'steps':steps,'denoising_seconds':1.2}
            def close(self):pass
        with tempfile.TemporaryDirectory() as d,patch('studio_core.load_config',return_value=None):
            core=StudioCore(d,backend_factory=Backend)
            try:
                core.configure({'mode':'account','user':'owner','token':'KGAT_fixture'})
                first=core.submit('fox',request_id='one',model_id='sdxl-lightning')
                self.assertEqual(first['job_id'],core.submit('fox',request_id='one',model_id='sdxl-lightning')['job_id'])
                with self.assertRaises(ValueError):core.submit('fox',request_id='one',model_id='z-image-turbo')
                core.start()
                for _ in range(200):
                    result=core.store.get(first['job_id'])
                    if result['state'] in ('error','complete'):break
                    time.sleep(.01)
                self.assertEqual(result['state'],'complete')
                self.assertEqual(result['usage']['model_id'],'sdxl-lightning')
                self.assertEqual(core.backend.models,['sdxl-lightning'])
            finally:core.close()


if __name__=='__main__':unittest.main()
