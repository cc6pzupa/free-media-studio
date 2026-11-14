"""Image editing, sizes, negative prompts, image-to-video and batch ZIP export."""
import base64
import contextlib
import io
import json
import os
import re
import struct
import sys
import tempfile
import threading
import time
import types
import unittest
import zipfile
import zlib
from pathlib import Path
from unittest.mock import patch

import requests
import importlib.util
# Pillow and numpy exist on Kaggle and in ComfyUI, not in the local client.
HAS_PIL=importlib.util.find_spec('PIL') is not None
HAS_NUMPY=importlib.util.find_spec('numpy') is not None

from easy_gateway import create_server
from media_inputs import decode_image,digest,png_size
from model_registry import image_options
from session_backend import generation_code
from studio_core import StudioCore,private_write,validate_generation
from studio_worker import create_worker_server
from video_registry import validate_video
from video_session import video_generation_code


def png(width=8,height=8,color=(240,120,60)):
    def chunk(kind,data):return struct.pack('>I',len(data))+kind+data+struct.pack('>I',zlib.crc32(kind+data)&0xffffffff)
    pixels=b''.join(b'\0'+bytes(color)*width for _ in range(height))
    return b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('>IIBBBBB',width,height,8,2,0,0,0))+chunk(b'IDAT',zlib.compress(pixels))+chunk(b'IEND',b'')


SOURCE=png(40,20);MASK=png(40,20,(255,255,255))


class Backend:
    calls=[]
    def __init__(self,config):pass
    def generate_with_progress(self,prompt,seed=42,steps=20,progress=None,model_id='qwen-image-2.1',**options):
        Backend.calls.append(dict(options,prompt=prompt,seed=seed,model_id=model_id))
        width,height=options.get('width',512),options.get('height',512)
        if prompt=='wrong size':width=512;height=512
        return png(width,height),{'model_id':model_id,'width':width,'height':height,'seed':seed,'steps':steps,
            'task':options.get('task','text'),'secret':'hidden'}
    def generate_video_with_progress(self,prompt,seed=42,steps=30,model_id='wan21-t2v-1.3b',frames=33,progress=None,**options):
        Backend.calls.append(dict(options,prompt=prompt,model_id=model_id))
        return b'\0\0\0\x18ftypisom'+b'fixture',{'model_id':model_id,'media_type':'video','frames':frames,
            'image_to_video':'image' in options}
    def close(self):pass


def wait(core,job_id):
    deadline=time.monotonic()+5
    while time.monotonic()<deadline:
        job=core.store.get(job_id)
        if job['state'] in ('complete','error'):return job
        time.sleep(.02)
    raise AssertionError('job did not finish')


class ValidationTests(unittest.TestCase):
    def test_options_are_omitted_when_default_and_checked_per_model(self):
        self.assertEqual(validate_generation('fox',42,None),{'prompt':'fox','seed':42,'steps':20})
        self.assertEqual(validate_generation('fox',42,None,size='512x512',task='text'),{'prompt':'fox','seed':42,'steps':20})
        self.assertEqual(image_options('qwen-image-2.1',20,'1024x576','blurry'),{'width':1024,'height':576,'negative_prompt':'blurry'})
        for args in [('qwen-image-2.1',20,None,None,'img2img',None,True),       # Qwen 2.1 has no edit pipeline
                     ('z-image-turbo',9,None,'blurry'),                        # no CFG, negative unused
                     ('z-image-turbo',9,'640x640'),                            # unsupported size
                     ('z-image-turbo',9,None,None,'img2img'),                  # missing image
                     ('z-image-turbo',9,None,None,'inpaint',None,True),        # missing mask
                     ('z-image-turbo',9,None,None,'img2img',None,True,True),   # mask with img2img
                     ('z-image-turbo',9,None,None,'text',0.5),                 # strength without editing
                     ('z-image-turbo',9,None,None,'img2img',1.5,True),
                     ('z-image-turbo',9,None,None,'img2img',True,True),
                     ('sdxl-lightning',4,None,None,'img2img',0.2,True)]:       # 4 x 0.2 < 1 step
            with self.assertRaises(ValueError,msg=args):image_options(*args)
        self.assertEqual(image_options('sdxl-lightning',4,None,None,'inpaint',None,True,True),{'task':'inpaint','strength':0.9})
        self.assertEqual(image_options('z-image-turbo',9,'768x512',None,'img2img',None,True),
            {'width':768,'height':512,'task':'img2img','strength':0.6})

    def test_inputs_are_checked_by_signature_without_pillow(self):
        self.assertEqual(decode_image('data:image/png;base64,'+base64.b64encode(SOURCE).decode()),SOURCE)
        self.assertEqual(decode_image(base64.b64encode(b'\xff\xd8\xff\xe0jpeg').decode())[:3],b'\xff\xd8\xff')
        self.assertEqual(png_size(SOURCE),(40,20))
        for bad in ['not base64!',base64.b64encode(b'GIF89a').decode(),'',None,{}]:
            with self.assertRaises(ValueError):decode_image(bad)
        with self.assertRaises(ValueError):decode_image(b'\x89PNG\r\n\x1a\n'+b'0'*(8*1024*1024))

    def test_image_to_video_only_on_supported_models(self):
        self.assertEqual(validate_video('cat',model_id='wan22-ti2v-5b',image='abc')['image'],'abc')
        self.assertIn('image',validate_video('cat',model_id='ltx-video-2b-distilled',image='abc'))
        with self.assertRaises(ValueError):validate_video('cat',model_id='wan21-t2v-1.3b',image='abc')
        self.assertEqual(validate_video('cat',negative_prompt='blur')['negative_prompt'],'blur')
        with self.assertRaises(ValueError):validate_video('cat',model_id='ltx-video-2b-distilled',negative_prompt='blur')
        self.assertNotIn('image',validate_video('cat'))


class QueueTests(unittest.TestCase):
    def setUp(self):
        self.folder=tempfile.TemporaryDirectory();Backend.calls=[]
        with patch('studio_core.load_config',return_value=None):
            self.core=StudioCore(Path(self.folder.name),backend_factory=Backend)
        self.core.configure({'mode':'account','user':'owner','token':'fixture'});self.core.start()

    def tearDown(self):self.core.close();self.folder.cleanup()

    def test_uploads_are_stored_by_digest_and_sent_as_bytes(self):
        encoded=base64.b64encode(SOURCE).decode();mask=base64.b64encode(MASK).decode()
        job=self.core.submit('repaint',7,None,'edit-1','z-image-turbo','768x512',None,'inpaint',0.8,encoded,mask)
        done=wait(self.core,job['job_id'])
        self.assertEqual(done['state'],'complete',done)
        call=Backend.calls[-1]
        self.assertEqual((call['image'],call['mask']),(SOURCE,MASK))
        self.assertEqual((call['width'],call['height'],call['task'],call['strength']),(768,512,'inpaint',0.8))
        stored=json.loads(self.core.store.db.execute('SELECT params FROM jobs').fetchone()[0])
        self.assertEqual(stored['image'],digest(SOURCE))
        self.assertTrue((Path(self.folder.name)/'inputs'/digest(MASK)).is_file())
        self.assertEqual(done['usage']['task'],'inpaint');self.assertNotIn('secret',done['usage'])
        self.assertEqual(png_size(self.core.store.image(job['job_id'])),(768,512))
        # Same request, same upload: deduplicated without a new generation.
        self.assertTrue(self.core.submit('repaint',7,None,'edit-1','z-image-turbo','768x512',None,'inpaint',0.8,encoded,mask)['deduplicated'])
        self.assertEqual(len(Backend.calls),1)
        self.core.store.edit_history(job['job_id'],'delete')
        self.assertFalse((Path(self.folder.name)/'inputs'/digest(SOURCE)).exists())

    def test_shared_input_survives_deleting_one_of_two_jobs(self):
        encoded=base64.b64encode(SOURCE).decode()
        first=self.core.submit('a',1,None,'a','z-image-turbo',None,None,'img2img',None,encoded)
        second=self.core.submit('b',2,None,'b','sdxl-lightning',None,None,'img2img',0.5,encoded)
        wait(self.core,first['job_id']);wait(self.core,second['job_id'])
        self.core.store.edit_history(first['job_id'],'delete')
        self.assertTrue((Path(self.folder.name)/'inputs'/digest(SOURCE)).is_file())

    def test_output_with_wrong_size_is_rejected(self):
        job=self.core.submit('wrong size',1,None,'size','z-image-turbo','1024x1024')
        self.assertEqual(wait(self.core,job['job_id'])['state'],'error')

    def test_invalid_uploads_are_not_stored(self):
        with self.assertRaises(ValueError):self.core.submit('a',1,None,'x','qwen-image-2.1',None,None,'img2img',None,base64.b64encode(SOURCE).decode())
        with self.assertRaises(ValueError):self.core.submit('a',1,None,'y','z-image-turbo',None,None,'img2img',None,'bm90IGFuIGltYWdl')
        self.assertEqual(list((Path(self.folder.name)/'inputs').iterdir()),[])

    def test_extended_options_need_account_mode(self):
        self.core.config=dict(self.core.config,mode='session')
        with self.assertRaises(ValueError):self.core.submit('a',1,None,'z',None,'768x768')
        with self.assertRaises(ValueError):self.core.submit('a',1,None,'n',None,None,'blurry')

    def test_image_to_video_carries_first_frame(self):
        job=self.core.submit_video('wave',1,None,'v','wan22-ti2v-5b',None,base64.b64encode(SOURCE).decode(),'blur')
        done=wait(self.core,job['job_id'])
        self.assertEqual(done['state'],'complete')
        self.assertEqual((Backend.calls[-1]['image'],Backend.calls[-1]['negative_prompt']),(SOURCE,'blur'))
        self.assertTrue(done['usage']['image_to_video'])


class RemoteCodeTests(unittest.TestCase):
    @unittest.skipUnless(HAS_PIL,'Pillow is only installed on the GPU kernel')
    def test_generation_cell_crops_inputs_and_passes_only_needed_options(self):
        from PIL import Image
        seen=[]
        class Result:
            def save(self,buffer,format):buffer.write(b'png')
        def generate_image(prompt,**kwargs):seen.append(kwargs);return Result(),{}
        state={'runtime_id':'r','load_count':1,'loaded_this_request':False,'generations':0}
        scope={'generate_image':generate_image,'_COMFY_QWEN_STATE':state,'_CC_MODEL_ID':'z-image-turbo'}
        with contextlib.redirect_stdout(io.StringIO()):
            exec(generation_code('fox',1,9,'z-image-turbo',768,512,None,'inpaint',0.8,SOURCE,MASK),scope)
            exec(generation_code('fox',1,9,'z-image-turbo'),scope)
        edit,plain=seen
        self.assertEqual(edit['image'].size,(768,512));self.assertEqual(edit['image'].mode,'RGB')
        self.assertEqual(edit['mask'].mode,'L');self.assertEqual((edit['task'],edit['strength']),('inpaint',0.8))
        self.assertEqual(plain,{'seed':1,'steps':9,'width':512,'height':512})
        self.assertIsInstance(edit['image'],Image.Image)

    def test_video_cell_carries_base64_first_frame(self):
        code=video_generation_code('wave',1,None,'ltx-video-2b-distilled',None,SOURCE)
        args=json.loads(json.loads(re.search(r"json\.loads\((\".*?\")\)",code).group(1)))
        self.assertEqual(base64.b64decode(args['image']),SOURCE)

    def test_edit_pipelines_reuse_loaded_modules(self):
        """Run remote_models.generate_image against stub torch/diffusers modules."""
        built=[]
        class Tensor:
            def to(self,*a,**k):return self
        class Finite:
            def all(self):return self
            def item(self):return True
        class Event:
            def __init__(self,**k):pass
            def record(self,*a):pass
            def elapsed_time(self,other):return 10.0
        cuda=types.SimpleNamespace(reset_peak_memory_stats=lambda i:None,synchronize=lambda *a:None,Event=Event,
            current_stream=lambda i:None,max_memory_allocated=lambda i:0)
        torch=types.SimpleNamespace(cuda=cuda,device=lambda v:v,isfinite=lambda v:Finite(),inference_mode=lambda:(lambda f:f),
            Generator=lambda device:types.SimpleNamespace(manual_seed=lambda s:s))
        class Picture:
            size=(768,512)
            def resize(self,size):return self
        class Pipe:
            def __init__(self,**modules):
                self.modules=dict(modules);self.calls=[];built.append(self)
                for k,v in modules.items():setattr(self,k,v)
            def register_modules(self,**modules):
                for k,v in modules.items():setattr(self,k,v)
            def encode_prompt(self,*a,**k):return [Tensor()],None
            def __call__(self,**kwargs):
                self.calls.append(kwargs)
                kwargs['callback_on_step_end'](self,0,1,{'latents':Tensor()})
                return types.SimpleNamespace(images=[Picture()])
        diffusers=types.ModuleType('diffusers')
        diffusers.ZImageImg2ImgPipeline=diffusers.ZImageInpaintPipeline=Pipe
        source=Path('remote_models.py').read_text(encoding='utf-8')
        runtime=source[source.index('def _cc_vae_boundary'):]
        class VAE:
            dtype='fp32'
            def __init__(self):self.moves=[]
            def parameters(self):return iter([types.SimpleNamespace(device='cuda:0')])
            def to(self,device):self.moves.append(device);return self
            def encode(self,x):return x
            def decode(self,x):return x
        vae=VAE();cuda.empty_cache=lambda:None
        base=Pipe(scheduler='s',vae=vae,tokenizer='t',transformer='x')
        scope={'torch':torch,'time':time,'pipe':base,'encoder':'encoder','_CC_MODEL_ID':'z-image-turbo',
               '_CC_MODEL_PROFILE':{'default_steps':9}}
        with patch.dict(sys.modules,{'diffusers':diffusers}):
            exec(runtime,scope)
            image,stats=scope['generate_image']('fox',seed=1,width=768,height=512,steps=9,task='inpaint',image='I',mask='M',strength=0.8)
        edit=built[-1]
        self.assertIsNot(edit,base)
        self.assertEqual((edit.transformer,edit.vae,edit.text_encoder),('x',vae,'encoder'))
        self.assertNotIn('encode',vars(vae))  # per-call VAE wrappers are removed afterwards
        call=edit.calls[0]
        self.assertEqual((call['image'],call['mask_image'],call['strength'],call['width']),('I','M',0.8,768))
        self.assertEqual((stats['task'],stats['strength']),('inpaint',0.8))
        self.assertEqual(base.calls,[])
        # SDXL image-to-image takes its size from the input image, not width/height.
        class SDXLPipe(Pipe):
            def encode_prompt(self,*a,**k):return Tensor(),None,Tensor(),None
        diffusers.StableDiffusionXLImg2ImgPipeline=diffusers.StableDiffusionXLInpaintPipeline=SDXLPipe
        base=SDXLPipe(vae='v',text_encoder='e1',text_encoder_2='e2',tokenizer='t1',tokenizer_2='t2',unet='u',scheduler='s')
        scope.update(pipe=base,encoder='e1',_CC_MODEL_ID='sdxl-lightning')
        with patch.dict(sys.modules,{'diffusers':diffusers}):
            scope['generate_image']('fox',seed=1,width=768,height=512,steps=4,task='img2img',image='I',strength=0.5)
        edit=built[-1];call=edit.calls[0]
        self.assertEqual((edit.unet,edit.text_encoder,edit.text_encoder_2),('u','e1','e2'))
        self.assertNotIn('width',call);self.assertEqual((call['image'],call['strength']),('I',0.5))

    def test_vae_boundary_casts_inputs_and_offloads_large_sizes(self):
        source=Path('remote_models.py').read_text(encoding='utf-8')
        helper=source[source.index('def _cc_vae_boundary'):source.index('def _cc_task_pipeline')]
        class Tensor:
            def __init__(self,device='cuda:0',dtype='fp16'):self.device,self.dtype=device,dtype
            def to(self,device=None,dtype=None):return Tensor(device or self.device,dtype or self.dtype)
        class VAE:
            dtype='fp32'
            def __init__(self):self.device='cuda:0';self.calls=[]
            def parameters(self):return iter([types.SimpleNamespace(device='cuda:0')])
            def to(self,device):self.device=device;return self
            def encode(self,x):
                self.calls.append(('encode',x.device,x.dtype,self.device))
                return types.SimpleNamespace(latent_dist=types.SimpleNamespace(parameters=Tensor('cuda:1','fp32')))
            def decode(self,x,return_dict=False):self.calls.append(('decode',x.device,x.dtype,self.device));return (x,)
        distribution=types.ModuleType('vae');outputs=types.ModuleType('outputs')
        distribution.DiagonalGaussianDistribution=lambda parameters:types.SimpleNamespace(parameters=parameters)
        outputs.AutoencoderKLOutput=lambda latent_dist:types.SimpleNamespace(latent_dist=latent_dist)
        scope={'torch':types.SimpleNamespace(cuda=types.SimpleNamespace(empty_cache=lambda:None))}
        exec(helper,scope)
        with patch.dict(sys.modules,{'diffusers.models.autoencoders.vae':distribution,'diffusers.models.modeling_outputs':outputs}):
            vae=VAE();restore=scope['_cc_vae_boundary'](vae,offload=False)
            vae.encode(Tensor());restore()
            self.assertEqual(vae.calls,[('encode','cuda:0','fp32','cuda:0')]);self.assertNotIn('encode',vars(vae))
            vae=VAE();restore=scope['_cc_vae_boundary'](vae,offload=True)
            latents=vae.encode(Tensor()).latent_dist.parameters
            vae.decode(Tensor());restore()
        self.assertEqual(vae.calls,[('encode','cuda:1','fp32','cuda:1'),('decode','cuda:1','fp32','cuda:1')])
        self.assertEqual((latents.device,vae.device),('cuda:0','cuda:0'))

    def test_failed_generation_releases_frames_and_reports_the_cause(self):
        import gc
        scope={'_COMFY_QWEN_STATE':{'runtime_id':'r','load_count':1,'loaded_this_request':False,'generations':0},'_CC_MODEL_ID':'z-image-turbo'}
        held=[]
        class Big:pass
        def generate_image(prompt,**kwargs):
            tensor=Big();held.append(__import__('weakref').ref(tensor))
            raise MemoryError('CUDA out of memory')
        scope['generate_image']=generate_image
        torch=types.SimpleNamespace(cuda=types.SimpleNamespace(empty_cache=lambda:None))
        with patch.dict(sys.modules,{'torch':torch}),self.assertRaises(RuntimeError) as caught:
            exec(generation_code('fox',1,9,'z-image-turbo'),scope)
        gc.collect()
        self.assertIn('MemoryError: CUDA out of memory',str(caught.exception))
        self.assertIsNone(held[0]())  # the failed call's locals are not kept alive
        sys.last_traceback=caught.exception.__traceback__
        with patch.dict(sys.modules,{'torch':torch}),contextlib.redirect_stdout(io.StringIO()):
            scope['generate_image']=lambda prompt,**k:(types.SimpleNamespace(save=lambda b,format:b.write(b'png')),{})
            exec(generation_code('fox',1,9,'z-image-turbo'),scope)
        self.assertIsNone(getattr(sys,'last_traceback',None))

    @unittest.skipUnless(HAS_PIL,'Pillow is only installed on the GPU kernel')
    def test_video_first_frame_and_image_pipeline_view(self):
        from PIL import Image
        built={}
        class View:
            def __init__(self,**modules):built.update(modules)
        diffusers=types.ModuleType('diffusers');diffusers.WanImageToVideoPipeline=diffusers.LTXImageToVideoPipeline=View
        with patch.dict(sys.modules,{'torch':types.SimpleNamespace(inference_mode=lambda:(lambda f:f)),'diffusers':diffusers}):
            sys.modules.pop('video_runtime',None)
            import video_runtime
            runtime=video_runtime.VideoRuntime('wan22-ti2v-5b')
            buffer=io.BytesIO();Image.new('RGB',(300,600),'red').save(buffer,format='PNG')
            frame=runtime.first_frame(base64.b64encode(buffer.getvalue()).decode())
            self.assertEqual(frame.size,(768,448))
            runtime.pipe=types.SimpleNamespace(tokenizer='t',text_encoder='e',vae='v',scheduler='s',transformer='x')
            runtime.image_pipeline()
            self.assertEqual(built,{'tokenizer':'t','text_encoder':'e','vae':'v','scheduler':'s','transformer':'x','expand_timesteps':True})
            sys.modules.pop('video_runtime',None)


class HttpTests(unittest.TestCase):
    def test_gui_edit_submission_and_zip_export(self):
        Backend.calls=[]
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
                nonce=re.search("const nonce='([^']+)'",requests.get(base+'/',timeout=5).text).group(1)
                headers={'X-Local-Session':nonce}
                image=base64.b64encode(SOURCE).decode()
                bad=requests.post(base+'/jobs',headers=headers,json={'prompt':'fox','model_id':'qwen-image-2.1','task':'img2img','image':image},timeout=5)
                self.assertEqual(bad.status_code,400);self.assertIn('Z-Image',bad.json()['error']['message'])
                ids=[]
                for seed in (1,2):
                    r=requests.post(base+'/jobs',headers=headers,json={'prompt':'fox','model_id':'z-image-turbo','seed':seed,
                        'size':'576x1024','task':'img2img','strength':0.5,'image':image,'request_id':f'batch-{seed}'},timeout=10)
                    self.assertEqual(r.status_code,202,r.text);ids.append(r.json()['job_id'])
                for job_id in ids:
                    deadline=time.monotonic()+5
                    while time.monotonic()<deadline:
                        job=requests.get(base+'/jobs/'+job_id,headers=headers,timeout=5).json()
                        if job['state']=='complete':break
                        time.sleep(.02)
                    self.assertEqual(png_size(base64.b64decode(job['result']['data'][0]['b64_json'])),(576,1024))
                self.assertEqual(requests.post(base+'/gallery/zip',json={'job_ids':ids},timeout=5).status_code,401)
                archive=requests.post(base+'/gallery/zip',headers=headers,json={'job_ids':ids},timeout=5)
                self.assertEqual(archive.headers['Content-Type'],'application/zip')
                with zipfile.ZipFile(io.BytesIO(archive.content)) as z:
                    names=z.namelist();self.assertEqual(len(names),2)
                    self.assertTrue(names[0].startswith('001-seed1-'));self.assertEqual(png_size(z.read(names[1])),(576,1024))
                self.assertEqual(requests.post(base+'/gallery/zip',headers=headers,json={'job_ids':['../x']},timeout=5).status_code,400)
                self.assertEqual(requests.post(base+'/gallery/zip',headers=headers,json={'job_ids':[]},timeout=5).status_code,400)
                # Gallery hand-offs: pages accept query strings, inputs are readable with the UI session.
                self.assertIn('text/html',requests.get(base+'/?edit='+ids[0],timeout=5).headers['Content-Type'])
                self.assertIn('text/html',requests.get(base+'/video?frame='+ids[0],timeout=5).headers['Content-Type'])
                self.assertEqual(requests.get(base+'/gallery/inputs/'+digest(SOURCE),timeout=5).status_code,401)
                stored=requests.get(base+'/gallery/inputs/'+digest(SOURCE),headers=headers,timeout=5)
                self.assertEqual((stored.content,stored.headers['Content-Type']),(SOURCE,'image/png'))
                self.assertEqual(requests.get(base+'/gallery/inputs/'+'0'*64,headers=headers,timeout=5).status_code,404)
                self.assertEqual(requests.get(base+'/gallery/inputs/../settings.json',headers=headers,timeout=5).status_code,404)
                video=requests.post(base+'/video/jobs',headers=headers,json={'prompt':'wave','model_id':'wan21-t2v-1.3b','image':image},timeout=5)
                self.assertEqual(video.status_code,400);self.assertIn('图生视频',video.json()['error']['message'])
            finally:
                gui.shutdown();thread.join();gui.server_close()
                worker.shutdown();worker_thread.join();worker.server_close();core.close()


if __name__=='__main__':unittest.main()


class ComfyNodeTests(unittest.TestCase):
    @unittest.skipUnless(HAS_PIL and HAS_NUMPY,'ComfyUI supplies Pillow and numpy')
    def test_tensors_become_png_inputs_and_edit_node_forwards_options(self):
        """Load the custom-node package with a stub torch (ComfyUI supplies the real one)."""
        import importlib.util
        import numpy as np
        class Tensor:
            def __init__(self,array):self.array=np.asarray(array,dtype=np.float32)
            def detach(self):return self
            def cpu(self):return self
            def float(self):return self
            def numpy(self):return self.array
        torch=types.ModuleType('torch');torch.from_numpy=lambda a:types.SimpleNamespace(unsqueeze=lambda d:a[None])
        root=Path(__file__).resolve().parents[1]
        package=types.ModuleType('ccmedia_nodes');package.__path__=[str(root)]
        with patch.dict(sys.modules,{'torch':torch,'ccmedia_nodes':package}):
            spec=importlib.util.spec_from_file_location('ccmedia_nodes.easy_node',root/'easy_node.py')
            node=importlib.util.module_from_spec(spec);sys.modules[spec.name]=node;spec.loader.exec_module(node)
            image=node.tensor_png(Tensor(np.ones((1,6,4,3))*0.5))
            mask=node.tensor_png(Tensor(np.ones((1,6,4))),'L')
            self.assertEqual(png_size(image),(4,6));self.assertEqual(png_size(mask),(4,6))
            from PIL import Image
            self.assertEqual(Image.open(io.BytesIO(mask)).mode,'L')
            calls=[]
            backend=types.SimpleNamespace(generate=lambda *a,**k:(calls.append((a,k)),(png(4,6),{}))[1])
            with patch.object(node,'shared_backend',return_value=backend):
                result=node.CCMediaImageEdit().edit('hat',Tensor(np.zeros((1,6,4,3))),'sdxl-lightning','inpaint',0.7,3,'768x768',
                    mask=Tensor(np.ones((1,6,4))),steps=20)
                with self.assertRaises(RuntimeError):node.CCMediaImageEdit().edit('hat',Tensor(np.zeros((1,6,4,3))),'z-image-turbo','inpaint',0.7,3,'768x768')
            args,options=calls[0]
            self.assertEqual(args,('hat',3,4))
            self.assertEqual((options['task'],options['strength'],options['size'],options['model_id']),('inpaint',0.7,'768x768','sdxl-lightning'))
            self.assertEqual(png_size(options['mask']),(4,6))
            self.assertEqual(result[0].shape,(1,6,4,3))
            self.assertIn('size',node.KaggleEasyImage.INPUT_TYPES()['optional'])


@unittest.skipUnless(HAS_PIL and importlib.util.find_spec('ipywidgets') and importlib.util.find_spec('nbformat'),
                     'The notebook widgets need ipywidgets, nbformat and Pillow')
class NotebookTests(unittest.TestCase):
    def test_standalone_notebook_edit_controls_reach_the_runtime(self):
        import datetime
        import nbformat
        from PIL import Image
        cell=nbformat.read(str(Path(__file__).resolve().parents[1]/'kaggle_standalone.ipynb'),as_version=4).cells[2].source
        scope={'__name__':'notebook'}
        with patch('IPython.display.display'):exec(cell,scope)
        scope['_BOOTSTRAPS']={k:"_COMFY_QWEN_STATE={'runtime_id':'r','load_count':1,'loaded_this_request':True,'generations':0}\n_CC_MODEL_ID=%r\n"%k
                              for k in scope['_BOOTSTRAPS']}
        seen=[]
        def generate_image(prompt,**kwargs):
            seen.append(kwargs);return Image.new('RGB',(kwargs['width'],kwargs['height'])),{'model_reused':False,'model_load_count':1}
        scope['generate_image']=generate_image
        def upload(color):
            buffer=io.BytesIO();Image.new('RGB',(300,200),color).save(buffer,format='PNG')
            return ({'name':'x.png','type':'image/png','size':1,'last_modified':datetime.datetime.now(),'content':memoryview(buffer.getvalue())},)
        scope['task'].value='inpaint'
        self.assertEqual(scope['model'].value,'z-image-turbo')
        scope['source_upload'].value=upload('red');scope['mask_upload'].value=upload('white')
        scope['size'].value='1024x576';scope['negative'].value='ignored for Z-Image'
        scope['run'](None)
        call=seen[-1]
        self.assertEqual((call['task'],call['strength'],call['width'],call['height']),('inpaint',0.9,1024,576))
        self.assertEqual((call['image'].size,call['mask'].mode),((1024,576),'L'))
        self.assertNotIn('negative_prompt',call)
        self.assertIn('1024 × 576',scope['info'].value)
        scope['model'].value='qwen-image-2.1'
        self.assertEqual(scope['task'].value,'text')
        scope['negative'].value='blurry';scope['size'].value='512x512';scope['run'](None)
        self.assertEqual(seen[-1]['negative_prompt'],'blurry');self.assertNotIn('task',seen[-1])


class FreeGpuTests(unittest.TestCase):
    def test_cublas_workspaces_are_cleared_before_each_device_cache(self):
        from session_backend import FREE_GPU_CODE
        calls=[]
        class Device:
            def __init__(self,index):self.index=index
            def __enter__(self):calls.append(('device',self.index))
            def __exit__(self,*a):pass
        cuda=types.SimpleNamespace(device_count=lambda:2,device=Device,empty_cache=lambda:calls.append('empty'))
        torch=types.SimpleNamespace(cuda=cuda,_C=types.SimpleNamespace(_cuda_clearCublasWorkspaces=lambda:calls.append('cublas')))
        with patch.dict(sys.modules,{'torch':torch}):exec(FREE_GPU_CODE,{})
        self.assertEqual(calls,['cublas',('device',0),'empty',('device',1),'empty'])


class DiskBudgetTests(unittest.TestCase):
    def setUp(self):
        import asset_manager
        self.module=asset_manager
        self.folder=tempfile.TemporaryDirectory();root=Path(self.folder.name)
        self.images=root/'cc-media-models';self.video=root/'cc-media-video';self.qwen=root/'qwen-image-assets'
        self.roots=patch.object(asset_manager,'MODEL_ROOTS',(str(self.images),str(self.video),str(self.qwen)))
        self.roots.start()

    def tearDown(self):self.roots.stop();self.folder.cleanup()

    def write(self,folder,name,size,age):
        folder.mkdir(parents=True,exist_ok=True);path=folder/name
        path.write_bytes(os.urandom(size));os.utime(folder,(time.time()-age,)*2)
        return path

    def test_least_recently_used_unused_folders_are_removed_until_download_fits(self):
        self.write(self.images/'z-image-turbo','w',200_000,300)
        self.write(self.qwen,'transformer-q4.gguf',200_000,200)
        self.write(self.images/'sdxl-lightning','w',200_000,100)
        removed=self.module.release_model_disk(self.images/'sdxl-lightning',350_000,budget=700_000)
        self.assertEqual(removed,['z-image-turbo'])  # oldest only; enough room afterwards
        self.assertTrue(self.qwen.is_dir());self.assertTrue((self.images/'sdxl-lightning').is_dir())
        removed=self.module.release_model_disk(self.images/'sdxl-lightning',10_000_000,budget=700_000)
        self.assertEqual(removed,['qwen-image-assets'])  # never the folder being prepared
        self.assertTrue((self.images/'sdxl-lightning'/'w').is_file())

    def test_hardlinked_video_files_count_once_and_shared_copies_are_released(self):
        model=self.write(self.video/'ltx-video-2b-distilled','ckpt',300_000,200)
        (self.video/'_shared').mkdir();os.link(model,self.video/'_shared'/'abc')
        self.write(self.video/'wan22-ti2v-5b','ckpt',100_000,10)
        used=self.module._disk_bytes([self.video/'ltx-video-2b-distilled',self.video/'_shared'])
        self.assertLess(used,2*300_000)  # one inode, counted once
        self.assertEqual(self.module.release_model_disk(self.video/'wan22-ti2v-5b',350_000,budget=500_000),['ltx-video-2b-distilled'])
        self.assertEqual(list((self.video/'_shared').iterdir()),[])
        # Bytes already in the target folder are not counted as new.
        self.assertEqual(self.module.release_model_disk(self.video/'wan22-ti2v-5b',100_000,budget=150_000),[])

    def test_exception_chain_and_receipt_check(self):
        try:
            try:raise MemoryError('cannot map file')
            except MemoryError:raise OSError('Unable to load weights')
        except OSError as error:chain=self.module.exception_chain(error)
        self.assertEqual(chain,'OSError: Unable to load weights <- MemoryError: cannot map file')
        path=Path(self.folder.name)/'model.gguf';path.write_bytes(b'weights')
        self.assertIsNone(self.module.file_matches_receipt(path))
        import hashlib
        path.with_name('model.gguf.verified.json').write_text(json.dumps({'size':7,'sha256':hashlib.sha256(b'weights').hexdigest()}))
        self.assertTrue(self.module.file_matches_receipt(path))
        path.write_bytes(b'weightz');self.assertFalse(self.module.file_matches_receipt(path))
