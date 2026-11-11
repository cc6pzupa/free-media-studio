"""Persistent, two-process Ulysses runner for distilled LTX video sampling."""
import json
import hashlib
import os
import signal
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path


def context_transformer_folder(folder,profile):
    digest=hashlib.sha256(json.dumps(profile,sort_keys=True).encode()).hexdigest()[:16]
    return Path(folder)/('context-transformer-'+digest)


class LTXContextRunner:
    def __init__(self,folder,profile,transformer,vae_config,scheduler_config,progress):
        self.progress=progress;self.profile=profile;self.process=None;self.offset=0
        self.temporary=tempfile.TemporaryDirectory(prefix='cc-video-cp-',dir='/kaggle/temp')
        self.directory=Path(self.temporary.name)
        self.weights=context_transformer_folder(folder,profile)
        if not (self.weights/'config.json').is_file():
            transformer.save_pretrained(self.weights,safe_serialization=True,max_shard_size='1GB')
        (self.directory/'config.json').write_text(json.dumps({'weights':str(self.weights),
            'vae':dict(vae_config),'scheduler':dict(scheduler_config),'profile':profile}),encoding='utf-8')

    def start(self):
        self.log=(self.directory/'worker.log').open('w',encoding='utf-8')
        self.process=subprocess.Popen([sys.executable,'-m','torch.distributed.run','--standalone',
            '--nproc_per_node=2',str(Path(__file__).resolve()),'--worker',str(self.directory)],
            stdin=subprocess.DEVNULL,stdout=self.log,stderr=self.log,start_new_session=not bool(os.environ.get('CC_VIDEO_ACTOR')))
        deadline=time.monotonic()+240
        while time.monotonic()<deadline:
            self._check()
            if all((self.directory/f'ready-{i}').exists() for i in range(2)):return self
            time.sleep(.1)
        self.close();raise TimeoutError('Video context workers did not become ready')

    def _check(self):
        if self.process.poll() is not None:
            detail=(self.directory/'worker.log').read_text(encoding='utf-8',errors='replace')[-2500:]
            raise RuntimeError('Video context worker exited: '+detail)
        path=self.directory/'worker.log'
        with path.open(encoding='utf-8',errors='replace') as source:
            source.seek(self.offset)
            while True:
                line=source.readline()
                if not line:break
                if line.startswith('CC_CONTEXT_PROGRESS:'):
                    self.progress(json.loads(line.split(':',1)[1]))
            self.offset=source.tell()

    def generate(self,positive,mask,params):
        import torch
        job=uuid.uuid4().hex
        torch.save({'positive':positive.cpu(),'mask':mask.cpu()},self.directory/(job+'.input.pt'))
        temporary=self.directory/'request.tmp'
        temporary.write_text(json.dumps(dict(params,job=job)),encoding='utf-8')
        temporary.replace(self.directory/'request.json')
        deadline=time.monotonic()+14400
        while time.monotonic()<deadline:
            self._check()
            output=self.directory/(job+'.result.pt')
            if output.is_file():
                result=torch.load(output,map_location='cpu',weights_only=True)
                footprints=[json.loads((self.directory/f'{job}.rank-{i}.json').read_text()) for i in range(2)]
                result['worker_peak_gib']=[item['peak_gpu_gib'] for item in footprints]
                result['worker_resident_gib']=[item['resident_gpu_gib'] for item in footprints]
                (self.directory/(job+'.input.pt')).unlink();output.unlink()
                return result
            time.sleep(.1)
        raise TimeoutError('Video sampling timed out')

    def _terminate(self,kind):
        try:
            if os.environ.get('CC_VIDEO_ACTOR'):self.process.send_signal(kind)
            else:os.killpg(self.process.pid,kind)
        except ProcessLookupError:pass

    def close(self):
        if self.process is not None and self.process.poll() is None:
            self._terminate(signal.SIGTERM)
            try:self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._terminate(signal.SIGKILL);self.process.wait(timeout=10)
        if hasattr(self,'log'):self.log.close()
        self.temporary.cleanup()


def worker(directory):
    import torch
    import torch.distributed as dist
    from accelerate import init_empty_weights
    from diffusers import (LTXPipeline,LTXVideoTransformer3DModel,AutoencoderKLLTXVideo,
                           ContextParallelConfig)
    from video_schedule import ltx_distilled_scheduler
    rank=int(os.environ['LOCAL_RANK']);device=torch.device(f'cuda:{rank}')
    torch.cuda.set_device(device)
    dist.init_process_group(backend='nccl')
    config=json.loads((directory/'config.json').read_text());p=config['profile']
    transformer=LTXVideoTransformer3DModel.from_pretrained(config['weights'],
        torch_dtype=torch.float16,device_map={'':rank},low_cpu_mem_usage=True,local_files_only=True)
    from video_attention import enable_ltx_parallel_mask_broadcast
    enable_ltx_parallel_mask_broadcast()
    transformer.enable_parallelism(config=ContextParallelConfig(ulysses_degree=2))
    transformer.set_attention_backend('native')
    with init_empty_weights():vae=AutoencoderKLLTXVideo.from_config(config['vae'])
    class DevicePipeline(LTXPipeline):
        @property
        def _execution_device(self):return device
    pipeline=DevicePipeline(transformer=transformer,vae=vae,text_encoder=None,tokenizer=None,
        scheduler=ltx_distilled_scheduler(config['scheduler'],p['sigmas']))
    resident_gpu_gib=torch.cuda.memory_allocated(device)/2**30
    (directory/f'ready-{rank}').touch();previous=None
    try:
        while True:
            request=directory/'request.json'
            if not request.exists():time.sleep(.05);continue
            params=json.loads(request.read_text());job=params['job']
            if job==previous:time.sleep(.05);continue
            previous=job
            inputs=torch.load(directory/(job+'.input.pt'),map_location='cpu',weights_only=True)
            torch.cuda.reset_peak_memory_stats(device)
            def progress(pipe,index,timestep,kwargs):
                if not torch.isfinite(kwargs['latents']).all():raise RuntimeError('Non-finite context latents')
                if rank==0:print('CC_CONTEXT_PROGRESS:'+json.dumps({'phase':'generating','step':index+1,'steps':params['steps']}),flush=True)
                return kwargs
            generator=torch.Generator(device=device).manual_seed(params['seed'])
            result=pipeline(prompt_embeds=inputs['positive'].to(device),
                prompt_attention_mask=inputs['mask'].to(device),guidance_scale=1,
                height=p['height'],width=p['width'],num_frames=params['frames'],
                frame_rate=p['fps'],num_inference_steps=params['steps'],generator=generator,
                callback_on_step_end=progress,output_type='latent').frames
            torch.cuda.synchronize(device)
            (directory/f'{job}.rank-{rank}.json').write_text(json.dumps({'peak_gpu_gib':torch.cuda.max_memory_allocated(device)/2**30,'resident_gpu_gib':resident_gpu_gib}))
            dist.barrier()
            if rank==0:
                target=directory/(job+'.result.tmp')
                torch.save({'latents':result.cpu(),'generator_state':generator.get_state()},target)
                target.replace(directory/(job+'.result.pt'))
            del result,inputs
    finally:dist.destroy_process_group()


if __name__=='__main__':
    if len(sys.argv)!=3 or sys.argv[1]!='--worker':raise SystemExit('Use VideoRuntime')
    worker(Path(sys.argv[2]))
