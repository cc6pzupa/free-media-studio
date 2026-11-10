"""Explicit, isolated Qwen CFG experiment; does not modify the app configuration."""
import argparse
import gc
import json
import shutil
import statistics
import tempfile
import time
from pathlib import Path


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets',type=Path,required=True,help='Prepared Qwen INT8/Q4 asset directory')
    parser.add_argument('--output',type=Path,default=Path('/kaggle/working/cc-dual-cfg'))
    parser.add_argument('--runs',type=int,default=3)
    parser.add_argument('--steps',type=int,default=20)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--encoder-mode',choices=['release','offload'],default='release',
        help='release measures preencoded sampling; offload also measures full warm requests with cached models')
    parser.add_argument('--prompt',default='An adult orange-haired catgirl sitting beside a sunlit window, cream sweater, detailed anime illustration, gentle warm light')
    args=parser.parse_args(argv)
    if not 1<=args.runs<=10:parser.error('--runs must be 1–10')
    if not 1<=args.steps<=100:parser.error('--steps must be 1–100')

    print('PHASE import_runtime',flush=True)
    import numpy as np
    import torch
    from transformers import Qwen3VLForConditionalGeneration,Qwen3VLProcessor
    from diffusers import QwenImage21Pipeline,AutoencoderKLQwenImage21,FlowMatchEulerDiscreteScheduler
    from dual_gpu import ParallelCFG,load_replica

    if torch.cuda.device_count()!=2:raise RuntimeError('Select GPU T4 x2')
    print('PHASE runtime_ready',json.dumps({'torch':torch.__version__,'gpus':[torch.cuda.get_device_name(i) for i in range(2)]}),flush=True)
    assets=args.assets
    args.output.mkdir(parents=True,exist_ok=True)
    # GGUF uses mmap. Keep its checkpoint on the VM's local disk.
    checkpoint_dir=tempfile.TemporaryDirectory(prefix='cc-cfg-',dir='/kaggle/temp')
    local_checkpoint=Path(checkpoint_dir.name)/'transformer-q4.gguf'
    source=assets/'transformer-q4.gguf'
    print('PHASE load_processor',flush=True)
    processor=Qwen3VLProcessor.from_pretrained(assets/'processor',local_files_only=True)
    scheduler=FlowMatchEulerDiscreteScheduler.from_pretrained(assets/'scheduler',local_files_only=True)
    sigmas=json.loads((assets/'model_index.json').read_text()).get('sample_sigmas')
    print('PHASE load_encoder',flush=True)
    encoder=Qwen3VLForConditionalGeneration.from_pretrained(assets/'text_encoder',dtype=torch.float16,
        device_map={'':1},attn_implementation='sdpa',local_files_only=True)
    pipe=QwenImage21Pipeline(scheduler=scheduler,vae=None,text_encoder=encoder,
        processor=processor,transformer=None,sample_sigmas=sigmas)
    print('PHASE encode_prompt',flush=True)
    @torch.inference_mode()
    def encode_request():
        started=time.perf_counter()
        pipe.register_modules(text_encoder=encoder)
        try:
            positive=pipe.encode_prompt(args.prompt,device=torch.device('cuda:1'))
            negative=pipe.encode_prompt(' ',device=torch.device('cuda:1'))
            positive=tuple(v.to('cuda:0') if v is not None else None for v in positive)
            negative=tuple(v.to('cuda:0') if v is not None else None for v in negative)
            torch.cuda.synchronize(0);torch.cuda.synchronize(1)
            return positive,negative,round(time.perf_counter()-started,3)
        finally:pipe.register_modules(text_encoder=None)
    positive,negative,initial_encoding_seconds=encode_request()
    encoder_reference=encoder if args.encoder_mode=='offload' else None
    if args.encoder_mode=='release':del encoder
    del pipe
    gc.collect();torch.cuda.empty_cache()
    print('ENCODER_STATE',args.encoder_mode,flush=True)
    print('PHASE copy_gguf',flush=True)
    shutil.copyfile(source,local_checkpoint)
    print('PHASE load_primary',flush=True)
    primary=load_replica(assets,'cuda:0',checkpoint_path=local_checkpoint)
    # Sequentially stage VAE files too: lazy mmap over dataset storage made
    # cold initialization much slower than local-disk reads on the test VM.
    print('PHASE copy_vae',flush=True)
    local_vae=Path(checkpoint_dir.name)/'vae'
    shutil.copytree(assets/'vae',local_vae)
    print('PHASE load_vae',flush=True)
    vae=AutoencoderKLQwenImage21.from_pretrained(local_vae,dtype=torch.float16,local_files_only=True).to('cuda:0')
    pipe=QwenImage21Pipeline(scheduler=scheduler,vae=vae,text_encoder=None,
        processor=processor,transformer=primary,sample_sigmas=sigmas)

    @torch.inference_mode()
    def sample():
        torch.cuda.synchronize(0)
        start=time.perf_counter()
        image=pipe(prompt_embeds=positive[0],prompt_embeds_mask=positive[1],
            negative_prompt_embeds=negative[0],negative_prompt_embeds_mask=negative[1],
            width=512,height=512,num_inference_steps=args.steps,true_cfg_scale=5.0,use_kv_cache=True,
            sigmas=torch.linspace(1,0,args.steps+1)[:-1].tolist(),
            generator=torch.Generator(device='cuda:0').manual_seed(args.seed)).images[0]
        torch.cuda.synchronize(0)
        return image,round(time.perf_counter()-start,3)

    print('PHASE baseline_warmup',flush=True)
    reference,cold_seconds=sample()
    print('SAMPLE baseline_warmup',cold_seconds,flush=True)
    reference.save(args.output/'baseline.png')
    baseline=[];baseline_matches=[];baseline_requests=[]
    for repeat in range(args.runs):
        request_start=time.perf_counter()
        if args.encoder_mode=='offload':positive,negative,encoding_seconds=encode_request()
        image,seconds=sample()
        if args.encoder_mode=='offload':baseline_requests.append(round(time.perf_counter()-request_start,3))
        print('SAMPLE baseline',repeat,seconds,flush=True)
        baseline.append(seconds)
        baseline_matches.append(bool(np.array_equal(np.asarray(reference),np.asarray(image))))
    report={'gpu':[torch.cuda.get_device_name(i) for i in range(2)],'steps':args.steps,'seed':args.seed,
        'resolution':[512,512],'runs':args.runs,'mode':'preencoded_encoder_'+('released' if args.encoder_mode=='release' else 'offloaded'),
        'includes_encoder_transfer_or_reload':False,'production_enabled':False,
        'sampling_scope':'Preencoded sampling and VAE decode',
        'baseline_cold_seconds':cold_seconds,'baseline_seconds':baseline,
        'baseline_same_pixels':baseline_matches,
        'baseline_median_seconds':statistics.median(baseline)}
    if baseline_requests:report.update(baseline_request_seconds=baseline_requests,
        baseline_request_median_seconds=statistics.median(baseline_requests),initial_encoding_seconds=initial_encoding_seconds)
    replica=None
    original=primary.forward
    try:
        if args.encoder_mode=='offload':
            print('PHASE offload_encoder',flush=True)
            start=time.perf_counter()
            encoder.to('cpu');torch.cuda.synchronize(1)
            gc.collect();torch.cuda.empty_cache()
            report['encoder_offload_seconds']=round(time.perf_counter()-start,3)
        print('PHASE load_replica',flush=True)
        start=time.perf_counter()
        replica=load_replica(assets,'cuda:1',checkpoint_path=local_checkpoint)
        report['replica_load_seconds']=round(time.perf_counter()-start,3)
        parallel_times=[];errors=[];pairs=[]
        for repeat in range(args.runs):
            print('PHASE parallel_sample',repeat,flush=True)
            with ParallelCFG(primary,replica,negative) as parallel:
                image,seconds=sample()
            delta=np.abs(np.asarray(reference,dtype=np.int16)-np.asarray(image,dtype=np.int16))
            errors.append({'max':int(delta.max()),'mean':float(delta.mean())})
            pairs.append(parallel.calls);parallel_times.append(seconds)
            print('SAMPLE parallel',repeat,seconds,'PIXEL_ERROR',errors[-1],flush=True)
            image.save(args.output/f'parallel-{repeat}.png')
        report.update(parallel_seconds=parallel_times,parallel_median_seconds=statistics.median(parallel_times),
            pixel_errors=errors,cfg_pairs=pairs,parallel_same_pixels=all(row['max']==0 for row in errors))
        if args.encoder_mode=='offload':
            requests=[]
            for repeat in range(args.runs):
                print('PHASE full_warm_request',repeat,flush=True)
                start=time.perf_counter()
                replica.to('cpu');torch.cuda.synchronize(1);torch.cuda.empty_cache()
                encoder.to('cuda:1');torch.cuda.synchronize(1)
                positive,negative,encoding_seconds=encode_request()
                encoder.to('cpu');torch.cuda.synchronize(1);torch.cuda.empty_cache()
                replica.to('cuda:1');torch.cuda.synchronize(1)
                with ParallelCFG(primary,replica,negative) as parallel:
                    image,seconds=sample()
                elapsed=round(time.perf_counter()-start,3)
                same=bool(np.array_equal(np.asarray(reference),np.asarray(image)))
                row={'seconds':elapsed,'encoding_seconds':encoding_seconds,'sampling_seconds':seconds,
                     'migration_and_setup_seconds':round(elapsed-encoding_seconds-seconds,3),
                     'same_pixels':same,'cfg_pairs':parallel.calls,'encoder_reused':encoder is encoder_reference}
                requests.append(row)
                print('FULL_WARM_REQUEST',json.dumps(row),flush=True)
                image.save(args.output/f'warm-request-{repeat}.png')
            report.update(warm_requests=requests,warm_request_median_seconds=statistics.median(row['seconds'] for row in requests),
                encoder_load_count=1,transformer_load_count=2,
                warm_request_includes_encoding_and_device_transfers=True)
    except Exception as error:
        report.update(error_type=type(error).__name__,error=str(error).split('\n')[0][:200])
    finally:
        primary.forward=original
        if replica is not None:replica.to('cpu')
        del replica
        gc.collect();torch.cuda.empty_cache()
        if args.encoder_mode=='offload':
            encoder.to('cuda:1');torch.cuda.synchronize(1)
            report['encoder_restored_to_gpu1']=str(encoder.device)=='cuda:1'
        report['forward_restored']=primary.forward==original
    print('PHASE rollback_sample',flush=True)
    rollback,seconds=sample()
    rollback.save(args.output/'rollback.png')
    report.update(rollback_seconds=seconds,
        rollback_same_pixels=bool(np.array_equal(np.asarray(reference),np.asarray(rollback))))
    (args.output/'report.json').write_text(json.dumps(report,indent=2))
    print('CC_DUAL_FINAL',json.dumps(report),flush=True)
    if not report['rollback_same_pixels']:raise RuntimeError('Rollback output differs from baseline')
    if 'error_type' in report:raise RuntimeError('Parallel experiment failed; original forward restored')
    if not report['parallel_same_pixels']:raise RuntimeError('Parallel output differs from baseline')
    if any(not row['same_pixels'] or row['cfg_pairs']!=args.steps for row in report.get('warm_requests',[])):
        raise RuntimeError('Full warm request did not preserve the baseline output or CFG step count')
    return report


if __name__=='__main__':main()
