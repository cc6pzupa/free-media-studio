"""Reproducible Kaggle timing comparisons through the existing local worker."""
import argparse
import json
import secrets
import statistics
import time
from pathlib import Path
from model_registry import DEFAULT_MODEL,MODELS,model_profile,model_steps

METRICS=('encoding_seconds','generation_seconds','denoising_seconds','decode_seconds',
         'milliseconds_per_step','denoising_steps','request_seconds','model_reused','peak_gpu_gib','parallel_strategy')


def benchmark_plan(models=None,qwen_steps=(20,16,12)):
    cases=[]
    for model in models or list(MODELS):
        profile=model_profile(model)
        for steps in qwen_steps if model==DEFAULT_MODEL else (profile['default_steps'],):
            cases.append({'model_id':model,'steps':model_steps(model,steps),'seed':42})
    return cases


def run_plan(client,cases,prompt,runs,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    results=[]
    for case_index,case in enumerate(cases):
        row=dict(case,samples=[])
        # One unmeasured warm-up per case makes load/download time visible but
        # excludes it from the steady-state aggregate.
        for repeat in range(runs+1):
            request_id='benchmark-'+secrets.token_hex(16)
            job=client.submit(prompt,case['seed'],case['steps'],request_id,case['model_id'])
            print(case['model_id'],case['steps'],'warm-up' if repeat==0 else 'sample '+str(repeat),flush=True)
            while True:
                state=client.get_job(job['job_id'])
                if state['state']=='complete':break
                if state['state'] in ('error','interrupted'):raise RuntimeError(state.get('message','Benchmark task failed'))
                time.sleep(2)
            usage={key:state.get('usage',{}).get(key) for key in METRICS}
            target=output/f'{case_index:02}-{case["model_id"]}-{case["steps"]}-{repeat}.png'
            target.write_bytes(client.get_image(job['job_id']))
            if repeat==0:row['warmup']=usage
            else:row['samples'].append(usage)
        row['median']={key:statistics.median([sample[key] for sample in row['samples'] if isinstance(sample.get(key),(int,float)) and not isinstance(sample.get(key),bool)])
            for key in METRICS if any(isinstance(sample.get(key),(int,float)) and not isinstance(sample.get(key),bool) for sample in row['samples'])}
        results.append(row)
        (output/'timings.json').write_text(json.dumps({'prompt':prompt,'cases':results},ensure_ascii=False,indent=2))
    return results


def probe_existing_kernel():
    from studio_core import load_config
    from account_backend import AccountBackend
    from session_runtime import JupyterTransport
    config=load_config()
    if not config or config['mode']!='account':raise RuntimeError('先在 GUI 保存自动连接账号，并生成一张图片。')
    account=AccountBackend(config['user'],config['token'],dataset=config['dataset'])
    transport=None
    try:
        url=account._read().get('jupyter_url','')
        if not account._valid_connection(url):raise RuntimeError('没有已连接会话；先在 GUI 生成一张图片。')
        transport=JupyterTransport(url)
        response=transport.http.get(transport.base+'/api/kernels',timeout=15);response.raise_for_status()
        kernels=response.json()
        if len(kernels)!=1:raise RuntimeError('需要唯一的已连接 Notebook 内核。')
        transport.kernel_id=kernels[0]['id'];transport.start()
        source=Path(__file__).with_name('gpu_diagnostics.py').read_text()
        source+="\nprint('CC_GPU_PROBE:'+json.dumps(inspect_gpu_runtime()))"
        output=transport.execute(source,timeout=60)
        return json.loads(next(line[len('CC_GPU_PROBE:'):] for line in output.splitlines() if line.startswith('CC_GPU_PROBE:')))
    finally:
        if transport:transport.close()
        account.close()


def main(argv=None):
    parser=argparse.ArgumentParser(description='Compare fixed-prompt, fixed-seed warm Kaggle inference. Defaults to plan-only.')
    parser.add_argument('--gpu-probe',action='store_true',help='Inspect the already connected Kaggle kernel; does not start another session')
    parser.add_argument('--run',action='store_true',help='Submit real Kaggle GPU tasks using saved local credentials')
    parser.add_argument('--models',nargs='+',choices=list(MODELS))
    parser.add_argument('--runs',type=int,default=3)
    parser.add_argument('--output',type=Path,default=Path('.benchmarks'))
    parser.add_argument('--prompt',default='An adult orange-haired catgirl beside a window, cream sweater, soft morning light, detailed anime illustration')
    args=parser.parse_args(argv)
    if not 1<=args.runs<=20:parser.error('--runs must be 1–20')
    if args.gpu_probe:
        try:print(json.dumps(probe_existing_kernel(),ensure_ascii=False,indent=2))
        except Exception:raise SystemExit('无法读取已连接的 Kaggle 内核。先完成一次生图，再重试诊断。') from None
        return
    cases=benchmark_plan(args.models)
    if not args.run:
        print(json.dumps({'plan':cases,'measured_runs_per_case':args.runs,'warmups_per_case':1,'submits_gpu_tasks':False},indent=2));return
    from studio_client import StudioClient
    client=StudioClient()
    try:run_plan(client,cases,args.prompt,args.runs,args.output)
    finally:client.close()


if __name__=='__main__':main()
