"""Headless batch runner, using the existing Kaggle submission path."""
import json
import os
import re
import subprocess
import tempfile
import threading
import time
import uuid
from pathlib import Path


def kernel_status(output):
    """Read the status field, never words contained in the account/job name."""
    match = re.search(r'''\bhas status\s+["'](?:KernelWorkerStatus\.)?([A-Za-z_]+)["']''', output, re.I)
    if match:
        return match.group(1).upper()
    match = re.fullmatch(r'(?:KernelWorkerStatus\.)?([A-Za-z_]+)', output.strip(), re.I)
    return match.group(1).upper() if match else 'UNKNOWN'


class BatchBackend:
    def __init__(self, user, dataset, token, runner=subprocess.run, poll_seconds=15, timeout_minutes=60,acceleration='auto'):
        if not re.fullmatch(r'[A-Za-z0-9_-]+', user):
            raise ValueError('Kaggle 用户名只能包含字母、数字、下划线和短横线')
        if not re.fullmatch(r'[A-Za-z0-9_-]+/[A-Za-z0-9_-]+', dataset):
            raise ValueError('Dataset 请填写 用户名/数据集名称')
        if not token.strip():
            raise ValueError('请填写 Kaggle API Token')
        if acceleration not in ('auto','split'):raise ValueError('Invalid acceleration mode')
        self.acceleration=acceleration
        self.user, self.dataset, self.token = user, dataset, token.strip()
        self.runner, self.poll, self.timeout = runner, poll_seconds, timeout_minutes*60
        self.lock = threading.Lock()

    def generate(self, prompt, seed=42, steps=20):
        return self.generate_with_progress(prompt, seed, steps)

    def generate_with_progress(self, prompt, seed=42, steps=20, progress=None):
        with self.lock:
            return self._generate(prompt, seed, steps, progress)

    def _generate(self, prompt, seed, steps, progress=None):
        started=time.monotonic()
        run_id=uuid.uuid4().hex
        image_name='result_'+run_id+'.png'
        slug=self.user+'/qwen-client-'+run_id[:16]
        def report(phase, message, submitted=True):
            if progress:
                update={'phase':phase, 'message':message}
                if submitted:
                    update.update(kaggle_job=slug, kaggle_url='https://www.kaggle.com/code/'+slug+'/log')
                progress(update)
        report('submitting', '正在向 Kaggle 提交新任务。', submitted=False)
        notebook=json.loads(Path(__file__).with_name('template.ipynb').read_text(encoding='utf-8'))
        from session_backend import bootstrap_code
        notebook['cells'][2]['source']=bootstrap_code(self.dataset,install=False,acceleration=self.acceleration).splitlines(keepends=True)
        code=("import json\nfrom pathlib import Path\n"
              +"PROMPT = "+repr(prompt)+"\nSEED = "+repr(seed)+"\nSTEPS = "+repr(steps)+"\n"
              +"image, stats = generate_image(PROMPT,seed=SEED,steps=STEPS,width=512,height=512)\n"
              +"image.save("+repr('/kaggle/working/'+image_name)+")\n"
              +"Path('/kaggle/working/benchmark.json').write_text(json.dumps(stats))\n")
        notebook['cells'][3]['source']=code.splitlines(keepends=True)
        for c in notebook['cells']:
            if c['cell_type']=='code':c['execution_count']=None;c['outputs']=[]
        env=dict(os.environ,KAGGLE_API_TOKEN=self.token,PYTHONUTF8="1")
        def run(*args):
            # Never expose raw CLI output: it can include account/session details.
            result=self.runner(['kaggle',*args],capture_output=True,text=True,encoding="utf-8",errors="replace",env=env,timeout=180)
            if result.returncode:
                raise RuntimeError('Kaggle 操作失败。请检查 Token、用户名、Dataset 权限和 GPU 额度。')
            return result.stdout
        with tempfile.TemporaryDirectory(prefix='qwen_client_') as d:
            folder=Path(d)
            (folder/'job.ipynb').write_text(json.dumps(notebook,ensure_ascii=False),encoding='utf-8')
            meta={'id':slug,'title':slug.split('/')[1],'code_file':'job.ipynb','language':'python','kernel_type':'notebook','is_private':True,'enable_gpu':True,'enable_internet':True,'machine_shape':'NvidiaTeslaT4','dataset_sources':[self.dataset],'competition_sources':[],'kernel_sources':[],'model_sources':[]}
            (folder/'kernel-metadata.json').write_text(json.dumps(meta),encoding='utf-8')
            run('kernels','push','-p',str(folder))
            report('queued', '任务已提交，正在等待 Kaggle 分配运行环境。')
            deadline=time.monotonic()+self.timeout
            while time.monotonic()<deadline:
                status=kernel_status(run('kernels','status',slug))
                if status=='ERROR' or status.startswith('CANCEL'):
                    raise RuntimeError('Kaggle 任务失败或已停止。请查看任务日志：'+slug+'。之后的新请求会创建独立任务。')
                if status=='COMPLETE':
                    report('downloading', 'Kaggle 计算已结束，正在取回本次图片。')
                    out=folder/'output';run('kernels','output',slug,'-p',str(out))
                    image=out/image_name
                    if image.is_file():
                        stats={}
                        if (out/'benchmark.json').is_file():stats=json.loads((out/'benchmark.json').read_text(encoding='utf-8'))
                        stats['request_seconds']=round(time.monotonic()-started,2)
                        stats['kaggle_job']=slug
                        return image.read_bytes(),stats
                    raise RuntimeError('Kaggle 任务已完成，但没有找到本次结果图片。请查看任务输出和日志：'+slug)
                if status=='RUNNING':
                    report('running', 'Kaggle 正在准备环境、加载模型或生图，可打开任务日志查看具体进度。')
                elif status in {'QUEUED','NEW_SCRIPT'}:
                    report('queued', 'Kaggle 任务正在排队，等待运行。')
                else:
                    report('checking', '正在确认 Kaggle 任务状态，请勿重复提交。')
                time.sleep(self.poll)
        raise TimeoutError(f'等待已超过 {self.timeout/60:g} 分钟。请先查看任务 {slug} 的状态，再决定是否重试；本地超时不会取消远端任务。')
