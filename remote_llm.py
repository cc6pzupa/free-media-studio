"""Language models on the Kaggle session, through Ollama.

Runs inside the Kaggle kernel. Measured on dual T4 (Qwen3-8B Q4_K_M): Ollama
decodes 35 tok/s on GPU, 6.4 tok/s with half the layers on the CPU and 3.6
tok/s on the CPU alone. It installs in about a minute, splits a model across
both GPUs and system RAM by itself, and pulls GGUF builds straight from
Hugging Face. llm_select chooses which build.

Output lines for the local client:
    COMFY_PROGRESS:{json}   progress events (phase, message, percent)
    CC_LLM_READY:{json}     the loaded model after prepare()
    CC_LLM_SSE:{json}       one OpenAI chat-completion stream chunk
"""
import hashlib
import json
import os
import shutil
import signal
import subprocess
import time
import urllib.error
import urllib.request

ROOT = '/kaggle/temp/cc-llm'
HOST = '127.0.0.1:11434'
BASE = 'http://' + HOST


def _request(path, body=None, timeout=60):
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(BASE + path, data, {'Content-Type': 'application/json'})
    return urllib.request.urlopen(request, timeout=timeout)


def hardware():
    """GPU memory (MiB each), system RAM and CPU count of this session."""
    gpus = []
    if shutil.which('nvidia-smi'):
        try:
            out = subprocess.run(['nvidia-smi', '--query-gpu=memory.total', '--format=csv,noheader,nounits'],
                                 capture_output=True, text=True, timeout=30).stdout
            gpus = [int(x) for x in out.split()]
        except Exception:
            gpus = []
    with open('/proc/meminfo') as meminfo:
        ram = int(meminfo.readline().split()[1]) / 2**20
    return {'gpus': gpus, 'ram_gib': round(ram, 1), 'cpus': os.cpu_count() or 1}


def ollama_tag(build, repo):
    """The hf.co tag Ollama pulls; it matches the quantization label in the file name."""
    name = build['first'].rsplit('/', 1)[-1]
    quant = build['quant']
    label = 'UD-' + quant if ('UD-' + quant).lower() in name.lower() else quant
    return 'hf.co/%s:%s' % (repo, label)


class LlmRuntime:
    def __init__(self, progress=None):
        self.progress = progress or (lambda event: None)
        self.proc = None
        self.context = None
        self.model = None
        self.info = None

    def _emit(self, phase, **fields):
        self.progress(dict(fields, phase=phase, media_type='text'))

    def install(self):
        if shutil.which('ollama'):
            return
        self._emit('installing', message='正在安装 Ollama（约 1 分钟）。')
        # The installer unpacks a .tar.zst; the Kaggle image has no zstd.
        steps = '(command -v zstd >/dev/null || (apt-get update -qq && apt-get install -y -qq zstd)) && curl -fsSL https://ollama.com/install.sh | sh'
        result = subprocess.run(steps, shell=True, capture_output=True, text=True, timeout=900)
        if result.returncode != 0 or not shutil.which('ollama'):
            raise RuntimeError('Ollama install failed: ' + (result.stdout + result.stderr)[-600:])

    def alive(self):
        if self.proc is None or self.proc.poll() is not None:
            return False
        try:
            with _request('/api/version', timeout=5):
                return True
        except Exception:
            return False

    def serve(self, context):
        if self.alive() and self.context == context:
            return
        self.stop()
        os.makedirs(ROOT, exist_ok=True)
        env = dict(os.environ, OLLAMA_HOST=HOST, OLLAMA_MODELS=os.path.join(ROOT, 'models'),
                   OLLAMA_CONTEXT_LENGTH=str(context), OLLAMA_KEEP_ALIVE='-1', OLLAMA_NUM_PARALLEL='1',
                   OLLAMA_MAX_LOADED_MODELS='1')
        log = open(os.path.join(ROOT, 'ollama.log'), 'ab')
        self.proc = subprocess.Popen(['ollama', 'serve'], stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        deadline = time.time() + 120
        while time.time() < deadline:
            if self.alive():
                self.context = context
                return
            if self.proc.poll() is not None:
                break
            time.sleep(1)
        raise RuntimeError('Ollama did not start')

    def pull(self, tag):
        self._emit('download', message='正在从 Hugging Face 下载模型。', percent=0)
        last = 0
        with _request('/api/pull', {'model': tag, 'stream': True}, timeout=7200) as response:
            for raw in response:
                event = json.loads(raw)
                if event.get('error'):
                    raise RuntimeError('Pull failed: ' + str(event['error'])[:300])
                total, done = event.get('total'), event.get('completed')
                if total and done and time.time() - last > 2:
                    last = time.time()
                    self._emit('download', message='正在从 Hugging Face 下载模型。', percent=round(100 * done / total, 1),
                               downloaded_gb=round(done / 1e9, 2), total_gb=round(total / 1e9, 2))

    def prepare(self, name, context=8192, choice=None, select=None):
        """Find the best build of a model for this hardware, download and load it.

        choice: {'repo':..., 'first':...} picks one of the listed alternatives instead.
        """
        import llm_select
        select = select or llm_select
        hw = hardware()
        self._emit('searching', message='正在 Hugging Face 上查找最合适的量化版本。')
        best, considered = select.find(name, hw, context)
        options = []
        for entry in considered:
            for option in entry['options']:
                options.append(dict(option, repo=entry['repo']))
        if choice:
            picked = [o for o in options if o['repo'] == choice.get('repo') and o.get('first') == choice.get('first')]
            if not picked:
                raise ValueError('The chosen build is not available for this hardware')
            best = dict(picked[0])
        if not best:
            raise ValueError('No build of this model fits this session (%d GPU, %.0f GB RAM)' % (len(hw['gpus']), hw['ram_gib']))
        tag = ollama_tag(best, best['repo'])
        # Models pulled from Hugging Face bring their own default context (often 32K), which
        # makes the KV cache larger than planned; a named variant fixes num_ctx.
        alias = 'cc-' + hashlib.sha256(('%s|%d' % (tag, context)).encode()).hexdigest()[:12]
        self.install()
        self.serve(context)
        if self.model != alias:
            self.pull(tag)
            self.create(alias, tag, context)
            self._emit('loading', message='正在加载模型到显存。')
            # Load through the endpoint chat uses, so the first reply does not load
            # the model again with different settings.
            with _request('/v1/chat/completions', {'model': alias, 'max_tokens': 1,
                                                   'messages': [{'role': 'user', 'content': 'hi'}]}, timeout=1800):
                pass
            self.model = alias
        placement = self.placement(alias)
        self.info = {'name': name, 'tag': tag, 'repo': best['repo'], 'quant': best['quant'], 'file': best['first'],
                     'bytes': best['bytes'], 'placement': best['placement'], 'estimated_tok_per_s': best['tok_per_s'],
                     'context': context, 'hardware': hw, 'loaded': placement,
                     'alternatives': [{k: o.get(k) for k in ('repo', 'first', 'quant', 'bytes', 'placement', 'tok_per_s')} for o in options[:12]]}
        print('CC_LLM_READY:' + json.dumps(self.info), flush=True)
        return self.info

    def create(self, alias, tag, context):
        """A variant of the pulled model with a fixed context length."""
        try:
            with _request('/api/create', {'model': alias, 'from': tag, 'parameters': {'num_ctx': context}, 'stream': False}, timeout=600):
                return
        except urllib.error.HTTPError:
            pass
        # Older servers: the same through a Modelfile.
        modelfile = os.path.join(ROOT, alias + '.Modelfile')
        with open(modelfile, 'w') as handle:
            handle.write('FROM %s\nPARAMETER num_ctx %d\n' % (tag, context))
        env = dict(os.environ, OLLAMA_HOST=HOST)
        result = subprocess.run(['ollama', 'create', alias, '-f', modelfile], capture_output=True, text=True, timeout=600, env=env)
        if result.returncode != 0:
            raise RuntimeError('Could not set the context length: ' + (result.stdout + result.stderr)[-300:])

    def placement(self, tag):
        """How much of the loaded model Ollama put in GPU memory."""
        try:
            with _request('/api/ps', timeout=10) as response:
                for model in json.load(response).get('models', []):
                    if model.get('model') == tag or model.get('name') == tag:
                        size, vram = model.get('size') or 0, model.get('size_vram') or 0
                        return {'size_gb': round(size / 1e9, 2), 'vram_gb': round(vram / 1e9, 2),
                                'gpu_share': round(vram / size, 2) if size else None}
        except Exception:
            pass
        return None

    def chat(self, body):
        """Stream one OpenAI chat completion; each chunk is printed as CC_LLM_SSE."""
        if not self.model or not self.alive():
            raise RuntimeError('No language model is loaded')
        body = dict(body, model=self.model, stream=True, stream_options={'include_usage': True})
        started = time.time()
        try:
            with _request('/v1/chat/completions', body, timeout=3600) as response:
                for raw in response:
                    line = raw.decode('utf-8', 'replace').strip()
                    if not line.startswith('data:'):
                        continue
                    data = line[5:].strip()
                    if data == '[DONE]':
                        break
                    print('CC_LLM_SSE:' + data, flush=True)
        except urllib.error.HTTPError as error:
            raise RuntimeError('Chat failed: %s %s' % (error.code, error.read()[:300].decode('utf-8', 'replace')))
        return round(time.time() - started, 2)

    def stop(self):
        """Stop Ollama, which frees all GPU memory it held. Files stay for a quick reload."""
        if self.proc is not None and self.proc.poll() is None:
            try:
                os.killpg(self.proc.pid, signal.SIGTERM)
                self.proc.wait(30)
            except Exception:
                try:
                    os.killpg(self.proc.pid, signal.SIGKILL)
                except Exception:
                    pass
        self.proc = None
        self.context = None
        self.model = None

    unload = stop
