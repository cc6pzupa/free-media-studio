"""Language-model session adapter: prepares a model on the Kaggle kernel and
streams chat completions back through the Jupyter connection."""
import json
import re
import threading
from pathlib import Path

from session_runtime import JupyterTransport, ModelSession
from session_backend import RELEASE_FAILURE_CODE, FREE_GPU_CODE

LLM_FILES = ('llm_select.py', 'remote_llm.py')
LLM_MODEL_ID = 'language-model'
NAME = re.compile(r'^[A-Za-z0-9][A-Za-z0-9._/-]{0,119}$')
CONTEXTS = (2048, 4096, 8192, 16384, 32768)
ROLES = ('system', 'user', 'assistant')
# OpenAI chat parameters passed through to the model; everything else is dropped.
NUMBERS = {'temperature': (0, 2), 'top_p': (0, 1), 'presence_penalty': (-2, 2), 'frequency_penalty': (-2, 2),
           'max_tokens': (1, 32768), 'seed': (0, 2**31 - 1)}
MAX_PROMPT_CHARS = 400_000


def validate_name(name):
    if not isinstance(name, str) or not NAME.match(name.strip()) or '..' in name:
        raise ValueError('模型名称无效。请输入如 Qwen3-8B 或 Qwen/Qwen3-8B-GGUF。')
    return name.strip()


def validate_context(context):
    context = int(context or 8192)
    if context not in CONTEXTS:
        raise ValueError('上下文长度需为 %s 之一。' % '、'.join(map(str, CONTEXTS)))
    return context


def validate_choice(choice):
    if choice is None:
        return None
    if not isinstance(choice, dict) or not isinstance(choice.get('repo'), str) or not isinstance(choice.get('first'), str):
        raise ValueError('所选版本无效。')
    if not NAME.match(choice['repo']) or not re.fullmatch(r'[A-Za-z0-9._/-]{1,200}\.gguf', choice['first']):
        raise ValueError('所选版本无效。')
    return {'repo': choice['repo'], 'first': choice['first']}


def chat_body(body):
    """A clean OpenAI chat request: messages plus known numeric options."""
    if not isinstance(body, dict):
        raise ValueError('请求格式无效。')
    messages = body.get('messages')
    if not isinstance(messages, list) or not messages or len(messages) > 400:
        raise ValueError('messages 需为非空列表。')
    clean = []
    total = 0
    for message in messages:
        if not isinstance(message, dict) or message.get('role') not in ROLES:
            raise ValueError('每条消息需有 role（system、user 或 assistant）。')
        content = message.get('content')
        if isinstance(content, list):
            # OpenAI content parts: keep the text parts.
            content = ''.join(part.get('text', '') for part in content if isinstance(part, dict) and part.get('type') == 'text')
        if not isinstance(content, str):
            raise ValueError('消息内容需为文本。')
        total += len(content)
        clean.append({'role': message['role'], 'content': content})
    if total > MAX_PROMPT_CHARS:
        raise ValueError('对话内容过长。')
    result = {'messages': clean}
    for key, (low, high) in NUMBERS.items():
        value = body.get(key)
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not low <= value <= high:
            raise ValueError('%s 超出范围。' % key)
        result[key] = int(value) if key in ('max_tokens', 'seed') else float(value)
    stop = body.get('stop')
    if stop is not None:
        stops = [stop] if isinstance(stop, str) else stop
        if not isinstance(stops, list) or len(stops) > 4 or not all(isinstance(s, str) and 0 < len(s) <= 64 for s in stops):
            raise ValueError('stop 无效。')
        result['stop'] = stops
    return result


# Kernel code that leaves image and video models and keeps one LlmRuntime per kernel.
STOP_LLM_CODE = "if globals().get('_CC_LLM') is not None: globals()['_CC_LLM'].stop()\n"


def llm_bootstrap_code():
    sources = {name: Path(__file__).with_name(name).read_text(encoding='utf-8') for name in LLM_FILES}
    return f'''
import sys,json,gc
from pathlib import Path
_cc_llm_modules=Path('/kaggle/temp/cc-llm-runtime')
_cc_llm_modules.mkdir(parents=True,exist_ok=True)
for _name,_source in {sources!r}.items():
    (_cc_llm_modules/_name).write_text(_source,encoding='utf-8')
if str(_cc_llm_modules) not in sys.path:sys.path.insert(0,str(_cc_llm_modules))
if globals().get('_CC_MODEL_ID')!={LLM_MODEL_ID!r}:
    if (globals().get('pipe') is not None or globals().get('encoder') is not None) and not (globals().get('_CC_MANAGED_MODEL') or globals().get('_COMFY_QWEN_STATE')):
        raise RuntimeError('Unmanaged model exists; select the managed notebook kernel')
    # Free image and video models: the language model needs the GPU memory.
    if globals().get('_CC_VIDEO_PROCESS') is not None:globals()['_CC_VIDEO_PROCESS'].close()
    if globals().get('_CC_VIDEO_RUNTIME') is not None:globals()['_CC_VIDEO_RUNTIME'].close()
    for _name in ('pipe','encoder','transformer','vae','processor','scheduler','generate_image','_COMFY_QWEN_STATE','_CC_NEGATIVE_CACHE','_CC_PROMPT_CACHE','_CC_DUAL_STATE','_CC_VIDEO_RUNTIME','_CC_VIDEO_PROCESS','_CC_TASK_PIPES','_value','_module','_decoded'):
        globals().pop(_name,None)
    exec(compile({RELEASE_FAILURE_CODE!r},'<cc-release-failure>','exec'),globals())
    gc.collect()
    if 'torch' in sys.modules:
        try:exec(compile({FREE_GPU_CODE!r},'<cc-free-gpu>','exec'),globals())
        except ImportError:pass
for _module in {LLM_FILES!r}:sys.modules.pop(_module[:-3],None)
import remote_llm
def _cc_llm_progress(event):print('COMFY_PROGRESS:'+json.dumps(event),flush=True)
if globals().get('_CC_LLM') is None:
    _CC_LLM=remote_llm.LlmRuntime(_cc_llm_progress)
else:
    # Keep the running Ollama process; pick up the latest code for new calls.
    _cc_old=_CC_LLM;_CC_LLM=remote_llm.LlmRuntime(_cc_llm_progress)
    _CC_LLM.proc,_CC_LLM.context,_CC_LLM.model,_CC_LLM.info=_cc_old.proc,_cc_old.context,_cc_old.model,_cc_old.info
_CC_MODEL_ID={LLM_MODEL_ID!r};_CC_MANAGED_MODEL=True
print('CC_LLM_BOOT',flush=True)
'''


def prepare_code(name, context, choice=None):
    args = json.dumps({'name': name, 'context': context, 'choice': choice})
    return f'''
import json
assert globals().get('_CC_MODEL_ID')=={LLM_MODEL_ID!r},'Active model changed'
_cc_llm_args=json.loads({args!r})
try:
    _CC_LLM.prepare(_cc_llm_args['name'],_cc_llm_args['context'],_cc_llm_args['choice'])
except ValueError as _cc_error:
    print('CC_LLM_ERROR:'+json.dumps(str(_cc_error)),flush=True)
'''


def chat_code(body):
    return f'''
import json
assert globals().get('_CC_MODEL_ID')=={LLM_MODEL_ID!r},'Active model changed'
_cc_llm_seconds=_CC_LLM.chat(json.loads({json.dumps(body)!r}))
print('CC_LLM_DONE:'+json.dumps({{'seconds':_cc_llm_seconds}}),flush=True)
'''


def status_code():
    return f'''
import json
_cc_llm=globals().get('_CC_LLM')
print('CC_LLM_STATUS:'+json.dumps({{'loaded':bool(_cc_llm and _cc_llm.model and _cc_llm.alive()),'info':_cc_llm.info if _cc_llm and _cc_llm.alive() else None}}),flush=True)
'''


class LlmSessionBackend:
    model_id = LLM_MODEL_ID

    def __init__(self, url, transport_factory=None):
        self.lock = threading.Lock()
        self.listener = None
        self.buffer = ''
        self.info = None
        self.session = ModelSession(transport_factory or (lambda: JupyterTransport(url, output_callback=self._output)),
                                    llm_bootstrap_code(), lifetime_seconds=None)

    def _output(self, text):
        self.buffer += text
        lines = self.buffer.split('\n')
        self.buffer = lines.pop()
        listener = self.listener
        for line in lines:
            if not listener:
                continue
            try:
                if line.startswith('COMFY_PROGRESS:'):
                    listener('progress', json.loads(line.split(':', 1)[1]))
                elif line.startswith('CC_LLM_SSE:'):
                    listener('chunk', json.loads(line.split(':', 1)[1]))
            except (ValueError, TypeError):
                pass
        if len(self.buffer) > 4 * 1024 * 1024:
            self.buffer = ''

    def _run(self, code, timeout, listener):
        with self.lock:
            self.listener = listener
            try:
                return self.session.execute(code, timeout=timeout)
            finally:
                self.listener = None
                self.buffer = ''

    def prepare(self, name, context=8192, choice=None, progress=None):
        name, context, choice = validate_name(name), validate_context(context), validate_choice(choice)
        notify = progress or (lambda event: None)
        try:
            output = self._run(prepare_code(name, context, choice), 7200,
                               lambda kind, data: notify(data) if kind == 'progress' else None)
        except Exception:
            raise RuntimeError('准备语言模型失败，请查看专用 Notebook 日志。') from None
        for line in output.splitlines():
            if line.startswith('CC_LLM_ERROR:'):
                raise ValueError('找不到能在本会话运行的版本：' + json.loads(line.split(':', 1)[1])[:200])
            if line.startswith('CC_LLM_READY:'):
                self.info = json.loads(line.split(':', 1)[1])
                return self.info
        raise RuntimeError('准备语言模型失败：没有返回结果。')

    def chat(self, body, on_chunk):
        """Stream a chat completion; on_chunk receives each OpenAI chunk dict."""
        body = chat_body(body)
        if not self.info:
            raise ValueError('请先选择并加载语言模型。')
        try:
            output = self._run(chat_code(body), 3600, lambda kind, data: on_chunk(data) if kind == 'chunk' else None)
        except Exception:
            raise RuntimeError('语言模型请求失败，请查看专用 Notebook 日志。') from None
        for line in output.splitlines():
            if line.startswith('CC_LLM_DONE:'):
                return json.loads(line.split(':', 1)[1])
        raise RuntimeError('语言模型请求没有完成。')

    def close(self):
        self.session.close()
