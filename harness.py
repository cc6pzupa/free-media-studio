"""Find AI tools on this computer and connect them to the Kaggle language model.

Every tool here speaks the OpenAI chat-completions API; connecting one points
it at this app's local endpoint (http://127.0.0.1:<port>/v1) with the local
API key. Changes are additive: a provider, model entry or separate config
file is added, the tool's own defaults are left alone, and any file that is
changed is copied to <name>.bak-ccmedia first. Installs run fixed commands
only.
"""
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path

PROVIDER = 'cc-media'
LABEL = 'CC Media (Kaggle)'


def home():
    return Path(os.environ.get('CC_MEDIA_HOME') or Path.home())


def _which(name):
    return shutil.which(name)


def _version(command):
    try:
        out = subprocess.run([command, '--version'], capture_output=True, text=True, timeout=8)
        text = (out.stdout or out.stderr).strip().splitlines()
        return text[0][:60] if text else None
    except Exception:
        return None


def _backup(path):
    if path.exists():
        backup = path.with_name(path.name + '.bak-ccmedia')
        shutil.copy2(path, backup)
        return str(backup)
    return None


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.ccmedia-tmp')
    temporary.write_text(text, encoding='utf-8')
    os.replace(temporary, path)


# ---- OpenCode: a custom provider in the global opencode.json ----

def _opencode_paths():
    base = Path(os.environ.get('XDG_CONFIG_HOME') or home() / '.config') / 'opencode'
    return base / 'opencode.json', base / 'opencode.jsonc'


def opencode_state():
    json_path, jsonc_path = _opencode_paths()
    if json_path.exists():
        try:
            config = json.loads(json_path.read_text(encoding='utf-8'))
            return {'configured': PROVIDER in (config.get('provider') or {}), 'path': str(json_path)}
        except ValueError:
            return {'configured': False, 'path': str(json_path), 'manual': 'opencode.json is not plain JSON'}
    if jsonc_path.exists():
        text = jsonc_path.read_text(encoding='utf-8')
        return {'configured': '"%s"' % PROVIDER in text, 'path': str(jsonc_path), 'manual': 'opencode.jsonc has comments'}
    return {'configured': False, 'path': str(json_path)}


def opencode_configure(base_url, api_key, model):
    json_path, jsonc_path = _opencode_paths()
    if jsonc_path.exists() and not json_path.exists():
        raise ValueError('OpenCode 使用带注释的 opencode.jsonc，请按下方设置手动添加。')
    config = {}
    if json_path.exists():
        try:
            config = json.loads(json_path.read_text(encoding='utf-8'))
        except ValueError:
            raise ValueError('opencode.json 不是标准 JSON，请手动添加。') from None
        if not isinstance(config, dict):
            raise ValueError('opencode.json 格式无法识别，请手动添加。')
    config.setdefault('$schema', 'https://opencode.ai/config.json')
    providers = config.setdefault('provider', {})
    entry = providers.get(PROVIDER) or {}
    models = dict(entry.get('models') or {})
    models[model] = {'name': model + ' (Kaggle)'}
    providers[PROVIDER] = {'npm': '@ai-sdk/openai-compatible', 'name': LABEL,
                           'options': {'baseURL': base_url, 'apiKey': api_key}, 'models': models}
    backup = _backup(json_path)
    _write(json_path, json.dumps(config, indent=2, ensure_ascii=False) + '\n')
    return {'files': [str(json_path)], 'backup': backup, 'run': 'opencode  →  /models → %s' % LABEL}


# ---- Aider: a separate config file, used with --config ----

def _aider_path():
    return home() / '.aider.cc-media.conf.yml'


def aider_configure(base_url, api_key, model):
    path = _aider_path()
    backup = _backup(path)
    _write(path, '# Written by CC Media Studio: the Kaggle language model.\n'
                 'openai-api-base: %s\nopenai-api-key: %s\nmodel: openai/%s\n' % (base_url, api_key, model))
    return {'files': [str(path)], 'backup': backup, 'run': 'aider --config "%s"' % path}


# ---- Continue: one model entry in ~/.continue/config.yaml ----

def _continue_path():
    return home() / '.continue' / 'config.yaml'


def _continue_entry(base_url, api_key, model, indent):
    pad = ' ' * indent
    return ''.join(pad + line + '\n' for line in (
        '- name: %s (Kaggle)' % model, '  provider: openai', '  model: %s' % model,
        '  apiBase: %s' % base_url, '  apiKey: %s' % api_key, '  roles:', '    - chat', '    - edit'))


def continue_state():
    path = _continue_path()
    if not path.exists():
        return {'configured': False, 'path': str(path)}
    return {'configured': 'apiBase: http://127.0.0.1' in path.read_text(encoding='utf-8'), 'path': str(path)}


def continue_configure(base_url, api_key, model):
    path = _continue_path()
    if not path.exists():
        backup = None
        text = 'name: Local Config\nversion: 1.0.0\nschema: v1\nmodels:\n' + _continue_entry(base_url, api_key, model, 2)
    else:
        text = path.read_text(encoding='utf-8')
        lines = text.splitlines(keepends=True)
        index = next((i for i, line in enumerate(lines) if re.fullmatch(r'models:\s*', line.rstrip('\n'))), None)
        if index is None:
            if re.search(r'^models\s*:', text, re.M):
                raise ValueError('config.yaml 的 models 写法无法自动修改，请按下方设置手动添加。')
            text = text.rstrip('\n') + '\nmodels:\n' + _continue_entry(base_url, api_key, model, 2)
        else:
            following = next((line for line in lines[index + 1:] if line.strip() and not line.lstrip().startswith('#')), '')
            match = re.match(r'( *)- ', following)
            if following and not match:
                raise ValueError('config.yaml 的 models 写法无法自动修改，请按下方设置手动添加。')
            indent = len(match.group(1)) if match else 2
            lines.insert(index + 1, _continue_entry(base_url, api_key, model, indent))
            text = ''.join(lines)
        backup = _backup(path)
    _write(path, text)
    return {'files': [str(path)], 'backup': backup, 'run': 'Continue → 模型列表 → %s (Kaggle)' % model}


# ---- Desktop apps: settings shown for copying ----

def _app_installed(names):
    system = platform.system()
    candidates = []
    for name in names:
        if system == 'Darwin':
            candidates += [Path('/Applications') / (name + '.app'), home() / 'Applications' / (name + '.app')]
        elif system == 'Windows':
            local = Path(os.environ.get('LOCALAPPDATA', home() / 'AppData' / 'Local'))
            candidates += [local / 'Programs' / name, Path(os.environ.get('ProgramFiles', 'C:/Program Files')) / name]
        else:
            candidates += [Path('/opt') / name, home() / 'Applications' / (name + '.AppImage')]
    return any(path.exists() for path in candidates)


def _installer():
    """A Python tool installer on PATH, for Python command-line tools."""
    if _which('uv'):
        return ['uv', 'tool', 'install', '--python', '3.12']
    if _which('pipx'):
        return ['pipx', 'install']
    return None


def tools():
    """Every known tool with what this app can do for it."""
    uv = _installer()
    vscode = any((home() / folder / 'extensions').glob('continue.continue-*') for folder in ('.vscode', '.vscode-insiders', '.cursor'))
    return [
        {'id': 'opencode', 'name': 'OpenCode', 'kind': 'cli', 'binary': 'opencode',
         'installed': bool(_which('opencode')), 'state': opencode_state(), 'configure': True,
         'install': ['npm', 'install', '-g', 'opencode-ai'] if _which('npm') else None,
         'install_hint': '需要 Node.js（npm）。', 'site': 'https://opencode.ai'},
        {'id': 'aider', 'name': 'Aider', 'kind': 'cli', 'binary': 'aider',
         'installed': bool(_which('aider')), 'state': {'configured': _aider_path().exists(), 'path': str(_aider_path())}, 'configure': True,
         'install': (uv + ['aider-chat']) if uv else None, 'install_hint': '需要 uv 或 pipx。', 'site': 'https://aider.chat'},
        {'id': 'continue', 'name': 'Continue (VS Code / JetBrains)', 'kind': 'editor',
         'installed': vscode or _continue_path().parent.exists(), 'state': continue_state(), 'configure': True,
         'install': None, 'install_hint': '在 VS Code 或 JetBrains 的扩展市场安装 Continue。', 'site': 'https://continue.dev'},
        {'id': 'open-webui', 'name': 'Open WebUI', 'kind': 'web', 'binary': 'open-webui',
         'installed': bool(_which('open-webui')), 'state': {'configured': False}, 'configure': False, 'launch': True,
         'install': (['uv', 'tool', 'install', '--python', '3.11', 'open-webui'] if _which('uv') else (['pipx', 'install', 'open-webui'] if _which('pipx') else None)),
         'install_hint': '需要 uv 或 pipx；首次安装较大（约 1–2 GB）。', 'site': 'https://openwebui.com'},
        {'id': 'cherry-studio', 'name': 'Cherry Studio', 'kind': 'app', 'installed': _app_installed(['Cherry Studio', 'cherry-studio']),
         'state': {'configured': False}, 'configure': False, 'manual': True, 'site': 'https://cherry-ai.com'},
        {'id': 'chatbox', 'name': 'Chatbox', 'kind': 'app', 'installed': _app_installed(['Chatbox', 'xyz.chatboxapp.app']),
         'state': {'configured': False}, 'configure': False, 'manual': True, 'site': 'https://chatboxai.app'},
        {'id': 'codex', 'name': 'Codex CLI', 'kind': 'cli', 'binary': 'codex', 'installed': bool(_which('codex')),
         'state': {'configured': False}, 'configure': False,
         'unsupported': 'Codex 只接受 OpenAI Responses API（已移除 chat completions），本程序暂不提供。'},
    ]


def discover():
    result = []
    for tool in tools():
        public = {k: v for k, v in tool.items() if k not in ('install', 'binary')}
        public['can_install'] = bool(tool.get('install')) and not tool['installed']
        if tool['installed'] and tool.get('binary'):
            public['version'] = _version(tool['binary'])
        result.append(public)
    return result


CONFIGURE = {'opencode': opencode_configure, 'aider': aider_configure, 'continue': continue_configure}
NAME = re.compile(r'^[A-Za-z0-9][A-Za-z0-9._/-]{0,119}$')


def configure(tool_id, base_url, api_key, model):
    if tool_id not in CONFIGURE:
        raise ValueError('这个工具不能自动配置。')
    if not isinstance(model, str) or not NAME.match(model):
        raise ValueError('请先加载模型或填写模型名称。')
    if not re.fullmatch(r'http://127\.0\.0\.1:\d{1,5}/v1', base_url) or not re.fullmatch(r'[A-Za-z0-9_-]{16,}', api_key or ''):
        raise ValueError('本机接口地址或 Key 无效。')
    return CONFIGURE[tool_id](base_url, api_key, model)


def install_command(tool_id):
    tool = next((t for t in tools() if t['id'] == tool_id), None)
    if tool is None or not tool.get('install'):
        raise ValueError('这个工具不能在这里自动安装。')
    command = list(tool['install'])
    # npm and uv are .cmd shims on Windows.
    resolved = shutil.which(command[0])
    if not resolved:
        raise ValueError('找不到 %s。' % command[0])
    command[0] = resolved
    return command


def install(tool_id, emit):
    """Run the tool's fixed install command; emit({'line': ...}) for each output line."""
    command = install_command(tool_id)
    emit({'line': '$ ' + ' '.join(Path(command[0]).name if i == 0 else part for i, part in enumerate(command))})
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors='replace')
    for line in process.stdout:
        emit({'line': line.rstrip()[:400]})
    code = process.wait(timeout=3600)
    if code != 0:
        raise RuntimeError('安装失败（退出码 %d）。' % code)
    return {'installed': bool(shutil.which(tool_id if tool_id != 'open-webui' else 'open-webui'))}


def launch_open_webui(base_url, api_key, port=8080):
    """Start Open WebUI connected to this app's endpoint; it opens at http://127.0.0.1:<port>."""
    binary = _which('open-webui')
    if not binary:
        raise ValueError('没有找到 Open WebUI。')
    env = dict(os.environ, OPENAI_API_BASE_URL=base_url, OPENAI_API_KEY=api_key, ENABLE_OLLAMA_API='false', PORT=str(port))
    flags = {'creationflags': subprocess.CREATE_NEW_PROCESS_GROUP} if sys.platform == 'win32' else {'start_new_session': True}
    subprocess.Popen([binary, 'serve', '--host', '127.0.0.1', '--port', str(port)], env=env,
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, **flags)
    return {'url': 'http://127.0.0.1:%d' % port}
