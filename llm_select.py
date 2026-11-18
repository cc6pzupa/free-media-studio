"""Choose the best quantized build of a language model for the hardware at hand.

Runs inside the Kaggle kernel, where Hugging Face is reachable; the pure
functions below are also imported by the local tests. Given a model name
("Qwen3-32B") or repository ("Qwen/Qwen3-32B-GGUF"), it lists GGUF builds on
Hugging Face, reads each file's size, estimates the memory for weights plus
the KV cache, and picks the highest-quality file that fits:

    1. entirely on the GPUs (fast),
    2. split between GPU and system RAM (larger models, slower), or
    3. on the CPU alone (CPU sessions, which use no GPU quota).
"""
import json
import re
import urllib.parse
import urllib.request

HF = 'https://huggingface.co'

# Approximate bits per weight of llama.cpp quantization types, used to rank quality.
QUANT_BITS = {
    'F32': 32, 'BF16': 16, 'F16': 16, 'Q8_0': 8.5, 'Q6_K': 6.56, 'Q5_K_M': 5.69, 'Q5_K_S': 5.54,
    'Q5_1': 6.0, 'Q5_0': 5.5, 'Q4_K_M': 4.85, 'Q4_K_S': 4.58, 'Q4_1': 5.0, 'Q4_0': 4.55,
    'IQ4_NL': 4.5, 'IQ4_XS': 4.25, 'Q3_K_L': 4.27, 'Q3_K_M': 3.91, 'Q3_K_S': 3.5, 'IQ3_M': 3.66,
    'IQ3_S': 3.44, 'IQ3_XS': 3.3, 'IQ3_XXS': 3.06, 'Q2_K_L': 3.2, 'Q2_K': 3.0, 'IQ2_M': 2.7,
    'IQ2_S': 2.5, 'IQ2_XS': 2.31, 'IQ2_XXS': 2.06, 'IQ1_M': 1.75, 'IQ1_S': 1.56,
    # unsloth "dynamic" builds keep sensitive layers at higher precision.
    'Q8_K_XL': 9.0, 'Q6_K_XL': 7.0, 'Q5_K_XL': 6.0, 'Q4_K_XL': 5.0, 'Q3_K_XL': 4.1, 'Q2_K_XL': 3.3,
    'IQ3_XXS_XL': 3.2, 'IQ2_M_XL': 2.9,
}
# Below this, answers degrade noticeably; such files are offered only when nothing else fits.
MIN_GOOD_BITS = 3.9
# Publishers that convert many models carefully; the model's own organization ranks first.
TRUSTED = ('unsloth', 'bartowski', 'lmstudio-community', 'ggml-org', 'mradermacher')

_QUANT = re.compile(r'(?<![A-Za-z0-9])(' + '|'.join(sorted(QUANT_BITS, key=len, reverse=True)) + r')(?![A-Za-z0-9])', re.I)
_SPLIT = re.compile(r'-(\d{5})-of-(\d{5})\.gguf$', re.I)


def quant_of(filename):
    """The llama.cpp quantization named in a GGUF file name, or None."""
    match = _QUANT.search(filename.rsplit('/', 1)[-1].replace('.', '-'))
    return match.group(1).upper() if match else None


def gguf_builds(files):
    """Group a repository's files into builds, keyed by the first file of each:
    {first_path: {'quant': 'Q4_K_M', 'files': [...], 'bytes': n, 'first': path}}.

    Split files (name-00001-of-00003.gguf) are one build; two files of the same
    quantization (Q4_K_M and UD-Q4_K_M) stay separate builds. Vision projector
    files (mmproj) and imatrix data are not model builds.
    """
    builds = {}
    for item in files:
        path = item.get('path') or item.get('rfilename') or ''
        size = item.get('size') or (item.get('lfs') or {}).get('size')
        lower = path.lower()
        if not lower.endswith('.gguf') or 'mmproj' in lower or 'imatrix' in lower or not size:
            continue
        quant = quant_of(path)
        if not quant:
            continue
        split = _SPLIT.search(path)
        if split and split.group(1) != '00001':
            first = path[:split.start()] + '-00001-of-%s.gguf' % split.group(2)
        else:
            first = path
        build = builds.setdefault(first, {'quant': quant, 'files': [], 'bytes': 0, 'first': first})
        build['files'].append(path)
        build['bytes'] += int(size)
    for build in builds.values():
        build['files'].sort()
    return builds


def kv_bytes_per_token(config):
    """fp16 KV cache bytes per token from a Hugging Face config.json (None if unknown)."""
    config = config.get('text_config') or config
    layers = config.get('num_hidden_layers')
    heads = config.get('num_attention_heads')
    kv_heads = config.get('num_key_value_heads') or heads
    head_dim = config.get('head_dim') or (config.get('hidden_size') // heads if heads and config.get('hidden_size') else None)
    if not (layers and kv_heads and head_dim):
        return None
    return 2 * layers * kv_heads * head_dim * 2


def active_fraction(config):
    """Share of weights read per token: below 1 for mixture-of-experts models."""
    config = config.get('text_config') or config
    experts = config.get('num_experts') or config.get('n_routed_experts') or config.get('num_local_experts')
    used = config.get('num_experts_per_tok') or config.get('num_experts_per_token')
    if experts and used and experts > used:
        # Attention and shared layers are always read; experts hold most of the weights.
        return min(1.0, 0.12 + 0.88 * used / experts)
    return 1.0


def plan(bytes_, hardware, kv_per_token=None, context=8192, active=1.0):
    """Where a build of this size runs, and roughly how fast. None if it does not fit at all.

    hardware: {'gpus': [MiB per GPU], 'ram_gib': system RAM, optional 'gpu_rate'
    and 'cpu_rate' in effective bytes/s}. Decoding is limited by memory
    bandwidth, so speed is estimated from the bytes read per token.
    """
    kv = (kv_per_token or 160 * 1024) * context
    gpus = hardware.get('gpus') or []
    # CUDA context, compute buffers and fragmentation per GPU.
    usable_gpu = sum(max(0, mib * 2**20 - 1.6 * 2**30) for mib in gpus)
    usable_ram = max(0, (hardware.get('ram_gib', 0) - 7) * 2**30)  # the system and Python keep some
    need = bytes_ + kv
    per_token = bytes_ * active
    # Effective bytes per second while decoding, measured on Kaggle (Qwen3-8B Q4_K_M):
    # 37 tok/s with llama.cpp on two T4s, 3.6 tok/s on the session's 4 CPU cores.
    gpu_rate = hardware.get('gpu_rate', 185e9)
    cpu_rate = hardware.get('cpu_rate', 18e9)
    if gpus and need <= usable_gpu:
        # Layers are split across GPUs and run one after another, so one GPU's bandwidth applies.
        return {'placement': 'gpu', 'gpu_share': 1.0, 'tok_per_s': round(gpu_rate / per_token, 1), 'need_bytes': need}
    if gpus and need <= usable_gpu + usable_ram:
        share = max(0.0, min(1.0, (usable_gpu - kv) / bytes_))
        seconds = per_token * share / gpu_rate + per_token * (1 - share) / cpu_rate
        return {'placement': 'hybrid', 'gpu_share': round(share, 2), 'tok_per_s': round(1 / seconds, 1), 'need_bytes': need}
    if not gpus and need <= usable_ram:
        return {'placement': 'cpu', 'gpu_share': 0.0, 'tok_per_s': round(cpu_rate / per_token, 1), 'need_bytes': need}
    return None


def choose(builds, hardware, config=None, context=8192, min_tok_per_s=4.0):
    """Pick the best build: highest quality that fits on the GPUs; otherwise the best
    split or CPU build that still reaches min_tok_per_s; otherwise the fastest that fits."""
    config = config or {}
    kv = kv_bytes_per_token(config)
    active = active_fraction(config)
    options = []
    for build in builds.values():
        where = plan(build['bytes'], hardware, kv, context, active)
        if where:
            options.append(dict(build, bits=QUANT_BITS.get(build['quant'], 0), **where))
    if not options:
        return None, []
    rank = {'gpu': 0, 'hybrid': 1, 'cpu': 1}

    any_fast = any(o['tok_per_s'] >= min_tok_per_s and o['bits'] >= MIN_GOOD_BITS for o in options)

    def score(option):
        good = option['bits'] >= MIN_GOOD_BITS
        fast = option['tok_per_s'] >= min_tok_per_s
        if any_fast:
            # Readable and usable builds first; among them fully on GPU, then the most bits.
            return (good and fast, rank[option['placement']] == 0, min(option['bits'], 8.5), option['tok_per_s'])
        # Nothing reaches a usable speed: readable quality, then the fastest.
        return (good, rank[option['placement']] == 0, option['tok_per_s'], option['bits'])
    options.sort(key=score, reverse=True)
    return options[0], options


def _get(url, timeout=30):
    request = urllib.request.Request(url, headers={'User-Agent': 'cc-media-studio'})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def search_repos(name, limit=20):
    """GGUF repositories for a model name, most downloaded first."""
    # The hub's search matches repository names, which seldom contain the original organization.
    query = urllib.parse.urlencode({'search': name.split('/')[-1], 'filter': 'gguf', 'sort': 'downloads', 'direction': -1, 'limit': limit})
    return [item['id'] for item in _get(HF + '/api/models?' + query)]


def repo_files(repo):
    return _get(HF + '/api/models/%s/tree/main?recursive=true' % repo)


def base_config(name, repo):
    """config.json of the original model, for the KV cache size; None if not found."""
    candidates = []
    if '/' in name:
        candidates.append(name.replace('-GGUF', '').replace('-gguf', ''))
    owner = repo.split('/')[0]
    model = repo.split('/')[1]
    for suffix in ('-GGUF', '-gguf', '-Instruct-GGUF'):
        model = model.replace(suffix, '')
    original = name.split('/')[-1]
    # The original first; public mirrors when it is gated (Llama, Gemma).
    candidates += [owner + '/' + model, 'Qwen/' + original, 'unsloth/' + original, 'meta-llama/' + original,
                   'mistralai/' + original, 'google/' + original, 'NousResearch/' + original]
    for candidate in dict.fromkeys(candidates):
        try:
            return _get(HF + '/%s/resolve/main/config.json' % candidate, 15)
        except Exception:
            continue
    return None


KNOWN_ORGS = ('qwen', 'metallama', 'mistralai', 'google', 'microsoft', 'deepseekai', 'nvidia', 'ibmgranite', 'thudm', 'zaiorg', 'openai')


def _norm(text):
    return re.sub(r'[^a-z0-9]', '', text.lower())


def same_model(name, repo):
    """True when the repository converts exactly this model: not a fine-tune,
    an "abliterated" or merged variant, or another model built on it."""
    wanted = _norm(name.split('/')[-1]).replace('gguf', '')
    owner = _norm(name.split('/')[0]) if '/' in name else None
    model = _norm(re.sub(r'[-_.]?gguf$', '', repo.split('/')[-1], flags=re.I))
    if not model.endswith(wanted):
        return False
    prefix = model[:-len(wanted)] if wanted else model
    # bartowski writes "Qwen_Qwen3-32B", others "meta-llama.Llama-3.3-70B-Instruct".
    return prefix == '' or prefix == owner or prefix in KNOWN_ORGS


def rank_repos(name, repos):
    """The model's own organization, then trusted converters, then by downloads (input order)."""
    wanted = name.split('/')[-1].lower().replace('-gguf', '')
    owner = name.split('/')[0].lower() if '/' in name else None

    def key(index_repo):
        index, repo = index_repo
        org, model = repo.split('/')
        exact = wanted in model.lower()
        own = org.lower() == owner if owner else model.lower().startswith(wanted) and org.lower() in ('qwen', 'meta-llama', 'mistralai', 'google', 'microsoft', 'deepseek-ai')
        trusted = org.lower() in TRUSTED
        return (not exact, not own, not trusted, index)
    return [repo for _, repo in sorted(enumerate(repos), key=key)]


def find(name, hardware, context=8192, max_repos=4, single_file=True):
    """Search Hugging Face and return the chosen build with its repository and alternatives.

    single_file: skip builds split across several files (Ollama pulls one GGUF file).
    """
    if name.lower().endswith('gguf') and '/' in name:
        repos = [name]
    else:
        repos = rank_repos(name, [r for r in search_repos(name, 40) if same_model(name, r)])[:max_repos]
    best = None
    considered = []
    for repo in repos:
        try:
            builds = gguf_builds(repo_files(repo))
        except Exception:
            continue
        if single_file:
            builds = {key: build for key, build in builds.items() if len(build['files']) == 1}
        if not builds:
            continue
        config = base_config(name, repo) or {}
        chosen, options = choose(builds, hardware, config, context)
        considered.append({'repo': repo, 'options': [{k: o[k] for k in ('quant', 'first', 'bytes', 'placement', 'tok_per_s')} for o in options]})
        if chosen and best is None:
            best = dict(chosen, repo=repo, kv_bytes_per_token=kv_bytes_per_token(config), moe=active_fraction(config) < 1)
    return best, considered
