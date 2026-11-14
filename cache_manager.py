"""Private, create-only Kaggle cache of prepared INT8/Q4 model assets.

Executed in the user's Kaggle kernel. Credentials are passed at connection time,
used only for Kaggle operations, and removed by the caller's finally block.
"""
import contextlib
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor

CACHE_VERSION = 1


def cache_file_hashes(folder):
    hashes = {}
    for file in Path(folder).rglob('*'):
        if file.is_file():
            with file.open('rb') as content:
                hashes[file.relative_to(folder).as_posix()] = hashlib.file_digest(content,'sha256').hexdigest()
    return hashes


@contextlib.contextmanager
def cache_credentials(handle, token):
    owner = handle.split('/')[0]
    values = {'KAGGLE_USERNAME': owner,'KAGGLE_API_TOKEN':None,'KAGGLE_KEY':None}
    if token.startswith('KGAT_'): values['KAGGLE_API_TOKEN'] = token
    else: values['KAGGLE_KEY'] = token
    old = {name: os.environ.get(name) for name in values}
    try:
        for name,value in values.items():
            if value is None: os.environ.pop(name,None)
            else: os.environ[name]=value
        yield
    finally:
        for name, value in old.items():
            if value is None: os.environ.pop(name, None)
            else: os.environ[name] = value


def cache_metadata(handle, token):
    owner, slug = handle.split('/')
    with requests.Session() as http:
        if token.startswith('KGAT_'): http.headers['Authorization'] = 'Bearer ' + token
        else: http.auth = (owner, token)
        response = http.get('https://www.kaggle.com/api/v1/datasets/view/'+owner+'/'+slug, timeout=(20,60))
        if response.status_code in (403,404):
            # Kaggle hides missing private slugs behind 403. Verify absence in
            # the authenticated owner listing; a permission failure stays fatal.
            for page in range(1,101):
                listing = http.get('https://www.kaggle.com/api/v1/datasets/list',
                    params={'user':owner,'search':slug,'page':page}, timeout=(20,60))
                listing.raise_for_status()
                records = listing.json()
                if not isinstance(records,list): raise RuntimeError('Invalid dataset listing')
                if any(item.get('ref') == handle for item in records):
                    raise RuntimeError('Existing cache is inaccessible')
                if len(records)<20: return None
            raise RuntimeError('Cannot confirm cache absence')
        response.raise_for_status()
        data = response.json()
        if data.get('isPrivate') is not True:
            raise RuntimeError('模型缓存必须为私有 Dataset；没有修改现有 Dataset。')
        return data


def validate_cache_manifest(manifest):
    generic=manifest.get('studio_model_id') is not None
    if generic:
        if (manifest.get('studio_cache_version')!=2 or manifest['studio_model_id']!=globals().get('_CC_MODEL_ID')
            or manifest.get('model_sources')!=globals().get('_CC_MODEL_SOURCES')):
            raise RuntimeError('所选模型缓存版本不兼容。')
    elif (manifest.get('studio_cache_version') != CACHE_VERSION
        or manifest.get('base_revision') != BASE_REVISION
        or manifest.get('gguf_revision') != GGUF_REVISION):
        raise RuntimeError('同名 Dataset 不是兼容的本程序缓存，没有覆盖它。')
    files = manifest.get('cache_files', {})
    if not files or (not generic and 'transformer-q4.gguf' not in files) or (generic and not any(name.endswith('.safetensors') for name in files)):
        raise RuntimeError('模型缓存文件清单不完整。')
    for name, size in files.items():
        safe_relative(name)
        if not isinstance(size,int) or size<0: raise RuntimeError('Invalid cached file size')
    return manifest


def validate_cache(folder):
    folder = Path(folder)
    manifest = validate_cache_manifest(json.loads((folder/'asset-sources.json').read_text()))
    for name, size in manifest['cache_files'].items():
        file = folder/name
        if not file.is_file() or file.stat().st_size != size:
            raise RuntimeError('模型缓存文件不完整。')
    return manifest


def cache_http(handle, token):
    http = requests.Session()
    if token.startswith('KGAT_'): http.headers['Authorization'] = 'Bearer '+token
    else: http.auth = (handle.split('/')[0],token)
    return http


def cache_file_url(handle, version, name):
    safe_relative(name)
    return ('https://www.kaggle.com/api/v1/datasets/download/'+handle+'/'+quote(name,safe='')+
        '?raw=true&dataset_version_number='+str(int(version)))


def read_cache_manifest(http, handle, version):
    response = http.get(cache_file_url(handle,version,'asset-sources.json'),timeout=(20,60))
    response.raise_for_status()
    return validate_cache_manifest(response.json())


def download_cache_assets(folder, handle, token, metadata):
    folder = Path(folder); folder.mkdir(parents=True,exist_ok=True)
    version = metadata['currentVersionNumber']
    with cache_http(handle,token) as http:
        manifest = read_cache_manifest(http,handle,version)
    total = sum(manifest['cache_files'].values()); received={}; last=[0.0]
    progress_lock=threading.Lock()
    def transfer(item):
        name,size=item
        def progress(value):
            with progress_lock:
                received[name]=value
                now=time.monotonic()
                if now-last[0]>=.8 or value==size:
                    emit_progress('cache-download','正在拉取私有 Dataset 模型缓存。',sum(received.values()),total,name)
                    last[0]=now
        with cache_http(handle,token) as http:
            download_file(http,cache_file_url(handle,version,name),folder/name,size,
                manifest.get('cache_sha256',{}).get(name),progress)
    # Independent resumable streams avoid serializing all multi-GB shards.
    files=sorted(manifest['cache_files'].items(),key=lambda item:item[1],reverse=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(transfer,files))
    (folder/'asset-sources.json').write_text(json.dumps(manifest,indent=2))
    validate_cache(folder)
    return folder


def prepare_auto_assets(folder):
    handle, token = globals().get('_QWEN_CACHE_HANDLE'), globals().get('_QWEN_CACHE_TOKEN')
    globals()['_QWEN_CACHE_STATUS'] = 'disabled'
    globals()['_QWEN_CACHE_STATUS_HANDLE'] = handle
    if handle and token:
        try:
            emit_progress('cache-check', '正在检查你的私有模型缓存。')
            metadata = cache_metadata(handle, token)
            if metadata:
                emit_progress('cache-download', '正在拉取私有 Dataset 模型缓存。')
                download_cache_assets(folder, handle, token, metadata)
                globals()['_QWEN_CACHE_STATUS'] = 'saved'
                globals()['_QWEN_RESOLVED_SOURCE'] = 'kaggle-cache'
                return Path(folder), False
            globals()['_QWEN_CACHE_STATUS'] = 'missing'
        except Exception:
            # A metadata/auth/compatibility error must never permit overwriting
            # an existing dataset. Generation can still use the public weights.
            globals()['_QWEN_CACHE_STATUS'] = 'unavailable'
            emit_progress('cache-warning', '私有缓存暂不可用，本次从 Hugging Face 准备；没有修改已有 Dataset。')
    globals()['_QWEN_RESOLVED_SOURCE'] = 'huggingface'
    return prepare_hf_assets(folder), True


def save_private_cache(folder, encoder):
    handle, token = globals().get('_QWEN_CACHE_HANDLE'), globals().get('_QWEN_CACHE_TOKEN')
    status = globals().get('_QWEN_CACHE_STATUS', 'missing') if globals().get('_QWEN_CACHE_STATUS_HANDLE') == handle else 'missing'
    globals()['_QWEN_CACHE_STATUS_HANDLE'] = handle
    if not handle or not token or status in ('saved','unavailable'): return
    try:
        emit_progress('cache-check', '正在确认私有缓存状态。')
        metadata = cache_metadata(handle, token)
        if metadata:
            with cache_http(handle,token) as http:
                read_cache_manifest(http,handle,metadata['currentVersionNumber'])
            globals()['_QWEN_CACHE_STATUS'] = 'saved'
            return
        folder = Path(folder)
        emit_progress('cache-prepare', '正在保存 INT8 编码器和 Q4 权重，首次需要额外等待。')
        generic=globals().get('_CC_MODEL_ID','qwen-image-2.1')!='qwen-image-2.1'
        # A bundle left by an interrupted save is never reused; remove it.
        for stale in Path('/kaggle/temp').glob('qwen-private-cache-*'):
            shutil.rmtree(stale, ignore_errors=True)
        # Qwen's bundle copies its weights (other models use hardlinks), so the
        # copy needs room within the model disk budget too.
        if not generic and callable(globals().get('release_model_disk')):
            release_model_disk(folder, _disk_bytes([folder]), count_existing=False,
                notify=lambda names: emit_progress('cleanup', '正在清理较早使用的模型文件以释放磁盘：'+'、'.join(names)))
        with tempfile.TemporaryDirectory(prefix='qwen-private-cache-', dir='/kaggle/temp') as temporary:
            bundle = Path(temporary)
            if generic:
                def link_or_copy(src,dst):
                    try:os.link(src,dst)
                    except OSError:shutil.copy2(src,dst)
                shutil.copytree(folder,bundle,dirs_exist_ok=True,copy_function=link_or_copy,
                    ignore=shutil.ignore_patterns('*.partial','*.verified.json','asset-sources.json'))
            for name in (() if generic else ('processor','scheduler','vae','transformer')):
                shutil.copytree(folder/name, bundle/name, ignore=shutil.ignore_patterns('*.partial','*.verified.json'))
            for name in (() if generic else ('transformer-q4.gguf','model_index.json','LICENSE')):
                if (folder/name).exists(): shutil.copy2(folder/name, bundle/name)
            if not generic:encoder.save_pretrained(bundle/'text_encoder', safe_serialization=True, max_shard_size='4GB')
            manifest = json.loads((folder/'asset-sources.json').read_text())
            manifest.update(studio_cache_version=2 if generic else CACHE_VERSION,
                cache_files={p.relative_to(bundle).as_posix():p.stat().st_size for p in bundle.rglob('*') if p.is_file()},
                cache_sha256=cache_file_hashes(bundle))
            (bundle/'asset-sources.json').write_text(json.dumps(manifest, indent=2))
            validate_cache(bundle)
            emit_progress('cache-upload', '正在上传私有 Dataset；Kaggle 将显示文件上传进度。')
            # Use KaggleHub's create-only primitive: its public upload helper
            # creates a new version on conflict, which could overwrite user data.
            from kagglehub import datasets
            api = datasets.dataset_upload.__globals__
            create = datasets.create_dataset_or_version.__globals__['_create_dataset']
            with cache_credentials(handle, token):
                uploads = api['upload_files_and_directories'](str(bundle), item_type=api['ApiBlobType'].DATASET,
                    ignore_patterns=api['normalize_patterns'](default=api['DEFAULT_IGNORE_PATTERNS'], additional=None))
                create(api['parse_dataset_handle'](handle), uploads)
            # The create primitive explicitly sends is_private=True. Confirm
            # privacy again from the server before reporting success.
            emit_progress('cache-processing', '文件已上传，Kaggle 正在处理私有 Dataset。')
            deadline = time.monotonic()+180
            while not cache_metadata(handle, token):
                if time.monotonic() >= deadline: raise RuntimeError('Dataset metadata not ready')
                time.sleep(5)
        globals()['_QWEN_CACHE_STATUS'] = 'saved'
        emit_progress('cache-saved', '私有模型缓存已保存，下次新会话优先拉取，无需重新下载或量化。')
    except Exception as error:
        globals()['_QWEN_CACHE_STATUS'] = 'failed'
        globals()['_QWEN_CACHE_FAILURE_TYPE'] = type(error).__name__
        emit_progress('cache-warning', '私有缓存保存未完成；本次仍可正常生图，重新连接后会再次检查。')
