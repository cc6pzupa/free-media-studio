"""Download public HF weights with resumable transfers and real byte progress."""
import hashlib
import json
import os
import shutil
import time
from pathlib import Path, PurePosixPath
from urllib.parse import quote
import requests

BASE_REPO = 'Qwen/Qwen-Image-2.1'
GGUF_REPO = 'azukivc/Qwen-Image-2.1-GGUF'
BASE_REVISION = 'd26bb61231c349cf6b7896fa83353113880e1ba3'
GGUF_REVISION = '4e43fd824763d3d24233630c343462592f68ca49'


def emit_progress(phase, message, done=None, total=None, filename=None):
    event = {'phase': phase, 'message': message}
    if done is not None: event['downloaded_bytes'] = done
    if total: event.update(total_bytes=total, percent=round(min(100, 100*done/total), 1))
    if filename: event['file'] = filename
    callback = globals().get('_QWEN_PROGRESS_CALLBACK')
    if callback: callback(event)
    print('COMFY_PROGRESS:' + json.dumps(event, ensure_ascii=False), flush=True)


def repository_manifest(repo, session=None, revision=None):
    http = session or requests.Session()
    endpoint = 'https://huggingface.co/api/models/' + repo
    if revision: endpoint += '/revision/' + revision
    response = http.get(endpoint, params={'blobs':'true'}, timeout=(20,60))
    response.raise_for_status()
    data = response.json()
    if data.get('private') or data.get('gated'):
        raise RuntimeError('默认权重必须能够公开下载，当前来源需要额外授权。')
    return data


def safe_relative(name):
    path = PurePosixPath(name)
    if path.is_absolute() or '..' in path.parts or '\\' in name:
        raise ValueError('Invalid model file path')
    return path


def download_file(http, url, destination, expected_size, expected_sha=None, progress=None, verify_progress=None):
    destination = Path(destination); destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + '.partial')
    receipt = destination.with_name(destination.name + '.verified.json')
    if destination.is_file() and destination.stat().st_size == expected_size:
        try:
            stamp = json.loads(receipt.read_text())
            if stamp == {'size':expected_size, 'sha256':expected_sha}:
                if progress: progress(expected_size)
                return
        except (OSError,ValueError): pass
        # Existing files without a verification receipt must still be checked.
        partial.unlink(missing_ok=True); destination.replace(partial)
    for attempt in range(3):
        try:
            offset = partial.stat().st_size if partial.exists() else 0
            if expected_size and offset > expected_size:
                partial.unlink(); offset = 0
            if offset != expected_size or not partial.exists():
                headers = {'Range':f'bytes={offset}-'} if offset else {}
                with http.get(url, headers=headers, stream=True, timeout=(20,60)) as response:
                    response.raise_for_status()
                    if offset and response.status_code == 206:
                        if not response.headers.get('Content-Range','').startswith(f'bytes {offset}-'):
                            raise ValueError('Invalid download resume range')
                    else:
                        offset = 0
                    with partial.open('ab' if offset else 'wb') as output:
                        for chunk in response.iter_content(chunk_size=4*1024*1024):
                            if chunk:
                                output.write(chunk); offset += len(chunk)
                                if progress: progress(offset)
            if expected_size and partial.stat().st_size != expected_size:
                raise OSError('Incomplete model download')
            if expected_sha:
                digest = hashlib.sha256()
                with partial.open('rb') as content:
                    verified = 0
                    for chunk in iter(lambda:content.read(16*1024*1024), b''):
                        digest.update(chunk); verified += len(chunk)
                        if verify_progress: verify_progress(verified)
                if digest.hexdigest() != expected_sha:
                    partial.unlink(); raise ValueError('Model checksum mismatch')
            partial.replace(destination)
            receipt.write_text(json.dumps({'size':expected_size,'sha256':expected_sha}))
            if progress: progress(destination.stat().st_size)
            return
        except (requests.RequestException, OSError, ValueError):
            if attempt == 2: raise
            time.sleep(2*(attempt+1))


def prepare_hf_assets(folder, http=None):
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
    http = http or requests.Session()
    emit_progress('manifest', '正在从 Hugging Face 获取模型文件清单。')
    base = repository_manifest(BASE_REPO, http, BASE_REVISION)
    quant = repository_manifest(GGUF_REPO, http, GGUF_REVISION)
    records = []
    for item in base['siblings']:
        name = item['rfilename']
        if name in {'model_index.json','LICENSE','transformer/config.json'} or any(name.startswith(p+'/') for p in ['processor','scheduler','vae','text_encoder']):
            if name.endswith('.json') or name.endswith('.safetensors') or name.endswith('.txt') or name.endswith('.model') or name.endswith('.jinja') or name.endswith('.tiktoken') or name == 'LICENSE':
                records.append((BASE_REPO,base['sha'],item,name))
    candidates = [f for f in quant['siblings'] if f['rfilename'].lower().endswith('q4_k_m.gguf')]
    if len(candidates) != 1: raise RuntimeError('兼容的 Q4_K_M 权重清单不明确。')
    records.append((GGUF_REPO,quant['sha'],candidates[0],'transformer-q4.gguf'))
    if not any(name.startswith('text_encoder/') and name.endswith('.safetensors') for _,_,_,name in records):
        raise RuntimeError('模型清单缺少文本编码器权重。')
    total = sum(int(item.get('size') or item.get('lfs',{}).get('size') or 0) for _,_,item,_ in records)
    cached=True
    for _,_,item,name in records:
        safe_relative(name)
        size=int(item.get('size') or item.get('lfs',{}).get('size') or 0)
        target=folder/name
        try:
            receipt=json.loads(target.with_name(target.name+'.verified.json').read_text())
            if not target.is_file() or target.stat().st_size!=size or receipt!={'size':size,'sha256':item.get('lfs',{}).get('sha256')}:
                cached=False;break
        except (OSError,ValueError):cached=False;break
    if cached:
        emit_progress('cached','已找到完整模型缓存，无需重新下载。',total,total or None)
        return folder
    done = 0; last = [0.0]
    for repo,revision,item,name in records:
        safe_relative(name)
        size = int(item.get('size') or item.get('lfs',{}).get('size') or 0)
        sha = item.get('lfs',{}).get('sha256')
        start = done
        def progress(value, name=name, start=start):
            now=time.monotonic()
            if now-last[0]>=.8 or value==size:
                emit_progress('download','首次准备：正在下载模型权重。',start+value,total or None,name)
                last[0]=now
        url='https://huggingface.co/'+repo+'/resolve/'+revision+'/'+quote(item['rfilename'],safe='/')
        def verify_progress(value, name=name, size=size):
            now=time.monotonic()
            if now-last[0]>=.8 or value==size:
                emit_progress('verify','正在校验下载的权重。',value,size,name);last[0]=now
        download_file(http,url,folder/name,size,sha,progress,verify_progress)
        done += size
    (folder/'asset-sources.json').write_text(json.dumps({'base_repo':BASE_REPO,'base_revision':base['sha'],'gguf_repo':GGUF_REPO,'gguf_revision':quant['sha'],'download_bytes':total},indent=2))
    emit_progress('download','模型权重下载完成。',done,total or None)
    return folder


def copy_assets(source, target):
    source,target=Path(source),Path(target)
    files=[p for p in source.rglob('*') if p.is_file()]
    total=sum(p.stat().st_size for p in files); done=0; last=0
    for path in files:
        destination=target/path.relative_to(source); destination.parent.mkdir(parents=True,exist_ok=True)
        with path.open('rb') as src, destination.open('wb') as dst:
            for chunk in iter(lambda:src.read(8*1024*1024),b''):
                dst.write(chunk);done+=len(chunk)
                now=time.monotonic()
                if now-last>=.8:
                    emit_progress('copy','正在准备已缓存的模型文件。',done,total,path.name);last=now
    emit_progress('copy','模型文件准备完成。',done,total)
