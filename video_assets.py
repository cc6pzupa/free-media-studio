"""Pinned, resumable video downloads using the existing checked transfer helper."""
import json
import os
import shutil
import time
from pathlib import Path
from urllib.parse import quote


def video_asset_records(profile,manifest):
    if manifest.get('sha')!=profile['revision']:raise RuntimeError('Video revision mismatch')
    records=[]
    for item in manifest['siblings']:
        name=item['rfilename']
        from asset_manager import safe_relative
        safe_relative(name)
        parts=name.split('/')
        metadata=name in ('model_index.json','README.md') or ('license' in name.lower() and len(parts)==1)
        component=parts[0] in ('text_encoder','tokenizer','transformer','scheduler','vae')
        keep=metadata or (component and name.endswith(('.json','.txt','.model','.safetensors')))
        if profile['family']=='ltx':
            keep=metadata or name==profile['checkpoint'] or (component and
                (not name.endswith('.safetensors') or parts[0]=='text_encoder'))
            if profile.get('config_repo') and (name=='model_index.json' or parts[0] in ('vae','scheduler','transformer')):keep=False
        if keep:records.append(item)
    if not any(i['rfilename'].endswith('.safetensors') for i in records):raise RuntimeError('Missing video weights')
    return records


def prepare_video_assets(profile,folder,progress=None):
    import requests
    from asset_manager import repository_manifest,download_file
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    notify=progress or (lambda event:None)
    notify({'phase':'manifest'})
    with requests.Session() as http:
        records=[(profile['repo'],profile['revision'],item) for item in
            video_asset_records(profile,repository_manifest(profile['repo'],http,profile['revision']))]
        if profile.get('config_repo'):
            manifest=repository_manifest(profile['config_repo'],http,profile['config_revision'])
            if manifest.get('sha')!=profile['config_revision']:raise RuntimeError('Video config revision mismatch')
            names={'model_index.json','vae/config.json','transformer/config.json','scheduler/scheduler_config.json'}
            records.extend((profile['config_repo'],profile['config_revision'],item) for item in manifest['siblings'] if item['rfilename'] in names)
        sizes=[int(i.get('size') or i.get('lfs',{}).get('size') or 0) for _,_,i in records]
        if any(size<=0 for size in sizes):raise RuntimeError('Missing video file size')
        total=sum(sizes);done=0;last=[0.0]
        for (repo,revision,item),size in zip(records,sizes):
            name=item['rfilename'];start=done
            def report(value,start=start,name=name,size=size):
                now=time.monotonic()
                if now-last[0]>=1 or value==size:
                    notify(dict(phase='download',file=name,downloaded_bytes=start+value,
                        total_bytes=total,percent=round(100*(start+value)/total,1)));last[0]=now
            url='https://huggingface.co/'+repo+'/resolve/'+revision+'/'+quote(name,safe='/')
            digest=item.get('lfs',{}).get('sha256');destination=folder/name
            shared=folder.parent/'_shared'/(digest or 'unused')
            if digest and shared.is_file():
                download_file(http,url,shared,size,digest)
                destination.parent.mkdir(parents=True,exist_ok=True)
                if not destination.exists():
                    os.link(shared,destination)
                    receipt=shared.with_name(shared.name+'.verified.json')
                    shutil.copy2(receipt,destination.with_name(destination.name+'.verified.json'))
                download_file(http,url,destination,size,digest,report)
            else:
                download_file(http,url,destination,size,digest,report)
                if digest:
                    shared.parent.mkdir(exist_ok=True)
                    try:os.link(destination,shared)
                    except FileExistsError:pass
                    shutil.copy2(destination.with_name(destination.name+'.verified.json'),
                        shared.with_name(shared.name+'.verified.json'))
            done+=size
    (folder/'asset-sources.json').write_text(json.dumps({'studio_model_id':profile['id'],
        'model_sources':video_sources(profile),'download_bytes':total}),encoding='utf-8')
    return folder


def video_cache_context(profile,handle,token,progress):
    """Use the existing account-owned, create-only Dataset cache protocol."""
    scope={'_CC_MODEL_ID':profile['id'],'_CC_MODEL_SOURCES':video_sources(profile),
        '_QWEN_CACHE_HANDLE':handle,'_QWEN_CACHE_TOKEN':token,'_QWEN_PROGRESS_CALLBACK':progress}
    for name in ('asset_manager.py','cache_manager.py'):
        exec(compile(Path(__file__).with_name(name).read_text(encoding='utf-8'),'<video-cache>','exec'),scope)
    return scope


def video_sources(profile):
    sources=[[profile['repo'],profile['revision']]]
    if profile.get('config_repo'):sources.append([profile['config_repo'],profile['config_revision']])
    return sources
