"""Pinned, resumable video downloads using the existing checked transfer helper."""
import json
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
        records=video_asset_records(profile,repository_manifest(profile['repo'],http,profile['revision']))
        sizes=[int(i.get('size') or i.get('lfs',{}).get('size') or 0) for i in records]
        if any(size<=0 for size in sizes):raise RuntimeError('Missing video file size')
        total=sum(sizes);done=0;last=[0.0]
        for item,size in zip(records,sizes):
            name=item['rfilename'];start=done
            def report(value,start=start,name=name,size=size):
                now=time.monotonic()
                if now-last[0]>=1 or value==size:
                    notify(dict(phase='download',file=name,downloaded_bytes=start+value,
                        total_bytes=total,percent=round(100*(start+value)/total,1)));last[0]=now
            url='https://huggingface.co/'+profile['repo']+'/resolve/'+profile['revision']+'/'+quote(name,safe='/')
            download_file(http,url,folder/name,size,item.get('lfs',{}).get('sha256'),report)
            done+=size
    (folder/'asset-sources.json').write_text(json.dumps({'studio_model_id':profile['id'],
        'model_sources':[[profile['repo'],profile['revision']]],'download_bytes':total}),encoding='utf-8')
    return folder
