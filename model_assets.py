"""Public, pinned, resumable model downloads. Executed after asset_manager.py."""

def model_asset_records(model_id, sources, http):
    records=[]
    for index,(repo,revision) in enumerate(sources):
        manifest=repository_manifest(repo,http,revision)
        if manifest.get('sha')!=revision:raise RuntimeError('Model revision mismatch')
        for item in manifest['siblings']:
            name=item['rfilename'];safe_relative(name)
            keep=False
            if model_id=='z-image-turbo':
                keep=(name in ('model_index.json','LICENSE','LICENSE.txt','README.md') or
                      name.split('/')[0] in ('text_encoder','tokenizer','transformer','vae','scheduler'))
                keep=keep and name.endswith(('.json','.safetensors','.txt','.model','.jinja','.tiktoken','.md'))
            elif index==0:
                # Lightning replaces the base UNet; do not download a second one.
                parts=name.split('/')
                keep=name in ('model_index.json','LICENSE.md','README.md','unet/config.json')
                if parts[0] in ('text_encoder','text_encoder_2','tokenizer','tokenizer_2','scheduler','vae'):
                    keep=name.endswith(('.json','.txt','.model','.jinja')) or name.endswith('.fp16.safetensors')
            else:keep=name in ('sdxl_lightning_4step_unet.safetensors','LICENSE.md','README.md')
            if keep:records.append((repo,revision,item,('base/' if index==0 else 'lightning/')+name if model_id=='sdxl-lightning' else name))
    if not any(name.endswith('.safetensors') for _,_,_,name in records):raise RuntimeError('Model weights missing')
    return records


def prepare_model_hf_assets(folder):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    with requests.Session() as http:
        emit_progress('manifest','正在获取所选模型清单。')
        records=model_asset_records(_CC_MODEL_ID,_CC_MODEL_SOURCES,http)
        total=sum(int(item.get('size') or item.get('lfs',{}).get('size') or 0) for _,_,item,_ in records)
        done=0;last=[0.0]
        for repo,revision,item,name in records:
            size=int(item.get('size') or item.get('lfs',{}).get('size') or 0)
            if size<=0:raise RuntimeError('Model file size missing')
            start=done
            def progress(value,name=name,start=start,size=size):
                now=time.monotonic()
                if now-last[0]>.8 or value==size:
                    emit_progress('download','正在下载所选模型。',start+value,total,name);last[0]=now
            url='https://huggingface.co/'+repo+'/resolve/'+revision+'/'+quote(item['rfilename'],safe='/')
            download_file(http,url,folder/name,size,item.get('lfs',{}).get('sha256'),progress)
            done+=size
    manifest={'studio_model_id':_CC_MODEL_ID,'model_sources':_CC_MODEL_SOURCES,'download_bytes':total}
    (folder/'asset-sources.json').write_text(json.dumps(manifest,indent=2))
    emit_progress('download','所选模型下载完成。',total,total)
    return folder


def prepare_selected_model_assets(folder):
    handle,token=globals().get('_QWEN_CACHE_HANDLE'),globals().get('_QWEN_CACHE_TOKEN')
    globals()['_QWEN_CACHE_STATUS']='disabled';globals()['_QWEN_CACHE_STATUS_HANDLE']=handle
    if handle and token:
        try:
            emit_progress('cache-check','正在检查所选模型的私有缓存。')
            metadata=cache_metadata(handle,token)
            if metadata:
                download_cache_assets(folder,handle,token,metadata)
                globals()['_QWEN_CACHE_STATUS']='saved';globals()['_QWEN_RESOLVED_SOURCE']='kaggle-cache'
                return Path(folder)
            globals()['_QWEN_CACHE_STATUS']='missing'
        except Exception:
            globals()['_QWEN_CACHE_STATUS']='unavailable'
            emit_progress('cache-warning','正在从 Hugging Face 准备所选模型。')
    globals()['_QWEN_RESOLVED_SOURCE']='huggingface'
    return prepare_model_hf_assets(folder)
