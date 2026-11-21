"""Video-only catalog; ordinary image clients do not import GPU packages."""
VIDEO_MODELS = {
    'ltx-video-2b-distilled': {
        'id':'ltx-video-2b-distilled','name':'LTX-Video 2B · Efficient','tier':'efficient',
        'family':'ltx','default_steps':8,'fixed_steps':8,'width':768,'height':448,
        'frames':33,'fps':16,'guidance':1.0,'cache_slug':'cc-media-ltx-2b-distilled','image_to_video':True,
        'first_use':{'minutes':'7–11','download_gb':27},
        # Landscape (default), portrait and square; none exceeds the default pixel count.
        'sizes':['768x448','448x768','512x512'],'last_frame':True,
        'repo':'Lightricks/LTX-Video','revision':'8984fa25007f376c1a299016d0957a37a2f797bb',
        'checkpoint':'ltxv-2b-0.9.6-distilled-04-25.safetensors',
        'sigmas':[1.0,.9937,.9875,.9812,.975,.9094,.725,.4219],
        'config_repo':'Lightricks/LTX-Video-0.9.5','config_revision':'e58e28c39631af4d1468ee57a853764e11c1d37e'},
    'wan21-t2v-1.3b': {
        'id':'wan21-t2v-1.3b','name':'Wan2.1 1.3B · Balanced','tier':'balanced',
        'family':'wan','default_steps':30,'fixed_steps':None,'width':832,'height':480,
        'frames':33,'fps':16,'guidance':5.0,'cache_slug':'cc-media-wan21-1-3b','image_to_video':False,
        'first_use':{'minutes':'12–14','download_gb':None},'sizes':['832x480','480x832','624x624'],'last_frame':False,
        'repo':'Wan-AI/Wan2.1-T2V-1.3B-Diffusers','revision':'0fad780a534b6463e45facd96134c9f345acfa5b'},
    'wan22-ti2v-5b': {
        'id':'wan22-ti2v-5b','name':'Wan2.2 5B · Quality','tier':'quality',
        'family':'wan22','default_steps':30,'fixed_steps':None,'width':768,'height':448,
        'frames':33,'fps':24,'guidance':5.0,'cache_slug':'cc-media-wan22-5b','image_to_video':True,
        'first_use':{'minutes':'19','download_gb':32},'sizes':['768x448','448x768','576x576'],'last_frame':False,
        'repo':'Wan-AI/Wan2.2-TI2V-5B-Diffusers','revision':'b8fff7315c768468a5333511427288870b2e9635'},
}


def video_profile(model_id='wan21-t2v-1.3b'):
    if not isinstance(model_id,str) or model_id not in VIDEO_MODELS:
        raise ValueError('未知视频模型 / Unknown video model.')
    return dict(VIDEO_MODELS[model_id])


def video_catalog():
    return [{k:v for k,v in p.items() if k not in ('repo','revision','checkpoint','config_repo','config_revision')}
            for p in VIDEO_MODELS.values()]


def video_size(profile,size=None):
    """(width, height) for an allowed size string; the first size is the default."""
    size=profile['sizes'][0] if size in (None,'') else size
    if size not in profile['sizes']:raise ValueError(profile['name']+' 支持的尺寸：'+'、'.join(profile['sizes'])+'。')
    width,height=(int(value) for value in size.split('x'))
    return width,height


def validate_video(prompt,seed=42,steps=None,model_id='wan21-t2v-1.3b',frames=None,image=None,negative_prompt=None,size=None,last_image=None):
    """Validate a video request. `image` is passed through unchanged: each layer
    carries its own representation (base64, stored digest, bytes)."""
    p=video_profile(model_id)
    if not isinstance(prompt,str) or not 1<=len(prompt.strip())<=4000:
        raise ValueError('提示词需要 1–4000 个字符。')
    if type(seed) is not int or not 0<=seed<=2147483647:raise ValueError('Invalid seed')
    steps=p['default_steps'] if steps is None else steps
    frames=p['frames'] if frames is None else frames
    if type(steps) is not int or not 1<=steps<=50:raise ValueError('steps must be 1–50')
    if p['fixed_steps'] and steps!=p['fixed_steps']:raise ValueError('LTX distilled requires 8 steps')
    if type(frames) is not int or not 9<=frames<=81 or (frames-1)%8:
        raise ValueError('frames must be 9, 17, …, 81')
    result={'prompt':prompt,'seed':seed,'steps':steps,'model_id':p['id'],'frames':frames}
    if negative_prompt is not None and negative_prompt!='':
        if not isinstance(negative_prompt,str) or len(negative_prompt)>2000:raise ValueError('负面提示词最多 2000 字。')
        if p['guidance']<=1:raise ValueError(p['name']+' 不使用负面提示词（无 CFG）。')
        if negative_prompt.strip():result['negative_prompt']=negative_prompt
    if image is not None and image!='':
        if not p['image_to_video']:raise ValueError(p['name']+' 不支持图生视频；请选择 LTX-Video 2B 或 Wan2.2 5B。')
        result['image']=image
    if last_image is not None and last_image!='':
        if not p['last_frame']:raise ValueError(p['name']+' 不支持尾帧；首尾帧请选择 LTX-Video 2B。')
        if 'image' not in result:raise ValueError('使用尾帧时需要同时提供首帧图片。')
        result['last_image']=last_image
    video_size(p,size)
    if size not in (None,'',p['sizes'][0]):result['size']=size
    return result
