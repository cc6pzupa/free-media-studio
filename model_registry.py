"""Client-side model catalog. Importing it never loads GPU libraries."""
DEFAULT_MODEL = 'qwen-image-2.1'
MODELS = {
    DEFAULT_MODEL: {'id':DEFAULT_MODEL,'name':'Qwen-Image 2.1 · Quality','default_steps':20,'fixed_steps':None,
                    'cache_slug':'qwen-image-studio-assets','family':'qwen21',
                    'tasks':['text'],'negative_prompt':True},
    'z-image-turbo': {'id':'z-image-turbo','name':'Z-Image Turbo · Efficient','default_steps':9,'fixed_steps':None,
                      'cache_slug':'cc-media-z-image-turbo','family':'zimage',
                      'tasks':['text','img2img','inpaint'],'negative_prompt':False},
    'sdxl-lightning': {'id':'sdxl-lightning','name':'SDXL-Lightning · 4-step Preview','default_steps':4,'fixed_steps':4,
                       'cache_slug':'cc-media-sdxl-lightning','family':'sdxl',
                       'tasks':['text','img2img','inpaint'],'negative_prompt':False},
}
MODEL_SOURCES = {
    'z-image-turbo': [('Tongyi-MAI/Z-Image-Turbo','f332072aa78be7aecdf3ee76d5c247082da564a6')],
    'sdxl-lightning': [('stabilityai/stable-diffusion-xl-base-1.0','462165984030d82259a11f4367a4eed129e94a7b'),
                       ('ByteDance/SDXL-Lightning','c9a24f48e1c025556787b0c58dd67a091ece2e44')],
}


def model_profile(model_id=None):
    model_id = DEFAULT_MODEL if model_id is None else model_id
    if not isinstance(model_id,str) or model_id not in MODELS:
        raise ValueError('未知模型 / Unknown model.')
    return dict(MODELS[model_id])


def model_steps(model_id, steps=None):
    profile=model_profile(model_id)
    steps=profile['default_steps'] if steps is None else steps
    if type(steps) is not int or not 1<=steps<=50:raise ValueError('steps 需要为 1–50 的整数。')
    if profile['fixed_steps'] is not None and steps!=profile['fixed_steps']:
        raise ValueError('SDXL-Lightning 固定使用 4 步 / requires 4 steps.')
    return steps


def model_catalog():return [dict(value) for value in MODELS.values()]


# Width x height. Every value is a multiple of 64, valid for all three models.
SIZES=('512x512','768x768','1024x1024','768x512','512x768','1024x576','576x1024')
DEFAULT_SIZE='512x512'
TASKS=('text','img2img','inpaint')
DEFAULT_STRENGTH={'img2img':0.6,'inpaint':0.9}


def parse_size(size=None):
    size=DEFAULT_SIZE if size is None else size
    if not isinstance(size,str) or size not in SIZES:
        raise ValueError('尺寸请选择：'+'、'.join(SIZES)+'。')
    width,height=(int(v) for v in size.split('x'))
    return width,height


def image_options(model_id,steps,size=None,negative_prompt=None,task=None,strength=None,image=False,mask=False,feather=None):
    """Validate optional editing/size parameters; return only non-default values.

    Defaults are omitted so earlier text-to-image request IDs keep their
    stored parameters and remain deduplicated after an update.
    """
    profile=model_profile(model_id);task='text' if task is None else task
    if task not in TASKS:raise ValueError('任务类型请选择 text、img2img 或 inpaint。')
    if task not in profile['tasks']:
        raise ValueError(profile['name']+' 暂不支持图生图或局部编辑；请选择 Z-Image Turbo 或 SDXL-Lightning。')
    width,height=parse_size(size);result={}
    if (width,height)!=(512,512):result.update(width=width,height=height)
    if negative_prompt is not None and negative_prompt!='':
        if not isinstance(negative_prompt,str) or len(negative_prompt)>2000:raise ValueError('负面提示词最多 2000 字。')
        if not profile['negative_prompt']:raise ValueError(profile['name']+' 不使用负面提示词（无 CFG）；请清空该字段。')
        if negative_prompt.strip():result['negative_prompt']=negative_prompt
    if feather not in (None,0) and task!='inpaint':raise ValueError('边缘羽化只用于局部编辑（inpaint）。')
    if task=='text':
        if image or mask or strength is not None:raise ValueError('文生图不接受参考图、蒙版或重绘强度；请选择 img2img 或 inpaint。')
        return result
    if not image:raise ValueError('图生图与局部编辑需要上传参考图。')
    if task=='inpaint' and not mask:raise ValueError('局部编辑需要蒙版（白色区域重绘）。')
    if task=='img2img' and mask:raise ValueError('图生图不使用蒙版；局部编辑请选择 inpaint。')
    strength=DEFAULT_STRENGTH[task] if strength is None else strength
    if isinstance(strength,bool) or not isinstance(strength,(int,float)) or not 0.05<=strength<=1:
        raise ValueError('重绘强度需要在 0.05–1 之间。')
    strength=round(float(strength),3)
    if int(steps*strength)<1:raise ValueError('步数 × 重绘强度至少为 1；请提高重绘强度。')
    result.update(task=task,strength=strength)
    if feather not in (None,0):
        if isinstance(feather,bool) or not isinstance(feather,int) or not 0<=feather<=64:raise ValueError('边缘羽化需要为 0–64 像素的整数。')
        result['feather']=feather
    return result
