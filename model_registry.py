"""Client-side model catalog. Importing it never loads GPU libraries."""
DEFAULT_MODEL = 'qwen-image-2.1'
MODELS = {
    DEFAULT_MODEL: {'id':DEFAULT_MODEL,'name':'Qwen-Image 2.1 · Quality','default_steps':20,'fixed_steps':None,
                    'cache_slug':'qwen-image-studio-assets','family':'qwen21'},
    'z-image-turbo': {'id':'z-image-turbo','name':'Z-Image Turbo · Efficient','default_steps':9,'fixed_steps':None,
                      'cache_slug':'cc-media-z-image-turbo','family':'zimage'},
    'sdxl-lightning': {'id':'sdxl-lightning','name':'SDXL-Lightning · 4-step Preview','default_steps':4,'fixed_steps':4,
                       'cache_slug':'cc-media-sdxl-lightning','family':'sdxl'},
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
