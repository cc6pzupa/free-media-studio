"""Build the self-contained Kaggle entry point from the shared runtime."""
import inspect
import json
from pathlib import Path
from session_backend import bootstrap_code, generation_code, RELEASE_FAILURE_CODE, FREE_GPU_CODE
from model_registry import MODELS


def build():
    bootstraps={key:bootstrap_code('qwen-studio/huggingface',asset_mode='auto',install=False,model_id=key) for key in MODELS}
    ui='''import base64, contextlib, html, io, json, time
import ipywidgets as widgets
from IPython.display import display
from PIL import Image

'''+Path(__file__).with_name('model_registry.py').read_text(encoding='utf-8')+'''
_BOOTSTRAPS = '''+repr(bootstraps)+'\nRELEASE_FAILURE_CODE = '+repr(RELEASE_FAILURE_CODE)+'\nFREE_GPU_CODE = '+repr(FREE_GPU_CODE)+'\n'+inspect.getsource(generation_code)+'''
model = widgets.Dropdown(options=[(p['name'],key) for key,p in MODELS.items()],value=DEFAULT_MODEL,description='Model / 模型',layout=widgets.Layout(width='95%'))
prompt = widgets.Textarea(value='A small orange fox beside a ceramic teacup, soft morning window light, natural photography', description='Prompt / 提示词', layout=widgets.Layout(width='95%', height='110px'))
steps = widgets.IntSlider(value=20,min=1,max=50,description='Steps / 步数')
size = widgets.Dropdown(options=list(SIZES),value=DEFAULT_SIZE,description='Size / 尺寸')
negative = widgets.Textarea(value='',description='Negative / 负面',placeholder='Qwen only / 仅 Qwen：不希望出现的内容',layout=widgets.Layout(width='95%',height='60px'))
task = widgets.Dropdown(options=[('Text to image / 文生图','text'),('Image to image / 图生图','img2img'),('Inpaint / 局部编辑','inpaint')],value='text',description='Mode / 方式')
strength = widgets.FloatSlider(value=0.6,min=0.05,max=1.0,step=0.05,description='Strength / 强度')
source_upload = widgets.FileUpload(accept='image/png,image/jpeg,image/webp',multiple=False,description='Image / 参考图')
mask_upload = widgets.FileUpload(accept='image/png,image/jpeg,image/webp',multiple=False,description='Mask / 蒙版')
edit_hint = widgets.HTML('<span style="color:#777">Z-Image Turbo / SDXL-Lightning。蒙版白色为重绘区域；输入按所选尺寸居中裁剪。 / White mask areas are repainted; inputs are center-cropped.</span>')
# Kaggle has no drawing widget without extra extensions: build an ellipse or
# rectangle mask from sliders on top of the uploaded image, or upload a mask.
mask_mode = widgets.ToggleButtons(options=[('Shape / 形状','shape'),('Upload / 上传','upload')],value='shape',description='Mask / 蒙版')
mask_shape = widgets.Dropdown(options=[('Ellipse / 椭圆','ellipse'),('Rectangle / 矩形','rectangle')],value='ellipse',description='Shape / 形状')
mask_x = widgets.IntSlider(value=50,min=0,max=100,description='X %')
mask_y = widgets.IntSlider(value=50,min=0,max=100,description='Y %')
mask_w = widgets.IntSlider(value=40,min=5,max=100,description='Width / 宽 %')
mask_h = widgets.IntSlider(value=40,min=5,max=100,description='Height / 高 %')
feather = widgets.IntSlider(value=8,min=0,max=64,description='Feather / 羽化')
mask_preview = widgets.Image(format='png',layout=widgets.Layout(max_width='360px'))
shape_box = widgets.VBox([mask_shape,mask_x,mask_y,mask_w,mask_h,mask_preview])
def shape_mask(source):
    """White shape on black at the source image size (white is repainted)."""
    from PIL import Image as PILImage, ImageDraw
    width,height = PILImage.open(io.BytesIO(source)).size
    w,h = width*mask_w.value/100, height*mask_h.value/100
    x,y = width*mask_x.value/100, height*mask_y.value/100
    box = (x-w/2, y-h/2, x+w/2, y+h/2)
    mask = PILImage.new('L',(width,height),0)
    (ImageDraw.Draw(mask).ellipse if mask_shape.value=='ellipse' else ImageDraw.Draw(mask).rectangle)(box,fill=255)
    buffer = io.BytesIO(); mask.save(buffer,format='PNG'); return buffer.getvalue()
def refresh_preview(change=None):
    source = uploaded(source_upload)
    if task.value!='inpaint' or mask_mode.value!='shape' or source is None:
        mask_preview.value=b'';return
    from PIL import Image as PILImage
    picture = PILImage.open(io.BytesIO(source)).convert('RGB')
    mask = PILImage.open(io.BytesIO(shape_mask(source)))
    tint = PILImage.new('RGB',picture.size,(255,64,64))
    preview = PILImage.composite(PILImage.blend(picture,tint,0.55),picture,mask)
    preview.thumbnail((360,360)); buffer = io.BytesIO(); preview.save(buffer,format='PNG'); mask_preview.value = buffer.getvalue()
edit_box = widgets.VBox([source_upload,mask_mode,shape_box,mask_upload,strength,feather,edit_hint])
def uploaded(widget):
    value=widget.value
    if not value:return None
    item=value[0] if isinstance(value,(tuple,list)) else next(iter(value.values()))
    return bytes(item['content'])
def refresh_controls(change=None):
    profile=model_profile(model.value)
    negative.layout.display='' if profile['negative_prompt'] else 'none'
    edit_box.layout.display='none' if task.value=='text' else ''
    inpaint = task.value=='inpaint'
    for widget in (mask_mode,feather):widget.layout.display='' if inpaint else 'none'
    shape_box.layout.display='' if inpaint and mask_mode.value=='shape' else 'none'
    mask_upload.layout.display='' if inpaint and mask_mode.value=='upload' else 'none'
    refresh_preview()
def model_changed(change):
    profile=model_profile(model.value);steps.value=profile['default_steps'];steps.disabled=profile['fixed_steps'] is not None
    if task.value not in profile['tasks']:task.value='text'
    refresh_controls()
def task_changed(change):
    if task.value!='text' and task.value not in model_profile(model.value)['tasks']:model.value='z-image-turbo'
    strength.value=0.9 if task.value=='inpaint' else 0.6
    refresh_controls()
model.observe(model_changed,names='value')
task.observe(task_changed,names='value')
mask_mode.observe(refresh_controls,names='value')
for _widget in (mask_shape,mask_x,mask_y,mask_w,mask_h,source_upload):_widget.observe(refresh_preview,names='value')
refresh_controls()
acceleration = widgets.Dropdown(options=[('Auto · 双卡优先','auto'),('Encoder/model split · 原分卡','split')],value='auto',description='Acceleration / 加速',layout=widgets.Layout(width='95%'))
seed = widgets.IntText(value=42,description='Seed / 种子')
button = widgets.Button(description='Generate / 生成',button_style='primary',icon='magic')
progress = widgets.IntProgress(value=0,min=0,max=100,description='Setup / 准备',bar_style='info',layout=widgets.Layout(width='95%'))
status = widgets.HTML('<span style="color:#777">第一次将从 Hugging Face 自动下载权重，无需 Dataset 或额外 Token。</span>')
image_view = widgets.Image(format='png',layout=widgets.Layout(max_width='640px'))
info = widgets.HTML()
cache_user = widgets.Text(description='Kaggle user / 账号',placeholder='可选：填写后启用私有 Dataset 缓存')
cache_token = widgets.Password(description='API Key',placeholder='可选：仅当前内核使用')
cache_box = widgets.Accordion(children=[widgets.VBox([cache_user,cache_token])])
cache_box.set_title(0,'Private cache / 私有缓存（可选 optional）')
cache_box.selected_index=None

def on_progress(event):
    message=html.escape(event.get('message',''))
    if event.get('total_bytes'):
        percent=event.get('percent',0)
        progress.value=int(percent)
        message += ' · %.1f%% · %.2f / %.2f GB' % (percent,event['downloaded_bytes']/1e9,event['total_bytes']/1e9)
    else:progress.value=0
    status.value='<span style="color:#777">'+message+'</span>'

def run(_):
    if not prompt.value.strip():
        status.value='请填写提示词。 / Enter a prompt.'
        return
    button.disabled=True;model.disabled=True;acceleration.disabled=True;task.disabled=True
    started=time.perf_counter()
    try:
        image=uploaded(source_upload) if task.value!='text' else None
        mask=None
        if task.value=='inpaint':
            mask=shape_mask(image) if mask_mode.value=='shape' and image is not None else uploaded(mask_upload)
        # The negative field is hidden for models without CFG; ignore any leftover text.
        negative_prompt=(negative.value or None) if model_profile(model.value)['negative_prompt'] else None
        options=image_options(model.value,steps.value,size.value,negative_prompt,task.value,
            strength.value if task.value!='text' else None,image is not None,mask is not None,
            feather.value if task.value=='inpaint' else None)
        globals()['_QWEN_PROGRESS_CALLBACK']=on_progress
        globals()['_CC_ACCELERATION']=acceleration.value
        capture=io.StringIO()
        if cache_user.value.strip() and cache_token.value.strip():
            import re
            if not re.fullmatch(r'[A-Za-z0-9_-]+',cache_user.value.strip()):raise ValueError('Kaggle 用户名格式不正确')
            globals()['_QWEN_CACHE_HANDLE']=cache_user.value.strip()+'/'+model_profile(model.value)['cache_slug']
            globals()['_QWEN_CACHE_TOKEN']=cache_token.value.strip()
        try:
            with contextlib.redirect_stdout(capture):exec(_BOOTSTRAPS[model.value],globals())
        finally:
            globals().pop('_QWEN_CACHE_TOKEN',None)
            globals().pop('_QWEN_CACHE_HANDLE',None)
            cache_token.value=''
        globals()['_CC_ACCELERATION']=acceleration.value
        status.value='模型已就绪，正在生成… / Model ready, generating…'
        progress.value=0
        capture=io.StringIO()
        width,height=parse_size(size.value)
        with contextlib.redirect_stdout(capture):exec(generation_code(prompt.value,seed.value,steps.value,model.value,width,height,
            options.get('negative_prompt'),options.get('task','text'),options.get('strength'),image,mask,options.get('feather')),globals())
        line=next(line for line in capture.getvalue().splitlines() if line.startswith('COMFY_RESULT:'))
        result=json.loads(line[len('COMFY_RESULT:'):])
        raw=base64.b64decode(result['png'])
        Image.open(io.BytesIO(raw)).verify()
        image_view.value=raw
        progress.value=100
        stats=result['stats']
        status.value='生成完成 · '+('已复用模型' if stats['model_reused'] else '首次准备完成，后续复用')
        info.value='本次 %.1f 秒 · 模型加载 %d 次 · PNG %d × %d' % (time.perf_counter()-started,stats['model_load_count'],width,height)+' · '+html.escape(stats.get('parallel_strategy',''))+(' · '+task.value if task.value!='text' else '')
        button.description='Generate / 生成'
    except Exception as error:
        status.value='<span style="color:#b94747">'+html.escape(str(error))+'</span>'
    finally:button.disabled=False;model.disabled=False;acceleration.disabled=False;task.disabled=False

button.on_click(run)
display(widgets.VBox([widgets.HTML('<h2 style="color:#6258ee">CC Media Studio · Kaggle</h2>'),model,task,prompt,negative,edit_box,size,steps,seed,acceleration,cache_box,button,progress,status,image_view,info]))
'''
    def cell(kind,source,name):
        data={'id':name,'cell_type':kind,'metadata':{},'source':source.splitlines(keepends=True)}
        if kind=='code':data.update(execution_count=None,outputs=[])
        return data
    nb={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'}},'cells':[
        cell('markdown','# CC Media Studio — 从零自动准备\n\n选择 **GPU T4 ×2** 并开启 Internet，然后运行所有单元格。 / Select **GPU T4 ×2**, enable Internet, and run all cells.\n\n不需要已有 Dataset、模型文件或 Hugging Face Token。填提示词并点击按钮，首次自动下载公开权重，显示真实下载进度；之后复用已加载的模型。\n\n如需跨会话缓存，展开可选设置，填写自己的 Kaggle 用户名与 API Key。首次自动创建私有 `用户名/qwen-image-studio-assets`，后续优先读取。留空仍可生图，权重仅保存在当前会话临时磁盘。不要保存带凭据的单元格输出。','intro'),
        cell('code','%pip install -q bitsandbytes==0.50.2 gguf requests ipywidgets \'diffusers>=0.36,<0.43\' \'transformers>=4.51,<6\' \'accelerate>=1.2\'','dependencies'),
        cell('code',ui,'studio-ui')]}
    path=Path(__file__).with_name('kaggle_standalone.ipynb')
    path.write_text(json.dumps(nb,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
    legacy={'nbformat':4,'nbformat_minor':5,'metadata':nb['metadata'],'cells':[
        cell('markdown','# CC Media Studio · Qwen on dual T4\n\nAutomatic dual-GPU CFG with persistent model reuse.','intro'),
        cell('code',"%pip install -q bitsandbytes==0.50.2 gguf",'dependencies'),
        cell('code',bootstrap_code('faberbruntch/qwen-image-t4-assets',install=False),'runtime'),
        cell('code',"import json\nfrom pathlib import Path\nPROMPT = 'An adult orange-haired catgirl beside a sunlit window, detailed anime illustration'\nSEED = 42\nWIDTH = HEIGHT = 512\nSTEPS = 20\nimage, stats = generate_image(PROMPT,seed=SEED,width=WIDTH,height=HEIGHT,steps=STEPS)\nimage.save('/kaggle/working/result.png')\nPath('/kaggle/working/benchmark.json').write_text(json.dumps(stats,indent=2))\nprint(stats)\ndisplay(image)\n",'generate')]}
    Path(__file__).with_name('template.ipynb').write_text(json.dumps(legacy,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
    return path

if __name__=='__main__':print(build())
