"""Build the self-contained Kaggle entry point from the shared runtime."""
import inspect
import json
from pathlib import Path
from session_backend import bootstrap_code, generation_code


def build():
    bootstrap=bootstrap_code('qwen-studio/huggingface',asset_mode='auto',install=False)
    ui='''import base64, contextlib, html, io, json, time
import ipywidgets as widgets
from IPython.display import display
from PIL import Image

_BOOTSTRAP = '''+repr(bootstrap)+'\n'+inspect.getsource(generation_code)+'''
prompt = widgets.Textarea(value='A small orange fox beside a ceramic teacup, soft morning window light, natural photography', description='提示词', layout=widgets.Layout(width='95%', height='110px'))
steps = widgets.IntSlider(value=20,min=1,max=50,description='步数')
seed = widgets.IntText(value=42,description='种子')
button = widgets.Button(description='自动准备并生成',button_style='primary',icon='magic')
progress = widgets.IntProgress(value=0,min=0,max=100,description='准备',bar_style='info',layout=widgets.Layout(width='95%'))
status = widgets.HTML('<span style="color:#777">第一次将从 Hugging Face 自动下载权重，无需 Dataset 或额外 Token。</span>')
image_view = widgets.Image(format='png',layout=widgets.Layout(max_width='512px'))
info = widgets.HTML()
cache_user = widgets.Text(description='Kaggle 账号',placeholder='可选：填写后启用私有 Dataset 缓存')
cache_token = widgets.Password(description='API Key',placeholder='可选：仅当前内核使用')
cache_box = widgets.Accordion(children=[widgets.VBox([cache_user,cache_token])])
cache_box.set_title(0,'跨会话缓存（可选，账号 + API Key）')
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
        status.value='请填写提示词。'
        return
    button.disabled=True
    started=time.perf_counter()
    try:
        globals()['_QWEN_PROGRESS_CALLBACK']=on_progress
        capture=io.StringIO()
        if cache_user.value.strip() and cache_token.value.strip():
            import re
            if not re.fullmatch(r'[A-Za-z0-9_-]+',cache_user.value.strip()):raise ValueError('Kaggle 用户名格式不正确')
            globals()['_QWEN_CACHE_HANDLE']=cache_user.value.strip()+'/qwen-image-studio-assets'
            globals()['_QWEN_CACHE_TOKEN']=cache_token.value.strip()
        try:
            with contextlib.redirect_stdout(capture):exec(_BOOTSTRAP,globals())
        finally:
            globals().pop('_QWEN_CACHE_TOKEN',None)
            globals().pop('_QWEN_CACHE_HANDLE',None)
            cache_token.value=''
        status.value='模型已就绪，正在生成…'
        progress.value=0
        capture=io.StringIO()
        with contextlib.redirect_stdout(capture):exec(generation_code(prompt.value,seed.value,steps.value),globals())
        line=next(line for line in capture.getvalue().splitlines() if line.startswith('COMFY_RESULT:'))
        result=json.loads(line[len('COMFY_RESULT:'):])
        raw=base64.b64decode(result['png'])
        Image.open(io.BytesIO(raw)).verify()
        image_view.value=raw
        progress.value=100
        stats=result['stats']
        status.value='生成完成 · '+('已复用模型' if stats['model_reused'] else '首次准备完成，后续复用')
        info.value='本次 %.1f 秒 · 模型加载 %d 次 · PNG 512 × 512' % (time.perf_counter()-started,stats['model_load_count'])
        button.description='生成图片'
    except Exception as error:
        status.value='<span style="color:#b94747">'+html.escape(str(error))+'</span>'
    finally:button.disabled=False

button.on_click(run)
display(widgets.VBox([widgets.HTML('<h2 style="color:#6258ee">Qwen Studio · Kaggle</h2>'),prompt,steps,seed,cache_box,button,progress,status,image_view,info]))
'''
    def cell(kind,source,name):
        data={'id':name,'cell_type':kind,'metadata':{},'source':source.splitlines(keepends=True)}
        if kind=='code':data.update(execution_count=None,outputs=[])
        return data
    nb={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'}},'cells':[
        cell('markdown','# Qwen Studio — 从零自动准备\n\n选择 **GPU T4 ×2** 并开启 Internet，然后运行所有单元格。\n\n不需要已有 Dataset、模型文件或 Hugging Face Token。填提示词并点击按钮，首次自动下载公开权重，显示真实下载进度；之后复用已加载的模型。\n\n如需跨会话缓存，展开可选设置，填写自己的 Kaggle 用户名与 API Key。首次自动创建私有 `用户名/qwen-image-studio-assets`，后续优先读取。留空仍可生图，权重仅保存在当前会话临时磁盘。不要保存带凭据的单元格输出。','intro'),
        cell('code','%pip install -q bitsandbytes==0.50.2 gguf requests ipywidgets','dependencies'),
        cell('code',ui,'studio-ui')]}
    path=Path(__file__).with_name('kaggle_standalone.ipynb')
    path.write_text(json.dumps(nb,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
    return path

if __name__=='__main__':print(build())
