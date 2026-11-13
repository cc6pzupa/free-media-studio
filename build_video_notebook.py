"""Build the standalone video notebook from the shared video runtime sources."""
import json
from pathlib import Path
from video_session import VIDEO_FILES

ROOT=Path(__file__).resolve().parent


def build():
    sources={name:(ROOT/name).read_text(encoding='utf-8') for name in VIDEO_FILES}
    setup=f'''import sys,json
from pathlib import Path
modules=Path('/kaggle/temp/cc-video-modules');modules.mkdir(exist_ok=True)
for name,source in {sources!r}.items():(modules/name).write_text(source,encoding='utf-8')
sys.path.insert(0,str(modules))
print('CC Media Studio video ready; first generation installs an isolated environment')
'''
    ui='''import ipywidgets as widgets
from IPython.display import Video,display,clear_output
from video_registry import VIDEO_MODELS
from video_process import VideoProcess
model=widgets.Dropdown(options=[(p['name'],p['id']) for p in VIDEO_MODELS.values()],value='wan21-t2v-1.3b',description='Model')
prompt=widgets.Textarea(value='An adult anime catgirl reading by a sunlit window, hair moving gently in the breeze. Slow camera movement.',description='Prompt',layout=widgets.Layout(width='95%',height='100px'))
frames=widgets.Dropdown(options=[9,17,33,49,65,81],value=33,description='Frames')
steps=widgets.IntSlider(value=30,min=1,max=50,description='Steps')
seed=widgets.IntText(value=42,description='Seed')
acceleration=widgets.Dropdown(options=[('Auto / 双卡优先','auto'),('Encoder split / 编码分卡','split')],value='auto',description='GPU')
negative=widgets.Textarea(value='',description='Negative',placeholder='Wan only / 仅 Wan',layout=widgets.Layout(width='95%',height='60px'))
first_frame=widgets.FileUpload(accept='image/png,image/jpeg,image/webp',multiple=False,description='First frame / 首帧')
frame_hint=widgets.HTML()
button=widgets.Button(description='Generate / 生成',button_style='primary')
output=widgets.Output();runtime=None
def selected(change=None):
    p=VIDEO_MODELS[model.value];steps.value=p['default_steps'];steps.disabled=bool(p['fixed_steps'])
    negative.layout.display='' if p['guidance']>1 else 'none'
    first_frame.layout.display='' if p['image_to_video'] else 'none'
    frame_hint.value='<span style="color:#777">'+('Optional first frame, center-cropped to %d × %d / 可选首帧，居中裁剪' % (p['width'],p['height']) if p['image_to_video'] else 'Text-to-video only / 仅文生视频')+'</span>'
model.observe(selected,names='value');selected()
def frame_data():
    value=first_frame.value
    if not value or not VIDEO_MODELS[model.value]['image_to_video']:return None
    import base64
    item=value[0] if isinstance(value,(tuple,list)) else next(iter(value.values()))
    return base64.b64encode(bytes(item['content'])).decode('ascii')
def generate(_):
    global runtime
    button.disabled=True
    try:
        with output:
            clear_output(wait=True)
            if runtime is not None and runtime.profile['id']!=model.value:runtime.close();runtime=None
            if runtime is None:runtime=VideoProcess(model.value,acceleration=acceleration.value,progress=lambda event:print(event,flush=True))
            runtime.acceleration=acceleration.value
            path,usage=runtime.generate(prompt.value,seed=seed.value,steps=steps.value,frames=frames.value,output='/kaggle/working/cc-media-video.mp4',
                image=frame_data(),negative_prompt=(negative.value or None) if VIDEO_MODELS[model.value]['guidance']>1 else None)
            display(Video(str(path),embed=True));print(usage)
    finally:button.disabled=False
button.on_click(generate)
display(widgets.VBox([model,prompt,negative,first_frame,frame_hint,widgets.HBox([frames,steps]),widgets.HBox([seed,acceleration]),button,output]))
'''
    def code(text):return {'cell_type':'code','metadata':{},'execution_count':None,'outputs':[],'source':text.splitlines(keepends=True)}
    notebook={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},'cells':[
        {'cell_type':'markdown','metadata':{},'source':['# CC Media Studio · Video Preview\n','选择 GPU T4 ×2，开启 Internet，然后运行全部单元格。 / Select GPU T4 ×2, enable Internet, and run all cells.\n','首次自动下载模型；后续请求复用当前模型。MP4 保存到 `/kaggle/working/cc-media-video.mp4`。 / First use downloads weights; subsequent requests reuse the active model.\n','结束后在 Kaggle 停止 GPU 会话。 / Stop the Kaggle GPU session when finished.']},code(setup),code(ui)]}
    for i,cell in enumerate(notebook['cells']):cell['id']='cc-video-'+str(i)
    (ROOT/'kaggle_video.ipynb').write_text(json.dumps(notebook,ensure_ascii=False,indent=1),encoding='utf-8')


if __name__=='__main__':build()
