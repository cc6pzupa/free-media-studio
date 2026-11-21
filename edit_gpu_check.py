"""Run the 0.10 preview features on a real Kaggle dual-T4 session.

Uses the same AccountBackend path as the GUI: one managed Kaggle session,
model switches, uploads, size checks and image-to-video. Each case is
independent; a failure is recorded and the next case runs. The session is
released at the end.

    KAGGLE_USERNAME=... KAGGLE_API_TOKEN=... python edit_gpu_check.py --output results
"""
import argparse
import io
import json
import os
import struct
import time
import traceback
from pathlib import Path

PROMPT='An adult orange-haired catgirl by a sunlit window, soft anime illustration, detailed'


def image_size(raw):return struct.unpack('>II',raw[16:24])


def mask_png(width,height):
    """White ellipse in the centre (repaint), black elsewhere."""
    from PIL import Image,ImageDraw
    mask=Image.new('L',(width,height),0)
    ImageDraw.Draw(mask).ellipse((width*.3,height*.15,width*.7,height*.6),fill=255)
    buffer=io.BytesIO();mask.save(buffer,format='PNG');return buffer.getvalue()


CASES=[
    # (name, kind, arguments); 'source' is replaced with the first text result.
    ('zimage-text-512','image',dict(model_id='z-image-turbo')),
    ('zimage-img2img-512','image',dict(model_id='z-image-turbo',prompt='Watercolor painting of the same scene',task='img2img',strength=0.6,image='source')),
    ('zimage-inpaint-512','image',dict(model_id='z-image-turbo',prompt='A large straw hat with a red ribbon',task='inpaint',strength=0.9,image='source',mask=True)),
    ('zimage-text-1024','image',dict(model_id='z-image-turbo',width=1024,height=1024)),
    ('sdxl-text-1024','image',dict(model_id='sdxl-lightning',steps=4,width=1024,height=1024)),
    ('sdxl-img2img-768x512','image',dict(model_id='sdxl-lightning',steps=4,width=768,height=512,prompt='Watercolor painting',task='img2img',strength=0.5,image='source')),
    ('sdxl-inpaint-512','image',dict(model_id='sdxl-lightning',steps=4,prompt='A large straw hat',task='inpaint',strength=0.9,image='source',mask=True)),
    ('qwen-text-1024-negative','image',dict(model_id='qwen-image-2.1',width=1024,height=1024,negative_prompt='blurry, text, watermark')),
    ('ltx-i2v-33','video',dict(model_id='ltx-video-2b-distilled',frames=33,image='source',prompt='She turns her head and smiles, hair moving in the breeze')),
    ('ltx-t2v-after-i2v','video',dict(model_id='ltx-video-2b-distilled',frames=33)),
    ('ltx-t2v-motion','video',dict(model_id='ltx-video-2b-distilled',frames=33,
        prompt='An adult anime catgirl reading a book by a sunlit window. Her hair moves gently in the breeze. Slow smooth camera movement.')),
    ('wan22-i2v-33','video',dict(model_id='wan22-ti2v-5b',frames=33,image='source',negative_prompt='blurry, distorted',
                                 prompt='She turns her head and smiles, hair moving in the breeze')),
    # 0.10 follow-ups: feathered inpainting, video sizes, LTX keyframes and dual-GPU image-to-video.
    ('zimage-inpaint-feather','image',dict(model_id='z-image-turbo',prompt='A large straw hat with a red ribbon',task='inpaint',strength=0.9,image='source',mask=True,feather=12)),
    ('ltx-t2v-portrait','video',dict(model_id='ltx-video-2b-distilled',frames=33,size='448x768',
        prompt='An adult anime catgirl walking toward the camera on a sunny street. Her hair moves in the wind. Smooth tracking shot.')),
    ('ltx-keyframes-square','video',dict(model_id='ltx-video-2b-distilled',frames=33,size='512x512',image='source',last_image='mirrored',
        prompt='She slowly turns from one side to the other, hair moving gently')),
    ('wan22-i2v-portrait','video',dict(model_id='wan22-ti2v-5b',frames=33,size='448x768',image='source',
        prompt='She turns her head and smiles, hair moving in the breeze')),
    ('wan21-t2v-square','video',dict(model_id='wan21-t2v-1.3b',frames=33,size='624x624',
        prompt='An adult anime catgirl reading a book by a sunlit window. Her hair moves gently in the breeze. Slow smooth camera movement.')),
]


def mirrored(raw):
    """The source image flipped horizontally, as a distinct last frame."""
    from PIL import Image,ImageOps
    buffer=io.BytesIO();ImageOps.mirror(Image.open(io.BytesIO(raw))).save(buffer,format='PNG');return buffer.getvalue()


PROBE='''
import sys,json,subprocess,traceback
info={}
_value=getattr(sys,'last_value',None);_parts=[]
while _value is not None and len(_parts)<6:
    _parts.append(type(_value).__name__+': '+str(_value)[:400]);_value=_value.__cause__ or _value.__context__
if _parts:info['exception_chain']=' <- '.join(_parts)
try:
    import os
    info['statvfs_free_gib']={p:round(os.statvfs(p).f_bavail*os.statvfs(p).f_frsize/2**30,1) for p in ('/kaggle/temp','/kaggle/working','/tmp','/') if os.path.exists(p)}
    info['mounts']=[l.strip() for l in open('/proc/mounts') if any(k in l.split()[1] for k in ('/kaggle','/tmp')) or l.split()[1]=='/'][:8]
    info['mem_available_gib']=round(int([l for l in open('/proc/meminfo') if l.startswith('MemAvailable')][0].split()[1])/2**20,1)
    info['kaggle_temp_used_gib']=round(int(subprocess.run(['du','-sb','/kaggle/temp'],capture_output=True,text=True,timeout=120).stdout.split()[0])/2**30,1)
except Exception as error:info['system']=repr(error)
if getattr(sys,'last_traceback',None) is not None:
    info['traceback']='\n'.join(l for l in ''.join(traceback.format_exception(sys.last_type,sys.last_value,sys.last_traceback)).splitlines() if l.strip().strip('^~')!='')[-6000:]
try:
    import torch
    info['torch_allocated_gib']=[round(torch.cuda.memory_allocated(i)/2**30,2) for i in range(torch.cuda.device_count())]
except Exception as error:info['torch']=repr(error)
try:
    import shutil
    from pathlib import Path
    info['kaggle_temp_free_gib']=round(shutil.disk_usage('/kaggle/temp').free/2**30,1)
    roots=[Path('/kaggle/temp/cc-media-models'),Path('/kaggle/temp/cc-media-video')]
    folders=[f for root in roots if root.is_dir() for f in root.iterdir()]+[Path('/kaggle/temp/qwen-image-assets')]
    info['model_folders_gib']={str(f):round(sum(x.stat().st_size for x in f.rglob('*') if x.is_file())/2**30,1) for f in folders if f.is_dir()}
except Exception as error:info['disk']=repr(error)
try:info['nvidia_smi_used_mib']=subprocess.run(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=30).stdout.split()
except Exception as error:info['nvidia_smi']=repr(error)
print('CC_PROBE:'+json.dumps(info),flush=True)
'''


def probe(backend):
    """Attach to the session's kernel and report the last traceback and GPU memory."""
    from session_runtime import JupyterTransport
    url=backend._read().get('jupyter_url')
    if not url:return {'probe':'no session'}
    transport=JupyterTransport(url)
    try:
        transport.start()
        line=[l for l in transport.execute(PROBE,timeout=120).splitlines() if l.startswith('CC_PROBE:')]
        return json.loads(line[0][9:]) if line else {'probe':'no output'}
    except Exception as error:return {'probe_error':type(error).__name__+': '+str(error)[:500]}
    finally:transport.close()


def run(user,token,output,only=None,diagnose=True,source=None):
    from account_backend import AccountBackend
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    backend=AccountBackend(user,token,cache_dir=output/'sessions')
    reports=[];last=[0.0]
    def progress(name):
        def report(event):
            now=time.monotonic()
            if event.get('phase')=='download' and now-last[0]<20:return
            last[0]=now
            print('GPU_CHECK_PROGRESS '+name+' '+json.dumps({k:event.get(k) for k in ('phase','message','percent','step','steps','file') if event.get(k) is not None},ensure_ascii=False),flush=True)
        return report
    try:
        for name,kind,arguments in CASES:
            if only and name not in only:continue
            arguments=dict(arguments);prompt=arguments.pop('prompt',PROMPT);started=time.monotonic()
            try:
                if arguments.get('image')=='source':
                    if source is None:raise RuntimeError('No source image from the first case')
                    arguments['image']=source
                if arguments.pop('mask',False):arguments['mask']=mask_png(512,512)
                if arguments.get('last_image')=='mirrored':arguments['last_image']=mirrored(source)
                print('GPU_CHECK_START '+name,flush=True)
                if kind=='image':
                    raw,stats=backend.generate_with_progress(prompt,42,arguments.pop('steps',None),progress(name),**arguments)
                    expected=(arguments.get('width',512),arguments.get('height',512))
                    if image_size(raw)!=expected:raise RuntimeError('PNG is %s, expected %s'%(image_size(raw),expected))
                    path=output/(name+'.png')
                    if source is None:source=raw
                else:
                    raw,stats=backend.generate_video_with_progress(prompt,42,None,progress=progress(name),**arguments)
                    path=output/(name+'.mp4')
                path.write_bytes(raw)
                reports.append(dict(case=name,ok=True,wall_seconds=round(time.monotonic()-started,1),path=str(path),
                    **{k:v for k,v in stats.items() if k!='png'}))
            except Exception as error:
                traceback.print_exc()
                # Backends hide transport errors (they may contain connection URLs); the
                # remote exception name and message are still chained as __context__.
                cause=error.__context__
                reports.append(dict(case=name,ok=False,wall_seconds=round(time.monotonic()-started,1),error=type(error).__name__+': '+str(error),
                    **({'remote_error':(type(cause).__name__+': '+str(cause))[:2000]} if cause is not None else {})))
                if diagnose:reports[-1]['probe']=probe(backend)
            print('GPU_CHECK_RESULT '+json.dumps(reports[-1],ensure_ascii=False),flush=True)
            (output/'report.json').write_text(json.dumps(reports,indent=2,ensure_ascii=False),encoding='utf-8')
        if diagnose:
            final=probe(backend);print('GPU_CHECK_SYSTEM '+json.dumps({k:v for k,v in final.items() if k!='traceback'},ensure_ascii=False),flush=True)
            reports.append(dict(case='system',ok=True,probe=final))
            (output/'report.json').write_text(json.dumps(reports,indent=2,ensure_ascii=False),encoding='utf-8')
    finally:
        try:backend.release_session();print('GPU_CHECK_RELEASED',flush=True)
        except Exception as error:print('GPU_CHECK_RELEASE_FAILED '+type(error).__name__+': '+str(error),flush=True)
        backend.close()
    return reports


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--output',default='gpu-check-results')
    parser.add_argument('--only',nargs='*',help='Run only these case names')
    parser.add_argument('--source',type=Path,help='PNG used as reference image instead of the first text result')
    args=parser.parse_args()
    reports=run(os.environ['KAGGLE_USERNAME'],os.environ.get('KAGGLE_API_TOKEN') or os.environ['KAGGLE_KEY'],args.output,args.only,
        source=args.source.read_bytes() if args.source else None)
    cases=[r for r in reports if r['case']!='system']
    print('GPU_CHECK_FINISHED %d/%d passed'%(sum(r['ok'] for r in cases),len(cases)),flush=True)
    raise SystemExit(0 if all(r['ok'] for r in cases) else 1)
