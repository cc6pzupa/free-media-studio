"""Explicit GPU probe, intended for the separate video test notebook."""
import argparse
import json
from pathlib import Path


def run(model,frames=9,steps=None,runs=1,acceleration='auto',output='/kaggle/working/cc-video-results',compare_split=False):
    from video_runtime import VideoRuntime
    target=Path(output);target.mkdir(parents=True,exist_ok=True)
    def progress(event):
        if event.get('phase')!='download' or event.get('percent',0)%5<.3:
            print('VIDEO_PROGRESS:'+json.dumps(event),flush=True)
    runtime=VideoRuntime(model,acceleration=acceleration,progress=progress).prepare()
    reports=[]
    for index in range(runs):
        path,stats=runtime.generate('An adult anime catgirl reading a book by a sunlit window. '
            'Her hair moves gently in the breeze. Slow smooth camera movement.',
            seed=42,steps=steps,frames=frames,output=target/(model+'-'+str(index)+'.mp4'))
        reports.append(stats);print('VIDEO_RESULT:'+json.dumps(stats),flush=True)
        (target/(model+'-report.json')).write_text(json.dumps(reports,indent=2),encoding='utf-8')
    if compare_split and model!='ltx-video-2b-distilled':
        runtime.acceleration='split'
        path,stats=runtime.generate('An adult anime catgirl reading a book by a sunlit window. '
            'Her hair moves gently in the breeze. Slow smooth camera movement.',seed=42,
            steps=steps,frames=frames,output=target/(model+'-split.mp4'))
        print('VIDEO_SPLIT_GENERATED:'+json.dumps(stats),flush=True)
        import imageio
        import numpy as np
        from itertools import zip_longest
        maximum=0;count=0
        with imageio.get_reader(str(target/(model+'-0.mp4')),'ffmpeg') as left, imageio.get_reader(str(path),'ffmpeg') as right:
            for a,b in zip_longest(left,right):
                if a is None or b is None or a.shape!=b.shape:raise RuntimeError('Video comparison shape mismatch')
                maximum=max(maximum,int(np.abs(a.astype(np.int16)-b.astype(np.int16)).max()));count+=1
        if not count:raise RuntimeError('Empty video comparison')
        stats['decoded_max_pixel_error']=maximum
        reports.append(stats);print('VIDEO_SPLIT_RESULT:'+json.dumps(stats),flush=True)
        (target/(model+'-report.json')).write_text(json.dumps(reports,indent=2),encoding='utf-8')
    return runtime


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',default='wan21-t2v-1.3b')
    parser.add_argument('--frames',type=int,default=9)
    parser.add_argument('--steps',type=int)
    parser.add_argument('--runs',type=int,default=1)
    parser.add_argument('--acceleration',choices=['auto','split'],default='auto')
    parser.add_argument('--compare-split',action='store_true')
    args=parser.parse_args()
    runtime=run(args.model,args.frames,args.steps,args.runs,args.acceleration,compare_split=args.compare_split)
