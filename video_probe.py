"""Explicit GPU probe, intended for the separate video test notebook."""
import argparse
import json
from pathlib import Path


def run(model,frames=9,steps=None,runs=1,acceleration='auto',output='/kaggle/working/cc-video-results'):
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
    return runtime


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',default='wan21-t2v-1.3b')
    parser.add_argument('--frames',type=int,default=9)
    parser.add_argument('--steps',type=int)
    parser.add_argument('--runs',type=int,default=1)
    parser.add_argument('--acceleration',choices=['auto','split'],default='auto')
    args=parser.parse_args()
    runtime=run(args.model,args.frames,args.steps,args.runs,args.acceleration)
