"""Run bounded, independent GPU checks for the video preview branch."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import traceback


def run(models,output='/kaggle/working/cc-video-results'):
    from video_assets import prepare_video_assets
    from video_registry import video_profile
    from video_process import VideoProcess
    target=Path(output);target.mkdir(parents=True,exist_ok=True);reports=[]
    def progress(model,event):
        if event.get('phase')=='download' and event.get('percent',0)%10>=.5:return
        print('VIDEO_CHECK_PROGRESS:'+json.dumps(dict(event,model_id=model)),flush=True)
    with ThreadPoolExecutor(max_workers=3) as pool:
        assets={model:pool.submit(prepare_video_assets,video_profile(model),
            Path('/kaggle/temp/cc-media-video')/model,lambda event,model=model:progress(model,event)) for model in models}
        for model in models:
            actor=None
            try:
                assets[model].result()
                print('VIDEO_CHECK_START:'+model,flush=True)
                actor=VideoProcess(model,progress=lambda event:progress(model,event))
                cases=[(33,8),(33,8)] if model=='ltx-video-2b-distilled' else ([(9,8),(33,30)] if model=='wan22-ti2v-5b' else [(9,4),(9,4)])
                for index,(frames,steps) in enumerate(cases):
                    path,stats=actor.generate('An adult anime catgirl reading a book by a sunlit window. Her hair moves gently in the breeze. Slow smooth camera movement.',
                        frames=frames,steps=steps,seed=42,output=str(target/(model+'-check-'+str(index)+'.mp4')))
                    reports.append(dict(stats,path=str(path)))
                    (target/'video-check-report.json').write_text(json.dumps(reports,indent=2))
                    print('VIDEO_CHECK_RESULT:'+json.dumps(reports[-1]),flush=True)
            except Exception as error:
                traceback.print_exc();reports.append(dict(model_id=model,error=type(error).__name__+': '+str(error)))
                (target/'video-check-report.json').write_text(json.dumps(reports,indent=2))
                print('VIDEO_CHECK_ERROR:'+json.dumps(reports[-1]),flush=True)
            finally:
                if actor is not None:actor.close()
    print('VIDEO_CHECK_FINISHED:'+json.dumps(reports),flush=True)
    return reports


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--models',nargs='+',default=['ltx-video-2b-distilled','wan22-ti2v-5b','wan21-t2v-1.3b'])
    args=parser.parse_args();reports=run(args.models)
    raise SystemExit(1 if any('error' in result for result in reports) else 0)
