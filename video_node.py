"""ComfyUI video node returning a local MP4 path for video loader nodes."""
from pathlib import Path
from .studio_client import StudioClient
from .video_registry import VIDEO_MODELS


class CCMediaVideo:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required':{'prompt':('STRING',{'multiline':True}),
            'model_id':(list(VIDEO_MODELS),),
            'seed':('INT',{'default':42,'min':0,'max':2147483647}),
            'frames':('INT',{'default':33,'min':9,'max':81,'step':8})},
            # Optional first frame: LTX-Video 2B and Wan2.2 5B support image-to-video.
            'optional':{'image':('IMAGE',),'negative_prompt':('STRING',{'multiline':True,'default':''})}}
    RETURN_TYPES=('STRING',)
    RETURN_NAMES=('mp4_path',)
    FUNCTION='generate'
    CATEGORY='CC Media Studio'

    def generate(self,prompt,model_id,seed,frames,image=None,negative_prompt=''):
        client=StudioClient()
        try:
            first=None
            if image is not None:
                from .easy_node import tensor_png
                first=tensor_png(image)
            job=client.submit_video(prompt,seed=seed,model_id=model_id,frames=frames,image=first,negative_prompt=negative_prompt or None)
            client.wait(job['job_id'])
            return (str((client.directory/'images'/(job['job_id']+'.mp4')).resolve()),)
        finally:client.close()
