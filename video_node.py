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
            'optional':{'image':('IMAGE',),'negative_prompt':('STRING',{'multiline':True,'default':''}),
                        # Empty uses the model's default landscape size.
                        'size':(['']+sorted({size for p in VIDEO_MODELS.values() for size in p['sizes']}),),
                        'last_image':('IMAGE',)}}
    RETURN_TYPES=('STRING',)
    RETURN_NAMES=('mp4_path',)
    FUNCTION='generate'
    CATEGORY='CC Media Studio'

    def generate(self,prompt,model_id,seed,frames,image=None,negative_prompt='',size='',last_image=None):
        client=StudioClient()
        try:
            from .easy_node import tensor_png
            first=tensor_png(image) if image is not None else None
            last=tensor_png(last_image) if last_image is not None else None
            job=client.submit_video(prompt,seed=seed,model_id=model_id,frames=frames,image=first,
                negative_prompt=negative_prompt or None,size=size or None,last_image=last)
            client.wait(job['job_id'])
            return (str((client.directory/'images'/(job['job_id']+'.mp4')).resolve()),)
        finally:client.close()
