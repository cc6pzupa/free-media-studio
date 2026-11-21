from .node import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
from .video_node import CCMediaVideo
NODE_CLASS_MAPPINGS['CCMediaVideo']=CCMediaVideo
NODE_DISPLAY_NAME_MAPPINGS['CCMediaVideo']='CC Media Video (Preview)'
from .easy_node import KaggleEasyImage, CCMediaImageEdit
NODE_CLASS_MAPPINGS['KaggleEasyImage'] = KaggleEasyImage
NODE_DISPLAY_NAME_MAPPINGS['KaggleEasyImage'] = 'CC Media Image'
NODE_CLASS_MAPPINGS['CCMediaImageEdit'] = CCMediaImageEdit
NODE_DISPLAY_NAME_MAPPINGS['CCMediaImageEdit'] = 'CC Media Image Edit (Preview)'
try:
    from .session_node import KaggleQwenSession
except ImportError:
    pass  # Optional experimental dependencies do not block the basic node.
else:
    NODE_CLASS_MAPPINGS['KaggleQwenSession'] = KaggleQwenSession
    NODE_DISPLAY_NAME_MAPPINGS['KaggleQwenSession'] = 'Kaggle Qwen Session (Experimental)'
