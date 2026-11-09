from .node import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
from .easy_node import KaggleEasyImage
NODE_CLASS_MAPPINGS['KaggleEasyImage'] = KaggleEasyImage
NODE_DISPLAY_NAME_MAPPINGS['KaggleEasyImage'] = 'Kaggle Easy Image'
try:
    from .session_node import KaggleQwenSession
except ImportError:
    pass  # Optional experimental dependencies do not block the basic node.
else:
    NODE_CLASS_MAPPINGS['KaggleQwenSession'] = KaggleQwenSession
    NODE_DISPLAY_NAME_MAPPINGS['KaggleQwenSession'] = 'Kaggle Qwen Session (Experimental)'
