import types
import unittest
from unittest.mock import patch
from video_precision import move_video_encoder


class Scale:
    def __init__(self,device):self.device=device
    def to(self,device):return Scale(device)


class Linear:
    def __init__(self):
        self.state=types.SimpleNamespace(CB='old-alias',SCB=Scale('cuda:1'))
        self.weight=types.SimpleNamespace(CB=None,SCB=None,device='cuda:1',data='gpu-data')


class Encoder:
    def __init__(self):self.layer=Linear();self.moves=[]
    def modules(self):return [self.layer]
    def parameters(self):return [self.layer.weight]
    def to(self,*args):raise AssertionError('Transformers .to must not be called')


class VideoPrecisionTest(unittest.TestCase):
    def test_roundtrip_uses_module_move_and_rebinds_unregistered_quantization_state(self):
        def move(model,target):
            layer=model.layer
            self.assertIsNone(layer.state.CB);self.assertIsNone(layer.state.SCB)
            layer.weight.device=target;layer.weight.data='data-'+target;model.moves.append(target)
            return model
        torch=types.SimpleNamespace(device=lambda x:x,nn=types.SimpleNamespace(Module=types.SimpleNamespace(to=move)))
        bnb=types.ModuleType('bitsandbytes.nn');bnb.Linear8bitLt=Linear
        with patch.dict('sys.modules',{'torch':torch,'bitsandbytes.nn':bnb}):
            model=Encoder();move_video_encoder(model,'cuda:1');self.assertEqual(model.moves,[])
            for device in ['cpu','cuda:1','cpu']:
                move_video_encoder(model,device)
                self.assertEqual(model.layer.state.SCB.device,device)
                self.assertEqual(model.layer.state.CB,model.layer.weight.data)
                self.assertIsNone(model.layer.weight.SCB)
            self.assertEqual(model.moves,['cpu','cuda:1','cpu'])
