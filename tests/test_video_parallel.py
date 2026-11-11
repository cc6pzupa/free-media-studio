import contextlib
import types
import unittest
from video_parallel import WanParallelCFG


class Tensor:
    def __init__(self,value,device='cuda:0'):self.value=value;self.device=device
    def to(self,device):return Tensor(self.value,device)


class Event:
    def record(self,*args):pass
    def synchronize(self):pass


TORCH=types.SimpleNamespace(is_tensor=lambda x:isinstance(x,Tensor),
    inference_mode=contextlib.nullcontext,cuda=types.SimpleNamespace(
        device=lambda _:contextlib.nullcontext(),Event=Event,current_stream=lambda _:None,synchronize=lambda _:None))


class Model:
    def __init__(self,fail=False):self.seen=[];self.fail=fail
    def forward(self,**kwargs):
        self.seen.append(kwargs)
        if self.fail:raise RuntimeError('fixture failure')
        return (Tensor(kwargs['encoder_hidden_states'].value),)
    def __call__(self,**kwargs):return self.forward(**kwargs)


class VideoParallelTest(unittest.TestCase):
    def test_both_cfg_branches_run_and_nested_tensors_use_the_secondary_device(self):
        primary=Model();secondary=Model();original=primary.forward
        with WanParallelCFG(primary,secondary,Tensor('negative'),TORCH) as router:
            for step in range(2):
                kwargs=dict(hidden_states=Tensor(step),timestep=Tensor(step),
                    encoder_hidden_states=Tensor('positive'),attention_kwargs={'mask':Tensor('mask')},return_dict=False)
                self.assertEqual(primary(**kwargs)[0].value,'positive')
                kwargs['encoder_hidden_states']=Tensor('negative')
                output=primary(**kwargs)[0]
                self.assertEqual((output.value,output.device),('negative','cuda:0'))
        self.assertEqual(router.calls,2);self.assertEqual(primary.forward,original)
        self.assertEqual(len(primary.seen),2);self.assertEqual(len(secondary.seen),2)
        self.assertTrue(all(k['attention_kwargs']['mask'].device=='cuda:1' for k in secondary.seen))

    def test_primary_failure_restores_forward_and_closes_the_pool(self):
        primary=Model(fail=True);original=primary.forward
        with self.assertRaisesRegex(RuntimeError,'fixture failure'):
            with WanParallelCFG(primary,Model(),Tensor('negative'),TORCH):
                primary(hidden_states=Tensor(1),encoder_hidden_states=Tensor('positive'),return_dict=False)
        self.assertEqual(primary.forward,original)

    def test_unpaired_forwards_are_rejected(self):
        primary=Model();original=primary.forward
        with self.assertRaisesRegex(RuntimeError,'Unpaired'):
            with WanParallelCFG(primary,Model(),Tensor('negative'),TORCH):
                primary(hidden_states=Tensor(1),encoder_hidden_states=Tensor('positive'),return_dict=False)
        self.assertEqual(primary.forward,original)


if __name__=='__main__':unittest.main()
