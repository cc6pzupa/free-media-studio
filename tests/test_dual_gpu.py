from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock,patch
import unittest
from dual_gpu import ParallelCFG,prepare_parallel_replica,restore_encoder

class Tensor:
    def __init__(self,value,shape=(1,4),device='cuda:0'):self.value,self.shape,self.device=value,shape,device
    def to(self,device):return Tensor(self.value,self.shape,device)
    def new_ones(self,*shape):return Tensor(1,shape,self.device)

class Event:
    def record(self,*args):pass
    def synchronize(self):pass

class Cache:
    def __init__(self,layers):self.layers=layers

class Model:
    def __init__(self,device):self.device=device;self.transformer_blocks=[1];self.seen=[]
    def forward(self,**kwargs):
        self.seen.append(kwargs)
        return (Tensor(kwargs['hidden_states'].value+kwargs['encoder_hidden_states'].value,device=self.device),)
    def __call__(self,**kwargs):return self.forward(**kwargs)
    def cache_context(self,name):return nullcontext()


def fake_torch():
    return SimpleNamespace(inference_mode=nullcontext,bool=bool,
        zeros_like=lambda mask,**k:Tensor(0,mask.shape,mask.device),
        cat=lambda values,dim:Tensor(0,(values[0].shape[0],sum(v.shape[1] for v in values)),values[0].device),
        cuda=SimpleNamespace(device=lambda i:nullcontext(),Event=Event,current_stream=lambda i:None,empty_cache=Mock()))

class DualGPUTest(unittest.TestCase):
    def inputs(self):
        return dict(hidden_states=Tensor(10,(1,4)),timestep=Tensor(1),encoder_hidden_states=Tensor(1),
            encoder_hidden_states_mask=Tensor(1),img_mask=Tensor(0,(1,5)),kv_cache=Cache(1),kv_cache_mode='extract',return_dict=False)

    def test_branches_run_on_distinct_models_and_preserve_negative_mask(self):
        primary,secondary=Model('cuda:0'),Model('cuda:1');original=primary.forward
        negative=(Tensor(2,(1,2)),Tensor(1,(1,2)),Tensor(0,(1,2)))
        with ParallelCFG(primary,secondary,negative,fake_torch()) as parallel:
            positive=primary(**self.inputs())[0]
            negative_result=primary(**self.inputs())[0]
            self.assertEqual(positive.value,11);self.assertEqual(negative_result.value,12)
            self.assertEqual(negative_result.device,'cuda:0');self.assertEqual(parallel.calls,1)
        self.assertEqual(primary.forward,original)
        self.assertEqual(len(primary.seen),1);self.assertEqual(len(secondary.seen),1)
        self.assertEqual(secondary.seen[0]['hidden_states'].device,'cuda:1')
        self.assertEqual(secondary.seen[0]['img_mask'].shape,(1,3))
        self.assertEqual(primary.seen[0]['img_mask'].shape,(1,5))
        self.assertIsNot(secondary.seen[0]['kv_cache'],primary.seen[0]['kv_cache'])

    def test_failed_secondary_restores_original_forward_and_single_gpu_still_runs(self):
        primary,secondary=Model('cuda:0'),Model('cuda:1');original=primary.forward
        secondary.forward=Mock(side_effect=RuntimeError('experimental failure'))
        with self.assertRaisesRegex(RuntimeError,'experimental'):
            with ParallelCFG(primary,secondary,(Tensor(2),Tensor(1),Tensor(0)),fake_torch()):
                primary(**self.inputs());primary(**self.inputs())
        self.assertEqual(primary.forward,original)
        self.assertEqual(primary(**self.inputs())[0].value,11)

    def test_replica_loading_failure_moves_encoder_back(self):
        encoder=Mock();torch=fake_torch()
        with patch('dual_gpu.load_replica',side_effect=RuntimeError('allocation failed')):
            with self.assertRaises(RuntimeError):prepare_parallel_replica('/unused',encoder,torch)
        self.assertEqual([c.args[0] for c in encoder.to.call_args_list],['cpu','cuda:1'])

    def test_cleanup_evicts_replica_before_restoring_encoder(self):
        order=[]
        encoder=SimpleNamespace(to=lambda d:order.append(('encoder',d)))
        replica=SimpleNamespace(to=lambda d:order.append(('replica',d)))
        restore_encoder(encoder,replica,fake_torch())
        self.assertEqual(order,[('replica','cpu'),('encoder','cuda:1')])

if __name__=='__main__':unittest.main()
