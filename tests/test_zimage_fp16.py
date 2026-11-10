import contextlib
import math
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from zimage_fp16 import stabilize_zimage_fp16

class Weight:
    def __init__(self,values):self.values=list(values)
    def div_(self,divisor):self.values=[v/divisor for v in self.values]

class ZImageTransformerBlock:
    def __init__(self):
        linear=lambda:SimpleNamespace(weight=Weight([60000.,-50000.,10000.]),bias=Weight([2.]))
        self.feed_forward=SimpleNamespace(w3=linear())
        self.attention=SimpleNamespace(to_out=[linear()])
        self.ffn_norm2=SimpleNamespace(eps=1e-5)
        self.attention_norm2=SimpleNamespace(eps=1e-5)

class ZImageFP16Test(unittest.TestCase):
    def test_rescaling_preserves_rmsnorm_and_is_not_applied_twice(self):
        block=ZImageTransformerBlock();dtype=object()
        model=SimpleNamespace(dtype=dtype,modules=lambda:[block])
        before=list(block.feed_forward.w3.weight.values)
        norm=lambda v,eps:[x/math.sqrt(sum(x*x for x in v)/len(v)+eps) for x in v]
        expected=norm(before,block.ffn_norm2.eps)
        with patch.dict(sys.modules,{'torch':SimpleNamespace(float16=dtype,no_grad=contextlib.nullcontext)}):
            self.assertEqual(stabilize_zimage_fp16(model),1)
            self.assertEqual(stabilize_zimage_fp16(model),0)
        actual=norm(block.feed_forward.w3.weight.values,block.ffn_norm2.eps)
        for a,b in zip(expected,actual):self.assertAlmostEqual(a,b,places=12)
        self.assertEqual(block.feed_forward.w3.weight.values,[x/32 for x in before])
        self.assertEqual(block.attention.to_out[0].weight.values,[x/4 for x in before])
        self.assertEqual(block.feed_forward.w3.bias.values,[2/32])

    def test_other_precisions_are_unchanged_and_unknown_fp16_layout_is_rejected(self):
        dtype=object();other=object();model=SimpleNamespace(dtype=other,modules=lambda:[])
        with patch.dict(sys.modules,{'torch':SimpleNamespace(float16=dtype,no_grad=contextlib.nullcontext)}):
            self.assertEqual(stabilize_zimage_fp16(model),0)
            model.dtype=dtype
            with self.assertRaisesRegex(RuntimeError,'Unsupported'):stabilize_zimage_fp16(model)

if __name__=='__main__':unittest.main()
