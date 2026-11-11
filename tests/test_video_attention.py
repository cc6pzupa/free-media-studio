import unittest
from types import SimpleNamespace
from video_attention import broadcast_ltx_prompt_mask


class Mask:
    ndim=4
    def __init__(self,heads):self.heads=heads;self.shape=(1,len(heads),1,len(heads[0]))
    def __getitem__(self,index):return Mask(self.heads[index[1]])
    def expand_as(self,other):return Mask(self.heads*other.shape[1])


class VideoAttentionTest(unittest.TestCase):
    def test_preserves_padding_while_broadcasting_identical_head_masks(self):
        mask=Mask([(0,-10000)]*32)
        torch=SimpleNamespace(equal=lambda a,b:a.heads==b.heads)
        shared=broadcast_ltx_prompt_mask(mask,torch)
        self.assertEqual(shared.shape,(1,1,1,2))
        self.assertEqual(shared.expand_as(mask).heads,mask.heads)
        self.assertIsNone(broadcast_ltx_prompt_mask(None))

    def test_rejects_head_specific_masks_instead_of_changing_attention(self):
        mask=Mask([(0,-10000)]*31+[(0,0)])
        torch=SimpleNamespace(equal=lambda a,b:a.heads==b.heads)
        with self.assertRaises(ValueError):broadcast_ltx_prompt_mask(mask,torch)


if __name__=='__main__':unittest.main()
