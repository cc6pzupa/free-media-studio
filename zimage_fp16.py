"""Range-preserving rescaling for Z-Image's sandwich RMSNorm blocks."""


def stabilize_zimage_fp16(transformer):
    import torch
    if getattr(transformer,'_cc_fp16_stabilized',False):return 0
    if transformer.dtype!=torch.float16:return 0
    count=0
    with torch.no_grad():
        for block in transformer.modules():
            if type(block).__name__!='ZImageTransformerBlock':continue
            # Scaling before a linear operation avoids FP16 intermediate
            # overflow. Adjust epsilon so the following RMSNorm is equivalent:
            # norm(x/s, eps/s²) == norm(x, eps).
            for linear,norm,divisor in ((block.feed_forward.w3,block.ffn_norm2,32.),
                                       (block.attention.to_out[0],block.attention_norm2,4.)):
                linear.weight.div_(divisor)
                if linear.bias is not None:linear.bias.div_(divisor)
                norm.eps/=divisor*divisor
            count+=1
    if not count:raise RuntimeError('Unsupported Z-Image FP16 block layout')
    transformer._cc_fp16_stabilized=True
    return count
