"""Explicit device moves for INT8 video encoders without changing image helpers."""


def move_video_encoder(model,device):
    import torch
    from bitsandbytes.nn import Linear8bitLt
    target=torch.device(device)
    if all(p.device==target for p in model.parameters()):return model
    layers=[]
    for module in model.modules():
        if not isinstance(module,Linear8bitLt):continue
        scale=module.state.SCB if module.state.SCB is not None else module.weight.SCB
        if scale is None:raise RuntimeError('Missing INT8 video scale')
        layers.append((module,scale))
    for module,_ in layers:
        module.state.CB=None;module.state.SCB=None;module.weight.CB=None;module.weight.SCB=None
    try:
        # Transformers forbids .to() on some 8-bit model classes even though
        # torch/bitsandbytes support moving their quantized parameters.
        return torch.nn.Module.to(model,target)
    finally:
        for module,scale in layers:
            module.state.CB=module.weight.data;module.state.SCB=scale.to(module.weight.device)
            module.weight.CB=None;module.weight.SCB=None
