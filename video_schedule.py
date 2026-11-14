"""Preserve the distilled checkpoint schedule through LTX's pipeline defaults."""


def ltx_distilled_scheduler(config,trained_sigmas):
    from diffusers import FlowMatchEulerDiscreteScheduler
    trained=tuple(float(value) for value in trained_sigmas)
    if len(trained)!=8 or trained[0]!=1.0 or not all(0<b<a<=1 for a,b in zip(trained,trained[1:])):
        raise ValueError('Invalid distilled LTX schedule')
    class CCLTXDistilledScheduler(FlowMatchEulerDiscreteScheduler):
        def set_timesteps(self,num_inference_steps=None,device=None,sigmas=None,mu=None,timesteps=None):
            if timesteps is not None:
                # LTXConditionPipeline (first + last frame) always passes explicit
                # timesteps; accept them only when they are the trained schedule.
                values=[float(value) for value in timesteps]
                if len(values)!=len(trained) or any(abs(value-1000*sigma)>0.5 for value,sigma in zip(values,trained)):
                    raise ValueError('Distilled LTX requires its trained eight-step sigma schedule')
                return super().set_timesteps(device=device,sigmas=list(trained),mu=mu)
            count=len(sigmas) if sigmas is not None else num_inference_steps
            if count!=len(trained):
                raise ValueError('Distilled LTX requires its trained eight-step sigma schedule')
            # LTXPipeline supplies a linear sigma array even when custom timesteps
            # are requested. Replace that array at the scheduler boundary.
            return super().set_timesteps(device=device,sigmas=list(trained),mu=mu)
    return CCLTXDistilledScheduler.from_config(config,use_dynamic_shifting=False,shift=1.0)
