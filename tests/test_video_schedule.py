import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from video_registry import VIDEO_MODELS
from video_schedule import ltx_distilled_scheduler


class Scheduler:
    @classmethod
    def from_config(cls,config,**kwargs):
        result=cls();result.config=dict(config,**kwargs);return result
    def set_timesteps(self,**kwargs):self.actual=kwargs


class VideoScheduleTest(unittest.TestCase):
    def test_pipeline_linear_sigmas_are_replaced_by_checkpoint_schedule(self):
        trained=VIDEO_MODELS['ltx-video-2b-distilled']['sigmas']
        with patch.dict(sys.modules,{'diffusers':SimpleNamespace(FlowMatchEulerDiscreteScheduler=Scheduler)}):
            scheduler=ltx_distilled_scheduler({},trained)
        scheduler.set_timesteps(sigmas=[1-i/8 for i in range(8)],device='cuda:1',mu=1.2)
        self.assertEqual(scheduler.actual['sigmas'],trained)
        self.assertEqual(scheduler.actual['device'],'cuda:1')
        self.assertFalse(scheduler.config['use_dynamic_shifting']);self.assertEqual(scheduler.config['shift'],1)
        with self.assertRaises(ValueError):scheduler.set_timesteps(sigmas=[1,.5])


if __name__=='__main__':unittest.main()
