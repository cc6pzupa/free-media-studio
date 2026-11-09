import importlib.util
import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from account_backend import AccountBackend
from studio_core import StudioCore, normalize_config
from usage_stats import duration_seconds, normalize_quota

spec=importlib.util.spec_from_file_location('retention_fixture',Path(__file__).with_name('fixtures')/'fake_worker.py')
fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)
CONFIG={'mode':'account','user':'owner','token':'KGAT_fixture_secret'}

class ReleasableBackend(fixture.FakeBackend):
    def __init__(self,*args):
        super().__init__(*args);self.releases=0;self.quota_calls=0;self.fail_release=False
    def release_session(self):
        self.releases+=1
        if self.fail_release:raise RuntimeError('private transport detail')
    def quota(self):
        self.quota_calls+=1
        return normalize_quota({'gpuQuota':{'timeUsed':'3600s','timeReserved':'1800s','totalTimeAllowed':'108000s'},'quotaRefreshTime':'2026-11-16T00:00:00Z'})

class UsageRetentionTest(unittest.TestCase):
    def make_core(self,d):
        with patch('studio_core.load_config',return_value=None):core=StudioCore(d,backend_factory=ReleasableBackend)
        core.configure(CONFIG);self.addCleanup(core.close)
        return core

    def finish(self,core,job):
        core.start()
        for _ in range(200):
            if core.store.get(job)['state']=='complete' and core.idle_since is not None:return
            time.sleep(.02)
        self.fail('generation timed out')

    def test_quota_durations_allowlist_and_unknown_fields(self):
        for raw,expected in [('3600s',3600),('1.500s',1.5),({'seconds':'3','nanos':500000000},3.5),(0,0)]:
            self.assertEqual(duration_seconds(raw),expected)
        for raw in [None,True,-1,'NaN',float('inf'),'secret']:
            self.assertIsNone(duration_seconds(raw))
        result=normalize_quota({'gpuQuota':{'timeUsed':'7200s','timeReserved':'3600s','totalTimeAllowed':'108000s','secret':'hidden'},'token':'hidden'})
        self.assertEqual(result['gpu']['remaining_seconds'],97200)
        self.assertNotIn('hidden',json.dumps(result))
        self.assertIsNone(normalize_quota({'gpuQuota':{'totalTimeAllowed':'108000s'}})['gpu']['remaining_seconds'])

    def test_idle_releases_once_and_next_generation_can_run(self):
        with tempfile.TemporaryDirectory() as d:
            core=self.make_core(d);job=core.submit('fox',request_id='first')['job_id'];self.finish(core,job)
            self.assertEqual(core.status()['keep_alive_minutes'],15)
            self.assertGreater(core.status()['idle_remaining_seconds'],890)
            future=time.monotonic()+901
            with patch('studio_core.time.monotonic',return_value=future):core.expire_idle();core.expire_idle()
            self.assertEqual(core.backend.releases,1)
            self.assertEqual(core.status()['session_state'],'released')
            new=core.submit('teapot',request_id='second')['job_id']
            for _ in range(200):
                if core.store.get(new)['state']=='complete':break
                time.sleep(.02)
            self.assertEqual(core.store.get(new)['state'],'complete')

    def test_queue_and_running_job_prevent_release_even_after_timeout(self):
        with tempfile.TemporaryDirectory() as d:
            core=self.make_core(d);job=core.submit('fox',request_id='queued')['job_id']
            core.idle_since=time.monotonic()-10000;core.session_state='active'
            core.expire_idle();self.assertEqual(core.backend.releases,0)
            core.store.claim();core.expire_idle();self.assertEqual(core.backend.releases,0)
            self.assertIsNone(core.status()['idle_remaining_seconds'])

    def test_timeout_change_preserves_backend_deduplication_and_zero_disables(self):
        with tempfile.TemporaryDirectory() as d:
            core=self.make_core(d);backend=core.backend
            first=core.submit('fox',request_id='one')
            core.configure(dict(CONFIG,keep_alive_minutes=30))
            self.assertIs(core.backend,backend)
            self.assertEqual(core.submit('fox',request_id='one')['job_id'],first['job_id'])
            core.configure(dict(CONFIG,keep_alive_minutes=0))
            core.configure(CONFIG)  # An old client must not reset the shared policy.
            self.assertEqual(core.status()['keep_alive_minutes'],0)
            core.store.update(first['job_id'],state='complete')
            core.idle_since=time.monotonic()-10000;core.session_state='active';core.expire_idle()
            self.assertEqual(backend.releases,0)
            for value in [-1,1441,True,1.2,'invalid']:
                with self.assertRaises(ValueError):normalize_config(dict(CONFIG,keep_alive_minutes=value))

    def test_failed_release_is_visible_throttled_and_does_not_leak_errors(self):
        with tempfile.TemporaryDirectory() as d:
            core=self.make_core(d);core.idle_since=time.monotonic()-901;core.session_state='active';core.backend.fail_release=True
            core.expire_idle();core.expire_idle()
            self.assertEqual(core.backend.releases,1)
            self.assertEqual(core.status()['session_state'],'release_failed')
            self.assertNotIn('private transport detail',json.dumps(core.status()))

    def test_quota_reads_are_cached_and_do_not_extend_idle_deadline(self):
        with tempfile.TemporaryDirectory() as d:
            core=self.make_core(d);core.idle_since=time.monotonic()-100;core.session_state='active';before=core.idle_since
            self.assertTrue(core.quota()['available']);core.quota();core.status()
            self.assertEqual(core.backend.quota_calls,1);self.assertEqual(core.idle_since,before)

    def test_cancel_uses_exact_connected_session_and_failure_preserves_record(self):
        with tempfile.TemporaryDirectory() as d:
            backend=AccountBackend('owner','KGAT_fixture_secret',cache_dir=d);self.addCleanup(backend.close)
            cached={'jupyter_url':'https://kkb-production.jupyter-proxy.kaggle.net/k/42/fixture/proxy','operation':'operations/fixture'}
            backend._save(cached)
            with patch.object(backend,'_call',return_value={}) as api:
                backend.release_session()
                api.assert_called_once_with('kernels.KernelsApiService','CancelKernelSession',{'kernelSessionId':42})
            self.assertEqual(backend._read(),{})
            backend._save(cached)
            with patch.object(backend,'_call',return_value={'errorMessage':'failed'}):
                with self.assertRaises(RuntimeError):backend.release_session()
            self.assertEqual(backend._read(),cached)

if __name__=='__main__':unittest.main()
