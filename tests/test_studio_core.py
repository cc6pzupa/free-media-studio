import concurrent.futures
import importlib.util
import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from studio_core import JobStore,StudioCore,configuration_fingerprint,normalize_config
spec=importlib.util.spec_from_file_location('fake_worker_fixture',Path(__file__).with_name('fixtures')/'fake_worker.py')
fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)
CONFIG={'mode':'account','user':'owner','token':'KGAT_fixture_secret'}


def wait(core,job_id):
    deadline=time.monotonic()+8
    while time.monotonic()<deadline:
        result=core.store.get(job_id)
        if result['state'] not in ('queued','running'):return result
        time.sleep(.02)
    raise AssertionError('Job timed out')


class CoreTest(unittest.TestCase):
    def test_failed_configuration_save_keeps_previous_backend_and_account(self):
        with tempfile.TemporaryDirectory() as d:
            core=self.make_core(d);previous=core.backend
            try:
                with patch('studio_core.private_write',side_effect=OSError('disk full')):
                    with self.assertRaises(OSError):
                        core.configure(dict(CONFIG,user='another'),remember=True,api_key='local-api-key')
                self.assertIs(core.backend,previous)
                self.assertEqual(core.status()['user'],'owner')
            finally:core.close()

    def make_core(self,directory):
        with patch('studio_core.load_config',return_value=None):core=StudioCore(directory,backend_factory=fixture.FakeBackend)
        core.configure(CONFIG)
        return core

    def test_concurrent_duplicate_submission_runs_once_and_keeps_model_backend(self):
        with tempfile.TemporaryDirectory() as d:
            core=self.make_core(d)
            try:
                with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
                    ids=list(pool.map(lambda _:core.submit('fox',request_id='one')['job_id'],range(8)))
                self.assertEqual(len(set(ids)),1)
                core.start();first=wait(core,ids[0]);second=wait(core,core.submit('teapot',request_id='two')['job_id'])
                self.assertEqual(core.backend.count,2)
                self.assertEqual(first['usage']['generation_number'],1)
                self.assertTrue(second['usage']['model_reused'])
                self.assertEqual(second['usage']['model_load_count'],1)
                with self.assertRaisesRegex(ValueError,'不同参数'):core.submit('different',request_id='one')
            finally:core.close()

    def test_crash_marks_running_interrupted_but_recovers_queued_jobs(self):
        with tempfile.TemporaryDirectory() as d:
            core=self.make_core(d)
            interrupted=core.submit('old',request_id='old')['job_id']
            queued=core.submit('new',request_id='new')['job_id']
            core.store.claim();core.close()
            with patch('studio_core.load_config',return_value=normalize_config(CONFIG)):
                replacement=StudioCore(d,backend_factory=fixture.FakeBackend)
            try:
                self.assertEqual(replacement.store.get(interrupted)['state'],'interrupted')
                replacement.start();result=wait(replacement,queued)
                self.assertEqual(result['state'],'complete')
                self.assertEqual(replacement.backend.count,1)
                self.assertEqual(replacement.store.get(interrupted)['state'],'interrupted')
            finally:replacement.close()

    def test_progress_errors_and_disk_records_never_include_credentials(self):
        with tempfile.TemporaryDirectory() as d:
            core=self.make_core(d)
            try:
                core.start();job=core.submit('fox',request_id='ok')['job_id']
                deadline=time.monotonic()+3;seen=None
                while time.monotonic()<deadline:
                    seen=core.store.get(job)
                    if seen.get('percent')==25:break
                    time.sleep(.02)
                self.assertEqual(seen['downloaded_bytes'],1024)
                self.assertNotIn(CONFIG['token'],json.dumps(seen))
                done=wait(core,job);self.assertEqual(done['state'],'complete')
                self.assertNotIn('token',done['usage'])
                failed=wait(core,core.submit('fail',request_id='failed')['job_id'])
                self.assertEqual(failed['state'],'error')
                self.assertNotIn(CONFIG['token'],json.dumps(failed))
                dump='\n'.join(core.store.db.iterdump())
                self.assertNotIn(CONFIG['token'],dump)
            finally:core.close()

    def test_images_survive_worker_restart_and_invalid_ids_cannot_traverse(self):
        with tempfile.TemporaryDirectory() as d:
            core=self.make_core(d);core.start()
            job=core.submit('fox',request_id='persist')['job_id'];wait(core,job)
            original=core.store.image(job);core.close()
            with patch('studio_core.load_config',return_value=None):replacement=StudioCore(d,backend_factory=fixture.FakeBackend)
            try:
                self.assertEqual(replacement.store.image(job),original)
                self.assertFalse(replacement.status()['configured'])
                for job_id in ['../settings.json','a'*32+'?token=secret','',None]:
                    with self.subTest(job_id=job_id),self.assertRaises(ValueError):replacement.store.get(job_id)
            finally:replacement.close()

    def test_configuration_cannot_change_account_while_jobs_are_pending(self):
        with tempfile.TemporaryDirectory() as d:
            core=self.make_core(d)
            try:
                core.submit('fox',request_id='pending')
                core.configure(CONFIG)
                with self.assertRaisesRegex(ValueError,'队列'):core.configure(dict(CONFIG,user='another'))
                self.assertEqual(core.config['user'],'owner')
                self.assertNotIn(CONFIG['token'],json.dumps(core.status()))
            finally:core.close()


if __name__=='__main__':unittest.main()
