"""Several Kaggle accounts: saved profiles, switching without ending sessions."""
import json
import os
import struct
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import studio_core
from studio_core import StudioCore, config_path, load_config


def png(width=512, height=512):
    return b'\x89PNG\r\n\x1a\n' + b'\x00\x00\x00\rIHDR' + struct.pack('>II', width, height) + b'\x00' * 32


class FakeAccount:
    """An automatic-account backend: it holds a session until release_session()."""
    instances = []

    def __init__(self, config):
        self.user = config['user']
        self.released = 0
        self.closed = 0
        self.generated = []
        FakeAccount.instances.append(self)

    def generate_with_progress(self, prompt, seed=42, steps=None, progress=None, **options):
        self.generated.append(prompt)
        return png(), {'model_id': 'qwen-image-2.1'}

    def release_session(self):
        self.released += 1

    def close(self):
        self.closed += 1


def account(user, keep_alive=15):
    return {'mode': 'account', 'user': user, 'token': 'token-' + user, 'keep_alive_minutes': keep_alive}


class Accounts(unittest.TestCase):
    def setUp(self):
        FakeAccount.instances = []
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        patcher = mock.patch.dict(os.environ, {'EASY_CONFIG_PATH': str(Path(self.temp.name) / 'settings.json')})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.core = StudioCore(Path(self.temp.name) / 'studio', backend_factory=FakeAccount)
        self.addCleanup(self.core.close)

    def by_user(self, user):
        return [b for b in FakeAccount.instances if b.user == user]

    def test_profiles_are_saved_and_the_active_one_loads(self):
        self.core.configure(account('alice'), remember=True, api_key='key')
        self.core.configure(account('bob'), remember=True, api_key='key')
        saved = json.loads(config_path().read_text())
        self.assertEqual(sorted(saved['profiles']), ['alice', 'bob'])
        self.assertEqual(saved['active'], 'bob')
        self.assertEqual(saved['user'], 'bob')  # older versions read the top level
        self.assertEqual(load_config()['user'], 'bob')
        listing = self.core.accounts()['accounts']
        self.assertEqual([(a['user'], a['active']) for a in listing], [('bob', True), ('alice', False)])
        self.assertNotIn('token', json.dumps(listing))

    def test_an_older_single_account_file_is_one_profile(self):
        config_path().parent.mkdir(parents=True, exist_ok=True)
        config_path().write_text(json.dumps(dict(account('carol'), api_key='key')))
        self.assertEqual(list(studio_core.saved_profiles()), ['carol'])
        self.assertEqual(load_config()['user'], 'carol')

    def test_switching_keeps_the_previous_session_and_switching_back_reuses_it(self):
        self.core.configure(account('alice'), remember=True, api_key='key')
        self.core.configure(account('bob'), remember=True, api_key='key')
        self.core.switch_account('alice')
        alice = self.core.backend
        with self.core.lock:
            self.core.session_state = 'active'; self.core.idle_since = time.monotonic()
        self.core.switch_account('bob')
        self.assertEqual(alice.closed, 0)
        self.assertEqual(alice.released, 0)
        rows = {a['user']: a for a in self.core.accounts()['accounts']}
        self.assertEqual(rows['alice']['session_state'], 'active')
        self.assertIsNotNone(rows['alice']['idle_remaining_seconds'])
        self.core.switch_account('alice')
        self.assertIs(self.core.backend, alice)
        self.assertEqual(self.core.session_state, 'active')
        self.assertEqual(json.loads(config_path().read_text())['active'], 'alice')

    def test_a_queued_job_runs_on_the_account_it_was_submitted_to(self):
        self.core.configure(account('alice'))
        job = self.core.submit('lighthouse', request_id='r1')
        self.core.configure(account('bob'))
        self.core.start()
        deadline = time.time() + 10
        while self.core.store.get(job['job_id'])['state'] != 'complete' and time.time() < deadline:
            time.sleep(0.05)
        self.assertEqual(self.core.store.get(job['job_id'])['state'], 'complete')
        self.assertEqual(self.by_user('alice')[0].generated, ['lighthouse'])
        self.assertEqual(self.by_user('bob')[0].generated, [])
        rows = {a['user']: a for a in self.core.accounts()['accounts']}
        self.assertEqual(rows['alice']['session_state'], 'active')

    def test_parked_sessions_are_released_by_their_own_idle_timer(self):
        self.core.configure(account('alice', keep_alive=1))
        alice = self.core.backend
        with self.core.lock:
            self.core.session_state = 'active'; self.core.idle_since = time.monotonic()
        self.core.configure(account('bob'))
        self.core._expire_parked()
        self.assertEqual(alice.released, 0)
        with self.core.lock:
            entry = next(iter(self.core.parked.values()))
            entry['idle_since'] -= 61
        self.core._expire_parked()
        self.assertEqual(alice.released, 1)
        self.assertEqual(alice.closed, 1)
        self.assertEqual(self.core.parked, {})

    def test_release_now_and_forget(self):
        self.core.configure(account('alice'), remember=True, api_key='key')
        alice = self.core.backend
        with self.core.lock:
            self.core.session_state = 'active'; self.core.idle_since = time.monotonic()
        self.core.configure(account('bob'), remember=True, api_key='key')
        self.core.release_account('alice')
        self.assertEqual(alice.released, 1)
        self.assertEqual(self.core.parked, {})
        with self.core.lock:
            self.core.session_state = 'active'; self.core.idle_since = time.monotonic()
        self.core.release_account('bob')
        self.assertEqual(self.core.session_state, 'released')
        self.core.forget_account('alice')
        self.assertEqual(list(studio_core.saved_profiles()), ['bob'])
        with self.assertRaises(ValueError):
            self.core.switch_account('alice')
        with self.assertRaises(ValueError):
            self.core.release_account('nobody')

    def test_not_remembering_removes_only_that_account(self):
        self.core.configure(account('alice'), remember=True, api_key='key')
        self.core.configure(account('bob'), remember=False)
        self.assertEqual(list(studio_core.saved_profiles()), ['alice'])
        self.core.configure(account('alice'), remember=False)
        self.assertFalse(config_path().exists())


if __name__ == '__main__':
    unittest.main()
