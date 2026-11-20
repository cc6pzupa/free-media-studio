"""CPU-only Kaggle sessions for language models: own notebook, no machine shape, released with the account."""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import account_backend
from account_backend import AccountBackend

URL = 'https://kkb-production.jupyter-proxy.kaggle.net/k/1/abc/proxy'


class FakeLlm:
    def __init__(self, name):
        self.name = name
        self.info = None

    def prepare(self, name, context, choice, progress):
        self.info = {'name': name, 'where': self.name}
        return dict(self.info)

    def chat(self, body, on_chunk):
        on_chunk({'where': self.name})
        return {'seconds': 0}

    def close(self):
        pass


class CpuSessions(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.calls = []

    def backend(self, **options):
        backend = AccountBackend('owner', 'KGAT_x', cache_dir=self.temp.name, **options)
        calls = self.calls

        def call(service, method, body, timeout=None):
            calls.append((backend.slug, method, dict(body)))
            if method == 'CreateKernelSession':
                return {'name': 'operations/create-kernel-session/1'}
            if method == 'GetOperation':
                return {'done': True, 'response': {'jupyterUrl': URL, 'kernelSessionId': 7}}
            return {}
        backend._call = call
        backend._ensure_notebook = lambda: None
        backend._remember_operation = lambda operation: None
        return backend

    def test_a_cpu_session_has_no_machine_shape(self):
        gpu = self.backend()
        cpu = self.backend(notebook_slug='cc-media-cpu', machine='cpu')
        gpu._ensure_session(lambda event: None)
        cpu._ensure_session(lambda event: None)
        created = [(slug, body) for slug, method, body in self.calls if method == 'CreateKernelSession']
        self.assertEqual(created[0][1]['machineShape'], 'NvidiaTeslaT4')
        self.assertEqual(created[1][0], 'owner/cc-media-cpu')
        self.assertNotIn('machineShape', created[1][1])
        with self.assertRaises(ValueError):
            AccountBackend('owner', 'KGAT_x', cache_dir=self.temp.name, machine='tpu')

    def test_cpu_target_uses_its_own_session_and_is_released_with_the_account(self):
        gpu = self.backend()

        def cpu():
            if gpu.cpu is None:
                gpu.cpu = self.backend(notebook_slug='cc-media-cpu', machine='cpu')
                def llm(notify, backend=gpu.cpu):
                    backend.llm_backend = backend.llm_backend or FakeLlm('cpu')
                    return backend.llm_backend
                gpu.cpu._llm = llm
            return gpu.cpu
        gpu._cpu = cpu
        info = gpu.llm_prepare('Qwen3-30B-A3B', 8192, None, None, target='cpu')
        self.assertEqual(info['where'], 'cpu')
        self.assertEqual(info['target'], 'cpu')
        got = []
        gpu.llm_chat({'messages': [{'role': 'user', 'content': 'hi'}]}, got.append)
        self.assertEqual(got, [{'where': 'cpu'}])
        self.assertEqual(gpu.llm_info()['target'], 'cpu')
        gpu.cpu.release_session = released = mock.Mock()
        gpu.release_session()
        self.assertEqual(released.call_count, 1)
        self.assertEqual(gpu.llm_target, 'gpu')
        with self.assertRaises(ValueError):
            gpu.llm_prepare('x', target='tpu')


if __name__ == '__main__':
    unittest.main()
