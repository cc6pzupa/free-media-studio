import ast, json, os, re, subprocess, sys, tempfile, threading, time, unittest, uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import requests
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from batch_backend import BatchBackend
from easy_gateway import create_server
from session_runtime import ModelSession

class RegressionTest(unittest.TestCase):
    def test_legacy_prompt_preserves_backslashes_and_newlines(self):
        module = ast.parse((ROOT / 'node.py').read_text())
        node_class = next(n for n in module.body if isinstance(n, ast.ClassDef))
        scope = dict(json=json, os=os, re=re, subprocess=subprocess, tempfile=tempfile,
                     time=time, uuid=uuid, Path=Path, __file__=str(ROOT / 'node.py'))
        exec(compile(ast.Module(body=[node_class], type_ignores=[]), 'node.py', 'exec'), scope)
        for prompt in [r'C:\new\cat', 'cat\\test\nsecond line', r'cat\1', "cat'; print('text') #"]:
            captured = {}
            def stop_after_render(args, **kwargs):
                nb = json.loads((Path(args[4]) / 'job.ipynb').read_text())
                assignment = next(n for n in ast.parse(''.join(nb['cells'][3]['source'])).body
                                  if isinstance(n, ast.Assign) and getattr(n.targets[0], 'id', '') == 'PROMPT')
                captured['prompt'] = ast.literal_eval(assignment.value)
                raise RuntimeError('stop before real submission')
            with self.subTest(prompt=prompt), patch.object(subprocess, 'run', stop_after_render):
                with self.assertRaisesRegex(RuntimeError, 'stop before'):
                    scope['KaggleQwen']().generate(prompt, 'owner', 'job', 'assets', 42, 20, 5)
                self.assertEqual(captured['prompt'], prompt)

    def test_completed_job_without_image_fails_immediately(self):
        calls = []
        def run(args, **kwargs):
            calls.append(args[2])
            if args[2] == 'output': Path(args[5]).mkdir()
            return SimpleNamespace(returncode=0, stdout='complete' if args[2] == 'status' else '')
        with self.assertRaisesRegex(RuntimeError, '没有找到本次结果图片'):
            BatchBackend('owner', 'owner/assets', 'token', runner=run, poll_seconds=0).generate('cat')
        self.assertEqual(calls, ['push', 'status', 'output'])

    def test_failed_session_is_discarded_without_retry(self):
        transports = []
        class Transport:
            def __init__(self): self.closed = False; self.requests = []
            def start(self): pass
            def execute(self, code, timeout):
                self.requests.append(code)
                if code == 'fail': raise RuntimeError('original error')
                return code
            def close(self):
                self.closed = True
                raise OSError('cleanup failed')
        def factory():
            t = Transport(); transports.append(t); return t
        session = ModelSession(factory, 'load', 600)
        try:
            with self.assertRaisesRegex(RuntimeError, 'original error'): session.execute('fail')
            self.assertIsNone(session.transport)
            self.assertTrue(transports[0].closed)
            self.assertEqual(transports[0].requests, ['load', 'fail'])
            self.assertEqual(session.execute('next'), 'next')
            self.assertEqual(len(transports), 2)
        finally:
            try: session.close()
            except OSError: pass

    def test_reload_finds_running_job_and_can_retrieve_result(self):
        started, finish = threading.Event(), threading.Event()
        class Backend:
            def __init__(self, user, dataset, token): self.user, self.dataset, self.token = user, dataset, token
            def generate(self, *args):
                started.set(); finish.wait(5)
                return b'image', {'request_seconds': 1}
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, {'EASY_CONFIG_PATH': folder + '/settings.json'}):
            server = create_server(0, Backend)
            thread = threading.Thread(target=server.serve_forever); thread.start()
            url = f'http://127.0.0.1:{server.server_port}'
            try:
                nonce = re.search("const nonce='([^']+)'", requests.get(url, timeout=5).text).group(1)
                headers = {'X-Local-Session': nonce}
                requests.post(url + '/configure', headers=headers, json={'user': 'owner', 'dataset': 'owner/assets', 'token': 'token'}, timeout=5).raise_for_status()
                job = requests.post(url + '/jobs', headers=headers, json={'prompt': 'cat'}, timeout=5).json()['job_id']
                self.assertTrue(started.wait(2))
                cfg = requests.get(url + '/configuration', headers=headers, timeout=5).json()
                self.assertEqual(cfg['active_job_id'], job)
                duplicate = requests.post(url + '/jobs', headers=headers, json={'prompt': 'cat'}, timeout=5)
                self.assertEqual(duplicate.status_code, 409)
                finish.set()
                for _ in range(100):
                    state = requests.get(url + '/jobs/' + job, headers=headers, timeout=5).json()
                    if state['state'] == 'complete': break
                    time.sleep(.01)
                self.assertEqual(state['state'], 'complete')
                self.assertIsNone(requests.get(url + '/configuration', headers=headers, timeout=5).json()['active_job_id'])
            finally:
                finish.set(); server.shutdown(); thread.join(); server.server_close()

if __name__ == '__main__': unittest.main()
