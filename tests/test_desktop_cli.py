"""Exercise the same entry point documented for a clean clone installation."""
import json
import os
import queue
import re
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]


class DesktopCLITest(unittest.TestCase):
    def test_installed_client_starts_without_account_gpu_or_native_gui(self):
        with tempfile.TemporaryDirectory() as temporary:
            environment = dict(os.environ, EASY_CONFIG_PATH=str(Path(temporary)/'settings.json'),
                               PYTHONUTF8='1', PYTHONUNBUFFERED='1')
            process = subprocess.Popen([sys.executable, 'desktop.py', '--no-open', '--port', '0'],
                cwd=ROOT, env=environment, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding='utf-8')
            lines = queue.Queue()
            def collect_lines():
                for line in process.stdout:
                    lines.put(line)
            reader = threading.Thread(target=collect_lines, daemon=True)
            reader.start()
            try:
                deadline = time.monotonic()+20
                address = None
                observed = []
                while time.monotonic() < deadline:
                    try:line = lines.get(timeout=.2)
                    except queue.Empty:
                        if process.poll() is not None:break
                        continue
                    observed.append(line)
                    match = re.match(r'CC Media Studio: (http://127\.0\.0\.1:\d+)',line)
                    if match:
                        address = match.group(1)
                        break
                self.assertIsNotNone(address, ''.join(observed))
                health = requests.get(address+'/health', timeout=5)
                self.assertEqual(health.status_code,200)
                self.assertEqual(health.json()['app'],'qwen-studio')
                page = requests.get(address, timeout=5)
                self.assertIn('CC Media Studio',page.text)
                nonce = re.search("const nonce='([^']+)'",page.text).group(1)
                configuration = requests.get(address+'/configuration',
                    headers={'X-Local-Session':nonce},timeout=5).json()
                self.assertFalse(configuration['configured'])
                notebook = requests.get(address+'/kaggle_standalone.ipynb',timeout=5)
                self.assertEqual(notebook.status_code,200)
                self.assertEqual(json.loads(notebook.content)['nbformat'],4)
                duplicate = subprocess.run([sys.executable,'desktop.py','--no-open','--port',address.rsplit(':',1)[1]],
                    cwd=ROOT,env=environment,capture_output=True,text=True,encoding='utf-8',timeout=10)
                self.assertEqual(duplicate.returncode,0,duplicate.stderr)
                self.assertIn(address,duplicate.stdout)
            finally:
                process.terminate()
                try:process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill();process.wait(timeout=5)
                reader.join(timeout=2)
                process.stdout.close()


if __name__ == '__main__':
    unittest.main()
