"""Connecting local AI tools to the Kaggle model: additive edits with backups."""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import harness

BASE = 'http://127.0.0.1:8787/v1'
KEY = 'a' * 43


class Tools(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.home = Path(self.temp.name)
        patcher = mock.patch.dict(os.environ, {'CC_MEDIA_HOME': str(self.home), 'XDG_CONFIG_HOME': str(self.home / '.config')})
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_opencode_adds_a_provider_and_keeps_the_rest(self):
        path = self.home / '.config' / 'opencode' / 'opencode.json'
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({'theme': 'tokyo', 'provider': {'other': {'name': 'Other'}}}))
        result = harness.configure('opencode', BASE, KEY, 'Qwen3-8B')
        config = json.loads(path.read_text())
        self.assertEqual(config['theme'], 'tokyo')
        self.assertIn('other', config['provider'])
        ours = config['provider']['cc-media']
        self.assertEqual(ours['npm'], '@ai-sdk/openai-compatible')
        self.assertEqual(ours['options'], {'baseURL': BASE, 'apiKey': KEY})
        self.assertIn('Qwen3-8B', ours['models'])
        self.assertTrue(Path(result['backup']).exists())
        harness.configure('opencode', BASE, KEY, 'Qwen3-32B')  # a second model joins the first
        self.assertEqual(sorted(json.loads(path.read_text())['provider']['cc-media']['models']), ['Qwen3-32B', 'Qwen3-8B'])
        self.assertTrue(harness.opencode_state()['configured'])

    def test_opencode_with_comments_is_left_alone(self):
        path = self.home / '.config' / 'opencode' / 'opencode.jsonc'
        path.parent.mkdir(parents=True)
        path.write_text('{\n  // mine\n}\n')
        with self.assertRaisesRegex(ValueError, '手动'):
            harness.configure('opencode', BASE, KEY, 'Qwen3-8B')
        self.assertEqual(path.read_text(), '{\n  // mine\n}\n')

    def test_aider_gets_its_own_config_file(self):
        result = harness.configure('aider', BASE, KEY, 'Qwen3-8B')
        text = (self.home / '.aider.cc-media.conf.yml').read_text()
        self.assertIn('openai-api-base: ' + BASE, text)
        self.assertIn('model: openai/Qwen3-8B', text)
        self.assertIn('--config', result['run'])
        self.assertFalse((self.home / '.aider.conf.yml').exists())

    def test_continue_new_and_existing_configs(self):
        path = self.home / '.continue' / 'config.yaml'
        harness.configure('continue', BASE, KEY, 'Qwen3-8B')
        self.assertIn('schema: v1', path.read_text())
        self.assertIn('    apiBase: ' + BASE, path.read_text())
        path.write_text('name: Mine\nversion: 1.0.0\nschema: v1\nmodels:\n    - name: GPT\n      provider: openai\n      model: gpt-4o\nrules:\n  - be nice\n')
        harness.configure('continue', BASE, KEY, 'Qwen3-8B')
        text = path.read_text()
        # Inserted with the list's own indentation, before the existing entry; the rest unchanged.
        self.assertIn('models:\n    - name: Qwen3-8B (Kaggle)\n      provider: openai\n', text)
        self.assertIn('    - name: GPT\n', text)
        self.assertTrue(text.endswith('rules:\n  - be nice\n'))
        self.assertTrue((path.parent / 'config.yaml.bak-ccmedia').exists())
        path.write_text('name: Mine\nversion: 1.0.0\nschema: v1\nmodels: []\n')
        with self.assertRaisesRegex(ValueError, '手动'):
            harness.configure('continue', BASE, KEY, 'Qwen3-8B')

    def test_inputs_are_checked(self):
        for args in (('codex', BASE, KEY, 'x'), ('aider', 'http://evil.example/v1', KEY, 'x'),
                     ('aider', BASE, 'short', 'x'), ('aider', BASE, KEY, '../x')):
            with self.assertRaises(ValueError):
                harness.configure(*args)
        with mock.patch.object(harness, '_which', return_value=None), mock.patch.object(harness.shutil, 'which', return_value=None):
            with self.assertRaises(ValueError):
                harness.install_command('opencode')

    def test_discover_lists_tools_without_commands(self):
        with mock.patch.object(harness, '_which', return_value=None):
            listing = harness.discover()
        ids = [t['id'] for t in listing]
        self.assertEqual(ids, ['opencode', 'aider', 'continue', 'open-webui', 'cherry-studio', 'chatbox', 'codex'])
        self.assertTrue(all('install' not in t for t in listing))
        self.assertIn('Responses', next(t for t in listing if t['id'] == 'codex')['unsupported'])


if __name__ == '__main__':
    unittest.main()
