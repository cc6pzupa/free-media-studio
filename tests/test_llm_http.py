"""Language-model routes end to end on this machine: gateway -> client -> worker -> backend.

The Kaggle side is replaced by a fake backend; everything local is real.
"""
import http.client
import json
import os
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import easy_gateway
import studio_client
import studio_worker
from llm_session import chat_body

KEY = 'k' * 43


class FakeLlmBackend:
    """Stands in for AccountBackend: prepares a model and streams OpenAI chunks."""
    def __init__(self):
        self.info = None
        self.prepared = []
        self.chats = []

    def llm_prepare(self, name, context=8192, choice=None, progress=None, target='gpu'):
        self.target = target
        progress({'phase': 'download', 'percent': 50.0, 'message': 'downloading'})
        self.prepared.append((name, context, choice))
        self.info = {'name': name, 'tag': 'hf.co/x/%s-GGUF:Q4_K_M' % name, 'repo': 'x/%s-GGUF' % name, 'quant': 'Q4_K_M',
                     'file': name + '-Q4_K_M.gguf', 'bytes': 5 * 10**9, 'placement': 'gpu', 'estimated_tok_per_s': 37.0,
                     'context': context, 'alternatives': []}
        return self.info

    def llm_chat(self, body, on_chunk):
        if not self.info:
            raise ValueError('请先选择并加载语言模型。')
        body = chat_body(body)
        self.chats.append(body)
        if body.get('tools'):
            # A tool call streamed in pieces, as Ollama sends it.
            for piece in ({'index': 0, 'id': 'call_9', 'type': 'function', 'function': {'name': 'get_weather', 'arguments': ''}},
                          {'index': 0, 'function': {'arguments': '{"city": '}}, {'index': 0, 'function': {'arguments': '"Oslo"}'}}):
                on_chunk({'id': 'c2', 'choices': [{'index': 0, 'delta': {'tool_calls': [piece]}, 'finish_reason': None}]})
            on_chunk({'id': 'c2', 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'tool_calls'}]})
            return {'seconds': 0.1}
        for piece in ('<think>hm</think>', 'Hello', ' there'):
            on_chunk({'id': 'c1', 'object': 'chat.completion.chunk', 'choices': [{'index': 0, 'delta': {'content': piece}, 'finish_reason': None}]})
        on_chunk({'id': 'c1', 'object': 'chat.completion.chunk', 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}],
                  'usage': {'prompt_tokens': 9, 'completion_tokens': 3, 'total_tokens': 12}})
        return {'seconds': 0.1}

    def llm_info(self):
        return self.info

    def close(self):
        pass


class LlmHttpTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.env = mock.patch.dict(os.environ, {'EASY_CONFIG_PATH': str(Path(cls.temp.name) / 'settings.json'),
                                               'CC_MEDIA_HOME': str(Path(cls.temp.name) / 'home'),
                                               'XDG_CONFIG_HOME': str(Path(cls.temp.name) / 'home' / '.config')})
        cls.env.start()
        cls.fake = FakeLlmBackend()
        ready = threading.Event()
        state = {}

        def on_ready(core, server):
            state.update(core=core, server=server)
            ready.set()
        cls.worker = threading.Thread(target=studio_worker.serve, kwargs=dict(
            directory=Path(cls.temp.name) / 'studio', backend_factory=lambda config: cls.fake, on_ready=on_ready), daemon=True)
        cls.worker.start()
        assert ready.wait(20)
        cls.core = state['core']
        cls.worker_server = state['server']
        cls.core.configure({'mode': 'account', 'user': 'owner', 'token': 'token'})
        studio_client.worker_starter = lambda: None
        with mock.patch.object(easy_gateway.secrets, 'token_urlsafe', return_value=KEY):
            cls.gateway = easy_gateway.create_server(0)
        threading.Thread(target=cls.gateway.serve_forever, daemon=True).start()
        cls.port = cls.gateway.server_port

    @classmethod
    def tearDownClass(cls):
        cls.gateway.shutdown()
        cls.gateway.server_close()
        cls.worker_server.shutdown()
        cls.worker.join(10)
        studio_client.worker_starter = None
        cls.env.stop()
        cls.temp.cleanup()

    def request(self, method, path, body=None, headers=None):
        connection = http.client.HTTPConnection('127.0.0.1', self.port, timeout=30)
        payload = json.dumps(body).encode() if body is not None else None
        connection.request(method, path, payload, dict({'Content-Type': 'application/json'}, **(headers or {})))
        response = connection.getresponse()
        data = response.read()
        connection.close()
        return response.status, response.getheader('Content-Type'), data

    def ui(self):
        return {'X-Local-Session': KEY}

    def api(self):
        return {'Authorization': 'Bearer ' + KEY}

    def test_1_gui_prepares_and_streams(self):
        status, _, page = self.request('GET', '/chat')
        self.assertEqual(status, 200)
        self.assertIn(b'CHAT PREVIEW', page)
        self.assertNotIn(b'__NONCE__', page)
        status, kind, data = self.request('POST', '/llm/prepare', {'name': 'Qwen3-8B', 'context': 8192}, self.ui())
        self.assertEqual(status, 200)
        self.assertIn('ndjson', kind)
        events = [json.loads(line) for line in data.splitlines()]
        self.assertEqual([e['type'] for e in events], ['progress', 'done'])
        self.assertEqual(events[0]['event']['percent'], 50.0)
        self.assertEqual(events[-1]['model']['quant'], 'Q4_K_M')
        status, _, data = self.request('GET', '/llm/status', headers=self.ui())
        self.assertEqual(json.loads(data)['model']['name'], 'Qwen3-8B')
        status, _, data = self.request('POST', '/llm/chat', {'messages': [{'role': 'user', 'content': 'hi'}], 'temperature': 0.2}, self.ui())
        events = [json.loads(line) for line in data.splitlines()]
        self.assertEqual(events[-1]['type'], 'done')
        text = ''.join(c['delta'].get('content', '') for e in events if e['type'] == 'chunk' for c in e['chunk']['choices'])
        self.assertEqual(text, '<think>hm</think>Hello there')
        self.assertEqual(self.fake.chats[-1], {'messages': [{'role': 'user', 'content': 'hi'}], 'temperature': 0.2})

    def test_2_openai_api(self):
        status, _, data = self.request('GET', '/v1/models', headers=self.api())
        self.assertEqual(status, 200)
        self.assertEqual([m['id'] for m in json.loads(data)['data']], ['Qwen3-8B'])
        # Non-streaming: one chat.completion with the reasoning kept apart by the model's own tags.
        status, _, data = self.request('POST', '/v1/chat/completions',
                                       {'model': 'default', 'messages': [{'role': 'user', 'content': 'hi'}]}, self.api())
        reply = json.loads(data)
        self.assertEqual(status, 200)
        self.assertEqual(reply['object'], 'chat.completion')
        self.assertEqual(reply['model'], 'Qwen3-8B')
        self.assertEqual(reply['choices'][0]['message']['content'], '<think>hm</think>Hello there')
        self.assertEqual(reply['choices'][0]['finish_reason'], 'stop')
        self.assertEqual(reply['usage']['completion_tokens'], 3)
        # Streaming SSE ends with [DONE].
        status, kind, data = self.request('POST', '/v1/chat/completions',
                                          {'model': 'Qwen3-8B', 'stream': True, 'messages': [{'role': 'user', 'content': 'hi'}]}, self.api())
        self.assertIn('text/event-stream', kind)
        lines = [l for l in data.decode().split('\n\n') if l.startswith('data: ')]
        self.assertEqual(lines[-1], 'data: [DONE]')
        chunks = [json.loads(l[6:]) for l in lines[:-1]]
        self.assertTrue(all(c['model'] == 'Qwen3-8B' for c in chunks))
        self.assertEqual(len(self.fake.prepared), 1)  # the loaded model was reused

    def test_7_tool_calls_without_streaming(self):
        tools = [{'type': 'function', 'function': {'name': 'get_weather', 'parameters': {'type': 'object', 'properties': {}}}}]
        status, _, data = self.request('POST', '/v1/chat/completions',
                                       {'model': 'default', 'messages': [{'role': 'user', 'content': 'Oslo?'}], 'tools': tools}, self.api())
        reply = json.loads(data)['choices'][0]
        self.assertEqual(status, 200)
        self.assertEqual(reply['finish_reason'], 'tool_calls')
        self.assertEqual(reply['message']['tool_calls'], [{'id': 'call_9', 'type': 'function',
                                                           'function': {'name': 'get_weather', 'arguments': '{"city": "Oslo"}'}}])
        self.assertEqual(self.fake.chats[-1]['tools'][0]['function']['name'], 'get_weather')

    def test_3_a_new_model_name_loads_that_model(self):
        status, _, data = self.request('POST', '/v1/chat/completions',
                                       {'model': 'Qwen3-14B', 'messages': [{'role': 'user', 'content': 'hi'}]}, self.api())
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(data)['model'], 'Qwen3-14B')
        self.assertEqual(self.fake.prepared[-1][0], 'Qwen3-14B')

    def test_4_auth_and_validation(self):
        status, _, _ = self.request('POST', '/v1/chat/completions', {'messages': [{'role': 'user', 'content': 'x'}]}, {'Authorization': 'Bearer wrong'})
        self.assertEqual(status, 401)
        status, _, _ = self.request('POST', '/llm/chat', {'messages': []}, {'X-Local-Session': 'wrong'})
        self.assertEqual(status, 401)
        status, _, data = self.request('POST', '/v1/chat/completions', {'model': 'default', 'messages': [{'role': 'robot', 'content': 'x'}]}, self.api())
        self.assertEqual(status, 400)
        self.assertIn('role', json.loads(data)['error']['message'])
        status, _, data = self.request('POST', '/llm/prepare', {'name': '../etc/passwd'}, self.ui())
        self.assertEqual(json.loads(data.splitlines()[-1])['type'], 'error')

    def test_6_local_tools(self):
        status, _, data = self.request('GET', '/harness', headers=self.ui())
        listing = json.loads(data)
        self.assertEqual(status, 200)
        self.assertEqual(listing['base_url'], 'http://127.0.0.1:%d/v1' % self.port)
        self.assertIn('aider', [t['id'] for t in listing['tools']])
        status, _, data = self.request('POST', '/harness/configure', {'id': 'aider', 'model': 'Qwen3-8B'}, self.ui())
        self.assertEqual(status, 200, data)
        written = (Path(self.temp.name) / 'home' / '.aider.cc-media.conf.yml').read_text()
        self.assertIn('openai-api-key: ' + KEY, written)
        self.assertIn('openai-api-base: http://127.0.0.1:%d/v1' % self.port, written)
        status, _, _ = self.request('POST', '/harness/configure', {'id': 'aider', 'model': 'x'}, {'X-Local-Session': 'wrong'})
        self.assertEqual(status, 401)
        status, _, data = self.request('POST', '/harness/install', {'id': 'continue'}, self.ui())
        self.assertEqual(status, 400)

    def test_5_chat_keeps_the_session_from_idle_release(self):
        before = self.core.idle_since
        self.request('POST', '/llm/chat', {'messages': [{'role': 'user', 'content': 'hi'}]}, self.ui())
        self.assertEqual(self.core.session_state, 'active')
        self.assertGreater(self.core.idle_since, before or 0)
        self.assertEqual(self.core.llm_active, 0)
        self.core.llm_active = 1
        try:
            self.assertIsNone(self.core.idle_remaining())
        finally:
            self.core.llm_active = 0


if __name__ == '__main__':
    unittest.main()
