"""The Kaggle-facing language-model layer, with a fake Jupyter transport."""
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import llm_session
import remote_llm
from llm_session import LlmSessionBackend, chat_body, llm_bootstrap_code, prepare_code, chat_code


class FakeTransport:
    """Answers each executed cell with scripted kernel output, streamed through the callback."""
    def __init__(self, backend, replies):
        self.backend = backend
        self.replies = replies
        self.cells = []

    def start(self):
        pass

    def execute(self, code, timeout=1200):
        self.cells.append(code)
        if 'CC_LLM_BOOT' in code:
            return 'CC_LLM_BOOT\n'
        output = self.replies.pop(0)
        if isinstance(output, Exception):
            raise output
        for line in output.splitlines(keepends=True):
            self.backend._output(line)
        return output

    def close(self):
        pass


def backend_with(replies):
    holder = {}
    backend = LlmSessionBackend('https://kkb-production.jupyter-proxy.kaggle.net/k/1/x/proxy',
                                transport_factory=lambda: holder['transport'])
    holder['transport'] = FakeTransport(backend, replies)
    return backend, holder['transport']


INFO = {'name': 'Qwen3-8B', 'tag': 'hf.co/Qwen/Qwen3-8B-GGUF:Q8_0', 'repo': 'Qwen/Qwen3-8B-GGUF', 'quant': 'Q8_0',
        'file': 'Qwen3-8B-Q8_0.gguf', 'bytes': 8709518112, 'placement': 'gpu', 'estimated_tok_per_s': 21.0, 'context': 8192}


class SessionBackend(unittest.TestCase):
    def test_prepare_reports_progress_and_returns_the_model(self):
        backend, transport = backend_with([
            'COMFY_PROGRESS:{"phase":"download","percent":40.0}\nCC_LLM_READY:' + json.dumps(INFO) + '\n'])
        events = []
        info = backend.prepare('Qwen3-8B', 8192, None, events.append)
        self.assertEqual(info['quant'], 'Q8_0')
        self.assertEqual(events, [{'phase': 'download', 'percent': 40.0}])
        self.assertIn('"name": "Qwen3-8B"', transport.cells[-1])

    def test_chat_streams_chunks(self):
        chunk = {'choices': [{'delta': {'content': 'Hi'}}]}
        backend, transport = backend_with(['CC_LLM_READY:' + json.dumps(INFO) + '\n',
                                           'CC_LLM_SSE:' + json.dumps(chunk) + '\nCC_LLM_DONE:{"seconds":0.5}\n'])
        backend.prepare('Qwen3-8B')
        got = []
        self.assertEqual(backend.chat({'messages': [{'role': 'user', 'content': 'hi'}]}, got.append), {'seconds': 0.5})
        self.assertEqual(got, [chunk])

    def test_errors(self):
        backend, _ = backend_with(['CC_LLM_ERROR:"No build of this model fits this session (2 GPU, 31 GB RAM)"\n'])
        with self.assertRaisesRegex(ValueError, 'No build'):
            backend.prepare('Llama-3.1-405B')
        with self.assertRaisesRegex(ValueError, '先选择'):
            backend.chat({'messages': [{'role': 'user', 'content': 'x'}]}, print)
        # Transport errors can contain connection URLs: never shown.
        backend, _ = backend_with([RuntimeError('wss://kkb-production.jupyter-proxy.kaggle.net/secret failed')])
        with self.assertRaises(RuntimeError) as caught:
            backend.prepare('Qwen3-8B')
        self.assertNotIn('kaggle.net', str(caught.exception))


class Validation(unittest.TestCase):
    def test_names_contexts_choices(self):
        self.assertEqual(llm_session.validate_name(' Qwen/Qwen3-8B-GGUF '), 'Qwen/Qwen3-8B-GGUF')
        for bad in ('', '../x', 'a b', 'x;rm', None, 'a' * 130):
            with self.assertRaises(ValueError):
                llm_session.validate_name(bad)
        self.assertEqual(llm_session.validate_context(None), 8192)
        with self.assertRaises(ValueError):
            llm_session.validate_context(9000)
        with self.assertRaises(ValueError):
            llm_session.validate_choice({'repo': 'x/y', 'first': '../../etc/passwd'})

    def test_chat_body_keeps_only_known_fields(self):
        body = chat_body({'model': 'x', 'messages': [{'role': 'user', 'content': [{'type': 'text', 'text': 'a'}, {'type': 'image_url'}]}],
                          'temperature': 0.5, 'max_tokens': 100, 'stop': 'END', 'tools': [{}], 'stream': True})
        self.assertEqual(body, {'messages': [{'role': 'user', 'content': 'a'}], 'temperature': 0.5, 'max_tokens': 100, 'stop': ['END']})
        for bad in ({'messages': []}, {'messages': [{'role': 'tool', 'content': 'x'}]}, {'messages': [{'role': 'user', 'content': 'x'}], 'temperature': 5},
                    {'messages': [{'role': 'user', 'content': 'x'}], 'max_tokens': True}):
            with self.assertRaises(ValueError):
                chat_body(bad)


class KernelCode(unittest.TestCase):
    def test_generated_cells_compile(self):
        for code in (llm_bootstrap_code(), prepare_code('Qwen3-8B', 8192, {'repo': 'a/b', 'first': 'c.gguf'}),
                     chat_code({'messages': [{'role': 'user', 'content': "quote ' and \"\"\" here"}]}), llm_session.status_code()):
            compile(code, '<cell>', 'exec')

    def test_image_and_video_switches_stop_ollama(self):
        from session_backend import model_switch_code
        from video_session import video_bootstrap_code
        self.assertIn("_CC_LLM'].stop()", model_switch_code('z-image-turbo'))
        self.assertIn("_CC_LLM'].stop()", video_bootstrap_code('ltx-video-2b-distilled'))

    def test_ollama_tags(self):
        self.assertEqual(remote_llm.ollama_tag({'first': 'Qwen3-8B-Q8_0.gguf', 'quant': 'Q8_0'}, 'Qwen/Qwen3-8B-GGUF'),
                         'hf.co/Qwen/Qwen3-8B-GGUF:Q8_0')
        self.assertEqual(remote_llm.ollama_tag({'first': 'Qwen3-8B-UD-Q4_K_XL.gguf', 'quant': 'Q4_K_XL'}, 'unsloth/Qwen3-8B-GGUF'),
                         'hf.co/unsloth/Qwen3-8B-GGUF:UD-Q4_K_XL')


if __name__ == '__main__':
    unittest.main()
