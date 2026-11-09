"""Local fixed-endpoint gateway; requires an already running Jupyter server."""
import base64
import hmac
import json
import os
import re
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from session_runtime import JupyterTransport, ModelSession
from session_backend import bootstrap_code, generation_code as session_generation_code


def generation_code(prompt, seed, steps):
    return session_generation_code(prompt, seed, steps)


def main():
    url = os.environ.get('KAGGLE_JUPYTER_URL', '').strip()
    key = os.environ.get('IMAGE_API_KEY', '').strip()
    dataset = os.environ.get('KAGGLE_DATASET', '')
    minutes = int(os.environ.get('MODEL_LIFETIME_MINUTES', '0'))
    if not url or not key or not re.fullmatch(r'[A-Za-z0-9_-]+/[A-Za-z0-9_-]+', dataset):
        raise SystemExit('Set KAGGLE_JUPYTER_URL, IMAGE_API_KEY and KAGGLE_DATASET before starting.')
    if minutes < 0:
        raise SystemExit('MODEL_LIFETIME_MINUTES must be zero or positive.')
    bootstrap = bootstrap_code(dataset)
    session = ModelSession(lambda: JupyterTransport(url), bootstrap, minutes * 60 if minutes else None)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass  # No prompts, credentials or remote URLs in HTTP access logs.

        def reply(self, status, data):
            raw = json.dumps(data, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Content-Length', str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_POST(self):
            if self.path != '/v1/images/generations':
                return self.reply(404, {'error': {'message': 'Unknown endpoint'}})
            if not hmac.compare_digest(self.headers.get('Authorization', ''), 'Bearer ' + key):
                return self.reply(401, {'error': {'message': 'Invalid API key'}})
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= 32768:
                    raise ValueError()
                data = json.loads(self.rfile.read(length))
                prompt = data.get('prompt', '')
                seed, steps = data.get('seed', 42), data.get('steps', 20)
                if not isinstance(prompt, str) or not 1 <= len(prompt.strip()) <= 4000:
                    raise ValueError()
                if type(seed) is not int or not 0 <= seed <= 2147483647:
                    raise ValueError()
                if type(steps) is not int or not 1 <= steps <= 50:
                    raise ValueError()
                if data.get('size', '512x512') != '512x512' or data.get('n', 1) != 1:
                    raise ValueError()
                if data.get('response_format', 'b64_json') != 'b64_json':
                    raise ValueError()
            except (ValueError, TypeError, AttributeError):
                return self.reply(400, {'error': {'message': 'Use prompt (1–4000 characters), size=512x512, n=1, response_format=b64_json, steps=1–50, seed=0–2147483647.'}})
            try:
                started = time.monotonic()
                output = session.execute(generation_code(prompt, seed, steps))
                matches = [line[13:] for line in output.splitlines() if line.startswith('COMFY_RESULT:')]
                if len(matches) != 1:
                    raise RuntimeError('Missing image')
                result = json.loads(matches[0])
                base64.b64decode(result['png'], validate=True)
                usage = dict(result['stats'], request_seconds=round(time.monotonic()-started, 2))
            except Exception:
                return self.reply(502, {'error': {'message': 'Remote generation failed. Check the Kaggle session and connection URL before retrying.'}})
            self.reply(200, {'created': int(time.time()), 'data': [{'b64_json': result['png']}], 'usage': usage})

    server = ThreadingHTTPServer(('127.0.0.1', int(os.environ.get('IMAGE_API_PORT', '8787'))), Handler)
    print('Image API listening on http://127.0.0.1:%s/v1/images/generations' % server.server_port, flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()
        session.close()


if __name__ == '__main__':
    main()
