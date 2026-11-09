"""Jupyter transport and fixed-lifetime model sessions. No Kaggle provisioning is assumed."""
import base64
import io
import json
import struct
import threading
import time
import uuid
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode, quote

import requests
import websocket


class RemoteError(RuntimeError):
    pass


def decode_message(raw):
    """Decode Jupyter's legacy JSON+binary-buffer websocket envelope."""
    if isinstance(raw, bytes):
        if len(raw) < 8:
            raise RemoteError('Invalid binary Jupyter message')
        count = struct.unpack_from('!I', raw)[0]
        if not 1 <= count <= (len(raw)-4)//4:
            raise RemoteError('Invalid Jupyter buffer count')
        offsets = struct.unpack_from('!'+'I'*count, raw, 4)
        if offsets[0] != 4*(count+1) or any(a>b for a,b in zip(offsets,offsets[1:])) or offsets[-1]>len(raw):
            raise RemoteError('Invalid Jupyter buffer offsets')
        raw = raw[offsets[0]:offsets[1] if count>1 else len(raw)].decode('utf-8')
    return json.loads(raw)


class JupyterTransport:
    def __init__(self, url, kernel_id=None, shutdown_on_close=False, output_callback=None):
        u = urlsplit(url)
        if u.scheme not in ('https', 'http') or not u.hostname:
            raise ValueError('A valid Jupyter server URL is required')
        query = dict(parse_qsl(u.query))
        self.token = query.pop('token', '')
        self.base = urlunsplit((u.scheme, u.netloc, u.path.rstrip('/'), '', ''))
        self.http = requests.Session()
        if self.token:
            self.http.headers['Authorization'] = 'token ' + self.token
        self.kernel_id = kernel_id
        self.shutdown_on_close = shutdown_on_close
        self.output_callback = output_callback

    def start(self):
        if self.kernel_id:
            r = self.http.get(self.base + '/api/kernels/' + quote(self.kernel_id, safe=''), timeout=60)
            r.raise_for_status()
            return
        r = self.http.get(self.base + '/api/sessions', timeout=60)
        r.raise_for_status()
        notebooks = [s['kernel']['id'] for s in r.json()
                     if s.get('type') == 'notebook' and s.get('kernel', {}).get('id')]
        if len(set(notebooks)) == 1:
            self.kernel_id = notebooks[0]
            return
        r = self.http.get(self.base + '/api/kernels', timeout=60)
        r.raise_for_status()
        kernels = r.json()
        if len(kernels) == 1:
            self.kernel_id = kernels[0]['id']
            return
        if len(kernels) > 1:
            raise RemoteError('Multiple remote kernels exist. Specify the notebook kernel_id to avoid loading another model.')
        r = self.http.post(self.base + '/api/kernels', json={'name': 'python3'}, timeout=60)
        r.raise_for_status()
        self.kernel_id = r.json()['id']

    def execute(self, code, timeout=1200):
        if not self.kernel_id:
            raise RemoteError('No live kernel')
        u = urlsplit(self.base)
        query = {'session_id': uuid.uuid4().hex}
        if self.token:
            query['token'] = self.token
        url = urlunsplit(('wss' if u.scheme == 'https' else 'ws', u.netloc,
                         u.path + '/api/kernels/' + quote(self.kernel_id, safe='') + '/channels',
                         urlencode(query), ''))
        ws = websocket.create_connection(url, timeout=min(timeout, 30),
            header={'Authorization': 'token ' + self.token} if self.token else {})
        msg_id = uuid.uuid4().hex
        msg = {'header': {'msg_id': msg_id, 'username': 'comfyui', 'session': uuid.uuid4().hex,
                         'msg_type': 'execute_request', 'version': '5.3'},
               'parent_header': {}, 'metadata': {}, 'channel': 'shell', 'buffers': [],
               'content': {'code': code, 'silent': False, 'store_history': False,
                           'user_expressions': {}, 'allow_stdin': False, 'stop_on_error': True}}
        chunks, error, idle, replied = [], None, False, False
        deadline = time.monotonic() + timeout
        try:
            ws.send(json.dumps(msg))
            while not (idle and replied):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError('Remote execution timed out; generation may still be running')
                ws.settimeout(min(remaining, 30))
                try:
                    raw = ws.recv()
                except websocket.WebSocketTimeoutException:
                    continue
                if not raw:
                    raise RemoteError('Jupyter connection closed')
                reply = decode_message(raw)
                if reply.get('parent_header', {}).get('msg_id') != msg_id:
                    continue
                kind = reply.get('msg_type') or reply.get('header', {}).get('msg_type')
                c = reply.get('content', {})
                if kind == 'stream':
                    chunk = c.get('text', '')
                    chunks.append(chunk)
                    if self.output_callback:
                        self.output_callback(chunk)
                elif kind == 'error':
                    error = c.get('ename', 'Remote error') + ': ' + c.get('evalue', '')
                elif kind == 'execute_reply':
                    replied = True
                    if c.get('status') == 'error':
                        error = c.get('ename', 'Remote error') + ': ' + c.get('evalue', '')
                elif kind == 'status' and c.get('execution_state') == 'idle':
                    idle = True
            if error:
                raise RemoteError(error)
            return ''.join(chunks)
        finally:
            ws.close()

    def close(self):
        if self.kernel_id and self.shutdown_on_close:
            r = self.http.delete(self.base + '/api/kernels/' + quote(self.kernel_id, safe=''), timeout=30)
            if r.status_code not in (204, 404):
                r.raise_for_status()
        self.kernel_id = None
        self.http.close()


class ModelSession:
    """Serialize calls, load once per kernel, expire after a fixed lifetime.

    The lifetime starts after bootstrap; an active generation finishes first.
    A transport close normally detaches the client and preserves the notebook
    kernel. Explicit shutdown_on_close is required to delete a kernel.
    """
    def __init__(self, factory, bootstrap, lifetime_seconds=3600, clock=time.monotonic):
        if lifetime_seconds is not None and lifetime_seconds <= 0:
            raise ValueError('lifetime_seconds must be positive or None')
        self.factory, self.bootstrap = factory, bootstrap
        self.lifetime, self.clock = lifetime_seconds, clock
        self.lock = threading.Lock()
        self.transport, self.deadline, self.timer = None, 0, None

    def _close(self):
        if self.timer:
            self.timer.cancel()
            self.timer = None
        transport, self.transport = self.transport, None
        if transport:
            transport.close()

    def close(self):
        with self.lock:
            self._close()

    def _expire(self):
        with self.lock:
            if self.transport and self.clock() >= self.deadline:
                self._close()

    def execute(self, code, timeout=1200):
        with self.lock:
            if self.transport and self.clock() >= self.deadline:
                self._close()
            if self.transport is None:
                candidate = self.factory()
                try:
                    candidate.start()
                    candidate.execute(self.bootstrap, timeout)
                except Exception:
                    try:
                        candidate.close()
                    except Exception:
                        pass
                    raise
                self.transport = candidate
                self.deadline = self.clock() + self.lifetime if self.lifetime is not None else float('inf')
                if self.lifetime is not None:
                    self.timer = threading.Timer(self.lifetime, self._expire)
                    self.timer.daemon = True
                    self.timer.start()
            # Do not retry a failed execution automatically: it may already have generated an image.
            try:
                return self.transport.execute(code, timeout)
            except Exception:
                # The kernel may have died, or its model state may be incomplete.
                # Discard it for the next explicit request; never resubmit this one.
                try:
                    self._close()
                except Exception:
                    pass
                raise
