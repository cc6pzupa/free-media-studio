"""Language-model routes of the local GUI server.

GUI (page session key):  GET /llm/status, POST /llm/prepare, POST /llm/chat
OpenAI-compatible (local API key):  GET /v1/models, POST /v1/chat/completions

Requests are forwarded to the shared local worker, which runs the model on
the Kaggle session. A /v1 request naming a model that is not loaded loads it
first, so a tool only needs the base URL, the key and a model name.
"""
import json
import time

from media_inputs import MAX_REQUEST_BYTES
from studio_client import LlmError, StudioClient, StudioHTTPError

# Names that mean "whatever is loaded" rather than a Hugging Face model.
LOADED_ALIASES = ('', 'default', 'kaggle', 'loaded', 'current')


def _read_json(handler, limit):
    length = int(handler.headers.get('Content-Length', '0'))
    if not 0 < length <= limit:
        raise ValueError('请求过大或为空。')
    body = json.loads(handler.rfile.read(length))
    if not isinstance(body, dict):
        raise ValueError('请求必须为 JSON 对象。')
    return body


def _start_stream(handler, kind):
    handler.send_response(200)
    handler.send_header('Content-Type', kind)
    handler.send_header('Cache-Control', 'no-store')
    handler.send_header('X-Content-Type-Options', 'nosniff')
    handler.end_headers()
    gone = []

    def write(text):
        if gone:
            return
        try:
            handler.wfile.write(text.encode('utf-8'))
            handler.wfile.flush()
        except OSError:
            gone.append(True)
    return write


def _openai_error(handler, status, message, kind='invalid_request_error'):
    return handler.send(status, {'error': {'message': message, 'type': kind}})


def loaded_name(status):
    model = (status or {}).get('model') or {}
    return model.get('name')


def _ensure_model(client, requested, on_event=None):
    """Load the requested model unless it (or nothing in particular) is asked for."""
    status = client.llm_status()
    if status.get('restart_required'):
        raise LlmError('请重启本机后台以启用语言模型。')
    current = loaded_name(status)
    wanted = (requested or '').strip()
    if wanted.lower() in LOADED_ALIASES or wanted in (current, (status.get('model') or {}).get('tag')):
        if not current:
            raise LlmError('还没有加载语言模型。请在请求的 model 中填写模型名，例如 Qwen3-8B，或先在聊天页选择模型。')
        return current
    client.llm_prepare(wanted, on_event=on_event or (lambda event: None))
    return wanted


def _base_url(handler):
    return 'http://127.0.0.1:%d/v1' % handler.server.server_port


def handle_get(handler, path, ui_ok, api_ok, api_key=None, remembered=None):
    """Return True when the request was a language-model route."""
    if path == '/harness':
        if not ui_ok():
            handler.send(401, {'error': {'message': '访问密钥无效'}})
            return True
        import harness
        client = StudioClient(allow_job_inheritance=True)
        try:
            model = loaded_name(client.llm_status())
        except RuntimeError:
            model = None
        finally:
            client.close()
        handler.send(200, {'tools': harness.discover(), 'base_url': _base_url(handler), 'model': model,
                           'remembered': bool(remembered and remembered())})
        return True
    if path == '/llm/status':
        if not ui_ok():
            handler.send(401, {'error': {'message': '访问密钥无效'}})
            return True
        client = StudioClient(allow_job_inheritance=True)
        try:
            handler.send(200, client.llm_status())
        except RuntimeError as error:
            handler.send(503, {'error': {'message': str(error)}})
        finally:
            client.close()
        return True
    if path == '/v1/models':
        if not api_ok():
            _openai_error(handler, 401, 'Invalid API key', 'authentication_error')
            return True
        client = StudioClient(allow_job_inheritance=True)
        try:
            name = loaded_name(client.llm_status())
            models = [{'id': name, 'object': 'model', 'created': int(time.time()), 'owned_by': 'kaggle'}] if name else []
            handler.send(200, {'object': 'list', 'data': models})
        except RuntimeError as error:
            _openai_error(handler, 503, str(error), 'server_error')
        finally:
            client.close()
        return True
    return False


def handle_post(handler, path, ui_ok, api_ok, api_key=None, remembered=None):
    if path in ('/harness/configure', '/harness/install', '/harness/launch'):
        if not ui_ok():
            handler.send(401, {'error': {'message': '访问密钥无效'}})
            return True
        import harness
        try:
            body = _read_json(handler, 1024)
            tool = body.get('id')
            if path == '/harness/configure':
                return handler.send(200, harness.configure(tool, _base_url(handler), api_key(), body.get('model'))) or True
            if path == '/harness/launch':
                if tool != 'open-webui':
                    raise ValueError('这个工具不能从这里启动。')
                return handler.send(200, harness.launch_open_webui(_base_url(handler), api_key())) or True
            harness.install_command(tool)  # refuse before streaming starts
        except (ValueError, TypeError) as error:
            handler.send(400, {'error': {'message': str(error) if isinstance(error, ValueError) else '请求格式有误。'}})
            return True
        except OSError:
            handler.send(500, {'error': {'message': '无法写入该工具的配置文件。'}})
            return True
        write = _start_stream(handler, 'application/x-ndjson; charset=utf-8')
        emit = lambda event: write(json.dumps(event, ensure_ascii=False) + '\n')
        try:
            emit(dict(harness.install(tool, lambda event: emit(dict(event, type='output'))), type='done'))
        except (RuntimeError, OSError, ValueError) as error:
            emit({'type': 'error', 'message': str(error) if isinstance(error, (RuntimeError, ValueError)) else '无法运行安装命令。'})
        return True
    if path in ('/llm/prepare', '/llm/chat'):
        if not ui_ok():
            handler.send(401, {'error': {'message': '访问密钥无效'}})
            return True
        try:
            body = _read_json(handler, MAX_REQUEST_BYTES if path == '/llm/chat' else 4096)
        except (ValueError, TypeError):
            handler.send(400, {'error': {'message': '请求格式有误。'}})
            return True
        write = _start_stream(handler, 'application/x-ndjson; charset=utf-8')
        emit = lambda event: write(json.dumps(event, ensure_ascii=False) + '\n')
        client = StudioClient(allow_job_inheritance=True)
        try:
            if path == '/llm/prepare':
                model = client.llm_prepare(body.get('name'), body.get('context', 8192), body.get('choice'), emit)
                emit({'type': 'done', 'model': model})
            else:
                final = client.llm_chat(body, emit)
                emit(final)
        except (LlmError, StudioHTTPError, RuntimeError) as error:
            emit({'type': 'error', 'message': str(error)})
        finally:
            client.close()
        return True
    if path == '/v1/chat/completions':
        if not api_ok():
            _openai_error(handler, 401, 'Invalid API key', 'authentication_error')
            return True
        try:
            body = _read_json(handler, MAX_REQUEST_BYTES)
        except (ValueError, TypeError):
            _openai_error(handler, 400, 'Invalid JSON body')
            return True
        _openai_chat(handler, body)
        return True
    return False


def _openai_chat(handler, body):
    client = StudioClient(allow_job_inheritance=True)
    stream = bool(body.get('stream'))
    created = int(time.time())
    try:
        if stream:
            write = _start_stream(handler, 'text/event-stream; charset=utf-8')
            # SSE comments keep the connection open while a model downloads.
            name = _ensure_model(client, body.get('model'), lambda event: write(': preparing\n\n'))

            def on_event(event):
                if event.get('type') == 'chunk':
                    chunk = dict(event['chunk'], model=name)
                    write('data: ' + json.dumps(chunk, ensure_ascii=False) + '\n\n')
            try:
                client.llm_chat(body, on_event)
            except (LlmError, StudioHTTPError, RuntimeError) as error:
                write('data: ' + json.dumps({'error': {'message': str(error), 'type': 'server_error'}}, ensure_ascii=False) + '\n\n')
            write('data: [DONE]\n\n')
            return
        name = _ensure_model(client, body.get('model'))
        content, reasoning, finish, usage, ident = [], [], None, None, None

        def collect(event):
            nonlocal finish, usage, ident
            if event.get('type') != 'chunk':
                return
            chunk = event['chunk']
            ident = ident or chunk.get('id')
            usage = chunk.get('usage') or usage
            for choice in chunk.get('choices') or []:
                delta = choice.get('delta') or {}
                content.append(delta.get('content') or '')
                reasoning.append(delta.get('reasoning') or delta.get('reasoning_content') or '')
                finish = choice.get('finish_reason') or finish
        client.llm_chat(body, collect)
        message = {'role': 'assistant', 'content': ''.join(content)}
        if ''.join(reasoning):
            message['reasoning_content'] = ''.join(reasoning)
        handler.send(200, {'id': ident or 'chatcmpl-kaggle', 'object': 'chat.completion', 'created': created, 'model': name,
                           'choices': [{'index': 0, 'message': message, 'finish_reason': finish or 'stop'}],
                           **({'usage': usage} if usage else {})})
    except StudioHTTPError as error:
        _openai_error(handler, error.status_code if 400 <= error.status_code < 500 else 502, str(error))
    except LlmError as error:
        _openai_error(handler, 400 if '请' in str(error) or 'model' in str(error) else 502, str(error), 'invalid_request_error')
    except RuntimeError as error:
        _openai_error(handler, 503, str(error), 'server_error')
    finally:
        client.close()
