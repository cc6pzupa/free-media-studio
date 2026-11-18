"""Run the language-model preview on a real Kaggle dual-T4 session.

Uses the same AccountBackend path as the GUI: automatic build selection,
download, streamed chat, switching builds, and switching between language,
image and video models on one session. The session is released at the end.

    KAGGLE_USERNAME=... KAGGLE_API_TOKEN=... python llm_gpu_check.py --output results
"""
import argparse
import json
import os
import time
import traceback
from pathlib import Path

PROMPT = [{'role': 'user', 'content': 'In three sentences, explain why lighthouses use Fresnel lenses.'}]


def chat(backend, max_tokens=256):
    pieces = []
    started = time.monotonic()
    first = [None]

    def on_chunk(chunk):
        for choice in chunk.get('choices') or []:
            delta = choice.get('delta') or {}
            text = (delta.get('content') or '') + (delta.get('reasoning') or '')
            if text:
                first[0] = first[0] or time.monotonic()
                pieces.append(text)
        if chunk.get('usage'):
            pieces_usage.append(chunk['usage'])
    pieces_usage = []
    backend.llm_chat({'messages': PROMPT, 'max_tokens': max_tokens, 'temperature': 0.6}, on_chunk)
    end = time.monotonic()
    tokens = (pieces_usage[-1] if pieces_usage else {}).get('completion_tokens') or len(pieces)
    decode = end - (first[0] or end)
    text = ''.join(pieces)
    if not text.strip():
        raise RuntimeError('Empty reply')
    return {'tokens': tokens, 'ttft_s': round((first[0] or end) - started, 2), 'tok_per_s': round(tokens / decode, 1) if decode else None,
            'reply': text[-300:]}


def run(user, token, output, only=None):
    from account_backend import AccountBackend
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    backend = AccountBackend(user, token, cache_dir=output / 'sessions')
    reports = []
    last = [0.0]

    def progress(name):
        def report(event):
            now = time.monotonic()
            if event.get('phase') == 'download' and now - last[0] < 20:
                return
            last[0] = now
            print('LLM_CHECK_PROGRESS ' + name + ' ' + json.dumps({k: event.get(k) for k in ('phase', 'message', 'percent', 'downloaded_gb', 'total_gb') if event.get(k) is not None}, ensure_ascii=False), flush=True)
        return report

    def case(name, step):
        if only and name not in only:
            return
        started = time.monotonic()
        print('LLM_CHECK_START ' + name, flush=True)
        try:
            result = step()
            reports.append(dict(case=name, ok=True, wall_seconds=round(time.monotonic() - started, 1), **result))
        except Exception as error:
            traceback.print_exc()
            cause = error.__context__
            reports.append(dict(case=name, ok=False, wall_seconds=round(time.monotonic() - started, 1), error=type(error).__name__ + ': ' + str(error),
                                **({'remote_error': (type(cause).__name__ + ': ' + str(cause))[:1500]} if cause is not None else {})))
        print('LLM_CHECK_RESULT ' + json.dumps(reports[-1], ensure_ascii=False), flush=True)
        (output / 'report.json').write_text(json.dumps(reports, indent=2, ensure_ascii=False), encoding='utf-8')

    state = {}

    def prepare(name, choice=None, context=8192):
        info = backend.llm_prepare(name, context, choice, progress(name))
        state['info'] = info
        return {k: info.get(k) for k in ('repo', 'quant', 'file', 'bytes', 'placement', 'estimated_tok_per_s', 'loaded', 'request_seconds')}

    try:
        case('qwen3-8b-auto', lambda: dict(prepare('Qwen3-8B'), chat=chat(backend)))
        def alternative():
            alts = [a for a in state['info']['alternatives'] if a['first'] != state['info']['file'] and a['placement'] == 'gpu']
            if not alts:
                raise RuntimeError('No alternative build listed')
            pick = min(alts, key=lambda a: a['bytes'])
            return dict(prepare('Qwen3-8B', {'repo': pick['repo'], 'first': pick['first']}), chat=chat(backend))
        case('qwen3-8b-alternative', alternative)
        def image():
            raw, stats = backend.generate_with_progress('A lighthouse on a cliff at dusk, watercolor', 42, None, progress('image'), model_id='z-image-turbo')
            (output / 'after-llm.png').write_bytes(raw)
            return {k: stats.get(k) for k in ('generation_seconds', 'peak_gpu_gib', 'model_reused')}
        case('image-after-llm', image)
        case('qwen3-30b-a3b-auto', lambda: dict(prepare('Qwen3-30B-A3B'), chat=chat(backend)))
        def video():
            raw, stats = backend.generate_video_with_progress('Waves breaking below a lighthouse, slow camera pan', 42, None,
                                                              model_id='ltx-video-2b-distilled', frames=33, progress=progress('video'))
            (output / 'after-llm.mp4').write_bytes(raw)
            return {k: stats.get(k) for k in ('generation_seconds', 'peak_gpu_gib', 'parallel_strategy')}
        case('video-after-llm', video)
    finally:
        try:
            backend.release_session()
            print('LLM_CHECK_RELEASED', flush=True)
        except Exception as error:
            print('LLM_CHECK_RELEASE_FAILED ' + type(error).__name__, flush=True)
        backend.close()
    return reports


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--output', default='llm-check-results')
    parser.add_argument('--only', nargs='*')
    args = parser.parse_args()
    reports = run(os.environ['KAGGLE_USERNAME'], os.environ.get('KAGGLE_API_TOKEN') or os.environ['KAGGLE_KEY'], args.output, args.only)
    print('LLM_CHECK_FINISHED %d/%d passed' % (sum(r['ok'] for r in reports), len(reports)), flush=True)
    raise SystemExit(0 if reports and all(r['ok'] for r in reports) else 1)
