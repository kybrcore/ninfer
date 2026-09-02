#!/usr/bin/env python3
"""ctxbench.py - the context/retrieval measurement harness.

One instrument for both the YaRN validation and the E8 phase, so numbers are comparable
across configurations rather than produced by whichever ad-hoc script existed that day.

MEASURES, per tier:
  prompt tokens (as the ENGINE counted them, never as we estimated)
  VRAM used / total, sampled at peak during the request
  prefill time and rate  (time-to-first-token from a STREAMED response)
  decode rate            (remaining tokens / remaining time)
  retrieval              5 needles at 10/25/50/75/90% depth, scored individually
  finish_reason          a truncated answer is not a pass

WHY STREAMING: without it, prefill and decode are one number, and a 275K prompt is ~99%
prefill - so a decode regression would be invisible. TTFT is the only way to separate them.

Usage: ctxbench.py <port> <label> <tokens...>
"""
import json, subprocess, sys, time, urllib.request

PORT, LABEL, TIERS = sys.argv[1], sys.argv[2], [int(x) for x in sys.argv[3:]]
CODES = {10: 'ZEPHYR-11', 25: 'MARLIN-27', 50: 'OBSIDIAN-53', 75: 'CINDER-71', 90: 'HALCYON-97'}
FILLER = 'The quick brown fox jumps over the lazy dog. ' * 40 + '\n'   # ~401 tokens


def vram():
    for exe in ('/usr/lib/wsl/lib/nvidia-smi', 'nvidia-smi'):
        try:
            out = subprocess.run([exe, '--query-gpu=memory.used,memory.total',
                                  '--format=csv,noheader,nounits'],
                                 capture_output=True, timeout=10).stdout.decode()
            u, t = (int(x) for x in out.strip().split(',')[:2])
            return u, t
        except Exception:
            continue
    return 0, 0


def build(tokens):
    blocks = max(20, tokens // 401)
    lines = [FILLER] * blocks
    for d, code in CODES.items():
        lines[min(len(lines) - 1, blocks * d // 100)] = (
            f'Remember: the {d} percent codeword is {code}.\n' + lines[min(len(lines) - 1, blocks * d // 100)])
    q = ('\n\nNow answer from memory. List the five codewords, one per line, exactly:\n'
         '10=<word> 25=<word> 50=<word> 75=<word> 90=<word>')
    return ''.join(lines) + q


def run(tier):
    body = json.dumps({'model': 'qwen3.8-27b',
                       'messages': [{'role': 'user', 'content': build(tier)}],
                       'max_tokens': 700, 'reasoning_effort': 'low', 'stream': True,
                       'stream_options': {'include_usage': True}}).encode()
    req = urllib.request.Request(f'http://127.0.0.1:{PORT}/v1/chat/completions', data=body,
                                 headers={'Content-Type': 'application/json',
                                          'Authorization': 'Bearer ollama'})
    t0 = time.time(); ttft = None; text = ''; usage = None; finish = None; peak = 0
    try:
        with urllib.request.urlopen(req, timeout=3600) as resp:
            for raw in resp:
                line = raw.decode('utf8', 'replace').strip()
                if not line.startswith('data: '):
                    continue
                payload = line[6:]
                if payload == '[DONE]':
                    break
                try:
                    d = json.loads(payload)
                except Exception:
                    continue
                if d.get('usage'):
                    usage = d['usage']
                for ch in d.get('choices', []):
                    delta = ch.get('delta') or {}
                    # TTFT keys off the FIRST token of ANY kind. This model emits
                    # reasoning_content before content, so keying on content alone folds
                    # the whole reasoning generation into 'prefill' and measures decode
                    # over a sliver - which reported an impossible 1080 tok/s once.
                    piece = delta.get('content') or ''
                    any_piece = piece or (delta.get('reasoning_content') or '')
                    if any_piece and ttft is None:
                        ttft = time.time() - t0
                        peak = max(peak, vram()[0])
                    text += piece
                    if ch.get('finish_reason'):
                        finish = ch['finish_reason']
    except Exception as e:
        return {'tier': tier, 'error': f'{type(e).__name__}: {str(e)[:110]}'}
    total = time.time() - t0
    peak = max(peak, vram()[0])
    hits = [str(k) for k in sorted(CODES) if CODES[k].upper() in text.upper()]
    out_tok = (usage or {}).get('completion_tokens') or 0
    return {'tier': tier, 'prompt': (usage or {}).get('prompt_tokens'),
            'out': out_tok, 'ttft': ttft, 'total': total,
            'prefill_rate': ((usage or {}).get('prompt_tokens') or 0) / ttft if ttft else None,
            'decode_rate': (out_tok - 1) / (total - ttft) if ttft and total > ttft else None,
            'hits': hits, 'finish': finish, 'vram': peak, 'vram_total': vram()[1]}


print(f'=== {LABEL} ===')
print(f'{"tier":>8} {"prompt":>8} {"prefill":>9} {"pf tok/s":>9} {"dec tok/s":>10} '
      f'{"vram":>12} {"finish":>8}  retrieval')
for tier in TIERS:
    r = run(tier)
    if 'error' in r:
        print(f'{tier:>8} {"":>8} {"":>9} {"":>9} {"":>10} {"":>12} {"":>8}  ERROR {r["error"]}')
        continue
    print(f'{r["tier"]:>8} {r["prompt"] or 0:>8} {r["ttft"] or 0:>8.1f}s '
          f'{r["prefill_rate"] or 0:>9.0f} {r["decode_rate"] or 0:>10.1f} '
          f'{r["vram"]:>6}/{r["vram_total"]:<5} {str(r["finish"]):>8}  '
          f'{len(r["hits"])}/5 [{" ".join(r["hits"])}]')
