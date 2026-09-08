"""Generate the deterministic froggeric v22.5 input matrix (offline, no network).

Each input is a JSON document:
  messages: list of {role, content, [reasoning_content|thinking|reasoning], [tool_calls], ...}
  tools:    optional list of OpenAI-style tool definitions
  kwargs:   optional template kwargs
  expect_error: optional substring that the oracle must raise

Inputs are oracle-facing (OpenAI wire shapes). The C++ test harness maps the same files
onto its isomorphic model (reasoning field aliases collapse into reasoning_content).
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(os.path.join(HERE, '..', '..', 'tests', 'fixtures', 'frontend',
                                   'froggeric_v22_5', 'inputs'))
os.makedirs(OUT, exist_ok=True)

TOOL = {"type": "function", "function": {
    "name": "get_weather", "description": "Get weather for a city",
    "parameters": {"type": "object", "properties": {
        "city": {"type": "string"}, "units": {"type": "string", "enum": ["c", "f"]}},
        "required": ["city"]}}}
TOOL2 = {"type": "function", "function": {
    "name": "zeta_search", "description": "Search the web",
    "parameters": {"type": "object", "properties": {"query": {"type": "string"}}}}}


def w(name, spec):
    path = os.path.join(OUT, name + '.json')
    with open(path, 'w', encoding='utf-8') as fh:
        json.dump(spec, fh, ensure_ascii=False, indent=1)


def tc(function=None, name=None, arguments=None, **extra):
    if isinstance(function, dict):
        call = {'function': function}
    elif function is not None:
        call = {'function': {'name': name, 'arguments': arguments}}
    else:
        call = {'name': name, 'arguments': arguments}
    call.update(extra)
    return [call]


# ---------------- text / thinking / effort ----------------
w('001-basic-user', {'messages': [{'role': 'user', 'content': 'hello'}]})
w('002-basic-noprompt', {'messages': [{'role': 'user', 'content': 'hello'}],
                         'kwargs': {'add_generation_prompt': False}})
w('003-system-single', {'messages': [{'role': 'system', 'content': 'You are terse.'},
                                     {'role': 'user', 'content': 'hi'}]})
w('004-system-multi-merge', {'messages': [
    {'role': 'system', 'content': 'Part A.'},
    {'role': 'developer', 'content': 'Part B.'},
    {'role': 'system', 'content': '   '},
    {'role': 'system', 'content': 'Part C.'},
    {'role': 'user', 'content': 'hi'}]})
w('005-system-multi-tag', {'messages': [
    {'role': 'system', 'content': 'Be nice.<|think_low|>'},
    {'role': 'user', 'content': 'hi'}]})
w('006-system-mid', {'messages': [
    {'role': 'system', 'content': 'A'},
    {'role': 'user', 'content': 'u1'},
    {'role': 'system', 'content': 'mid instruction'},
    {'role': 'user', 'content': 'u2'}]})
w('007-developer-mid', {'messages': [
    {'role': 'user', 'content': 'u1'},
    {'role': 'developer', 'content': 'mid'},
    {'role': 'user', 'content': 'u2'}]})
w('008-thinking-off', {'messages': [{'role': 'user', 'content': 'hi'}],
                       'kwargs': {'enable_thinking': False}})
w('009-effort-none', {'messages': [{'role': 'user', 'content': 'hi'}],
                      'kwargs': {'reasoning_effort': 'none'}})
w('010-effort-off', {'messages': [{'role': 'user', 'content': 'hi'}],
                     'kwargs': {'reasoning_effort': 'off'}})
w('011-effort-minimal', {'messages': [{'role': 'user', 'content': 'hi'}],
                         'kwargs': {'reasoning_effort': 'minimal'}})
w('012-effort-low', {'messages': [{'role': 'user', 'content': 'hi'}],
                     'kwargs': {'reasoning_effort': 'low'}})
w('013-effort-medium', {'messages': [{'role': 'user', 'content': 'hi'}],
                        'kwargs': {'reasoning_effort': 'medium'}})
w('014-effort-high', {'messages': [{'role': 'user', 'content': 'hi'}],
                      'kwargs': {'reasoning_effort': 'high'}})
w('015-effort-xhigh', {'messages': [{'role': 'user', 'content': 'hi'}],
                       'kwargs': {'reasoning_effort': 'xhigh'}})
w('016-effort-max', {'messages': [{'role': 'user', 'content': 'hi'}],
                     'kwargs': {'reasoning_effort': 'max'}})
w('017-effort-ultracode', {'messages': [{'role': 'user', 'content': 'hi'}],
                           'kwargs': {'reasoning_effort': 'ultracode'}})
w('018-effort-extreme', {'messages': [{'role': 'user', 'content': 'hi'}],
                         'kwargs': {'reasoning_effort': 'extreme'}})
w('019-effort-unknown', {'messages': [{'role': 'user', 'content': 'hi'}],
                         'kwargs': {'reasoning_effort': 'banana'}})
w('020-effort-with-disabled-thinking', {'messages': [{'role': 'user', 'content': 'hi'}],
                                        'kwargs': {'enable_thinking': False,
                                                  'reasoning_effort': 'xhigh'}})
w('021-system-with-effort', {'messages': [
    {'role': 'system', 'content': 'Be helpful.'},
    {'role': 'user', 'content': 'hi'}], 'kwargs': {'reasoning_effort': 'xhigh'}})
w('022-effort-low-with-tools', {'messages': [
    {'role': 'system', 'content': 'Be exact'}, {'role': 'user', 'content': 'hi'}],
    'tools': [TOOL], 'kwargs': {'reasoning_effort': 'low'}})
w('023-effort-none-with-tools', {'messages': [
    {'role': 'system', 'content': 'Be exact'}, {'role': 'user', 'content': 'hi'}],
    'tools': [TOOL], 'kwargs': {'reasoning_effort': 'none'}})
w('024-auto-disable-thinking-tools', {'messages': [
    {'role': 'system', 'content': 'Be exact'}, {'role': 'user', 'content': 'hi'}],
    'tools': [TOOL], 'kwargs': {'auto_disable_thinking_with_tools': True}})
w('025-auto-disable-no-tools', {'messages': [
    {'role': 'user', 'content': 'hi'}],
    'kwargs': {'auto_disable_thinking_with_tools': True}})
w('026-unknown-role', {'messages': [
    {'role': 'user', 'content': 'u'}, {'role': 'critic', 'content': 'x'}],
    'kwargs': {'add_generation_prompt': False},
    'oracle_only': True})

# ---------------- inline tags ----------------
w('030-tag-think-off', {'messages': [
    {'role': 'user', 'content': 'do it <|think_off|>'},
    {'role': 'assistant', 'content': 'B', 'reasoning_content': 'R'},
    {'role': 'user', 'content': 'again'}]})
w('031-tag-think-on-after-off', {'messages': [
    {'role': 'user', 'content': 'off <|think_off|>'},
    {'role': 'user', 'content': 'on <|think_on|>'}]})
w('032-tag-priority-same-string', {'messages': [
    {'role': 'user', 'content': 'x <|think_off|> y <|think_on|> z'}]})
w('033-tag-multitem-list', {'messages': [
    {'role': 'user', 'content': ['<|think_off|>', '<|think_on|>']}]})
w('034-tag-high', {'messages': [{'role': 'user', 'content': '<|think_high|> check'}]})
w('035-tag-ultracode', {'messages': [{'role': 'user', 'content': '<|think_ultracode|>'}]})
w('036-tag-max', {'messages': [{'role': 'user', 'content': '<|think_max|>'}]})
w('037-tag-extreme', {'messages': [{'role': 'user', 'content': '<|think_extreme|>'}]})
w('038-tag-low', {'messages': [{'role': 'user', 'content': '<|think_low|> quick'}]})
w('039-tag-minimal', {'messages': [{'role': 'user', 'content': '<|think_minimal|>'}]})
w('040-tag-medium', {'messages': [{'role': 'user', 'content': '<|think_medium|>'}]})
w('041-tag-in-leading-system', {'messages': [
    {'role': 'system', 'content': '<|think_off|> be brief'},
    {'role': 'user', 'content': 'hi'}]})
w('042-tag-multipart-dicts', {'messages': [
    {'role': 'user', 'content': [{'type': 'text', 'text': 'start'},
                                 {'type': 'text', 'text': '<|think_xhigh|>'}]}]})
w('043-tag-all-ten-strip', {'messages': [
    {'role': 'user', 'content':
     '<|think_off|>a<|think_on|>b<|think_xhigh|>c<|think_high|>d<|think_ultracode|>e'
     '<|think_extreme|>f<|think_max|>g<|think_low|>h<|think_minimal|>i<|think_medium|>j'}]})

# ---------------- preserve / reasoning fields ----------------
HIST = {'messages': [
    {'role': 'user', 'content': 'q1'},
    {'role': 'assistant', 'content': 'b1', 'reasoning_content': 'r1'},
    {'role': 'user', 'content': 'q2'},
    {'role': 'assistant', 'content': 'b2', 'reasoning_content': 'r2'}]}
w('050-preserve-default', HIST)
w('051-preserve-false', dict(HIST, kwargs={'preserve_thinking': False}))
w('052-preserve-true-explicit', dict(HIST, kwargs={'preserve_thinking': True}))
w('053-preserve-reasoning-alias', dict(HIST, kwargs={'preserve_reasoning': False}))
w('054-preserve-conflict-reasoning-wins', dict(
    HIST, kwargs={'preserve_reasoning': False, 'preserve_thinking': True}))
w('055-thinking-field-alias', {'messages': [
    {'role': 'user', 'content': 'q'},
    {'role': 'assistant', 'content': 'b', 'thinking': 'r'}]})
w('056-reasoning-field-alias', {'messages': [
    {'role': 'user', 'content': 'q'},
    {'role': 'assistant', 'content': 'b', 'reasoning': 'r'}]})
w('057-in-content-think', {'messages': [
    {'role': 'user', 'content': 'q'},
    {'role': 'assistant', 'content': '<think>\nR\n</think>\n\nB'}]})
w('058-in-content-thinking', {'messages': [
    {'role': 'user', 'content': 'q'},
    {'role': 'assistant', 'content': '<thinking>\nR\n</thinking>\n\nB'}]})
w('059-truncated-close-only', {'messages': [
    {'role': 'user', 'content': 'q'},
    {'role': 'assistant', 'content': '</think>\nrest'}]})
w('060-spaced-close', {'messages': [
    {'role': 'user', 'content': 'q'},
    {'role': 'assistant', 'content': 'a\n</ think>B'}]})
w('061-space-after-close', {'messages': [
    {'role': 'user', 'content': 'q'},
    {'role': 'assistant', 'content': 'a\n</think >B'}]})
w('062-explicit-plus-leading-think', {'messages': [
    {'role': 'user', 'content': 'q'},
    {'role': 'assistant', 'content': '<think>\nR2\n</think>\n\nB', 'reasoning_content': 'R1'}]})
w('063-assistant-empty', {'messages': [
    {'role': 'user', 'content': 'q'},
    {'role': 'assistant', 'content': ''}]})
w('064-assistant-think-only', {'messages': [
    {'role': 'user', 'content': 'q'},
    {'role': 'assistant', 'content': '<think>\nR\n</think>'}]})

# ---------------- content shapes ----------------
w('070-content-null', {'messages': [{'role': 'user', 'content': None}]})
w('071-content-empty', {'messages': [{'role': 'user', 'content': ''}]})
w('072-multipart-text-only', {'messages': [
    {'role': 'user', 'content': [{'type': 'text', 'text': 'a'}, {'type': 'text', 'text': 'b'}]}]})
w('073-empty-messages', {'messages': [], 'expect_error': 'No messages provided.'})

# ---------------- vision ----------------
w('080-image', {'messages': [
    {'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': 'what is this'}]}]})
w('081-video', {'messages': [
    {'role': 'user', 'content': [{'type': 'video'}]}]})
w('082-add-vision-id', {'messages': [
    {'role': 'user', 'content': [{'type': 'image'}, {'type': 'video'},
                                 {'type': 'image'}]}],
    'kwargs': {'add_vision_id': True}})
w('083-multimix', {'messages': [
    {'role': 'user', 'content': [{'type': 'text', 'text': 'a'}, {'type': 'image'},
                                 {'type': 'text', 'text': 'b'}, {'type': 'video'},
                                 {'type': 'text', 'text': 'c'}]}]})
w('084-system-image-error', {'messages': [
    {'role': 'system', 'content': [{'type': 'image'}]}],
    'expect_error': 'System message cannot contain images.'})
w('085-system-video-error', {'messages': [
    {'role': 'system', 'content': [{'type': 'text', 'text': 'x'}, {'type': 'video'}]}],
    'expect_error': 'System message cannot contain videos.'})

# ---------------- multi-step / last query ----------------
def tool_turn(query, answers):
    msgs = [{'role': 'user', 'content': query}]
    for i, (body, reasoning) in enumerate(answers):
        msgs.append({'role': 'assistant', 'content': body, 'reasoning_content': reasoning,
                     'tool_calls': tc(name='get_weather', arguments={'city': 'Paris%d' % i})})
        msgs.append({'role': 'tool', 'content': 'temp 20C step %d' % i})
    return msgs
w('090-multi-step-preserve-false', {
    'messages': [{'role': 'user', 'content': 'q'},
                 {'role': 'assistant', 'content': '', 'reasoning_content': 'r1',
                  'tool_calls': tc(name='get_weather', arguments={'city': 'A'})},
                 {'role': 'tool', 'content': '<tool_response>ok1</tool_response>'},
                 {'role': 'assistant', 'content': 'done', 'reasoning_content': 'r2'}],
    'kwargs': {'preserve_thinking': False}})
w('091-multi-step-preserve-true', {
    'messages': [{'role': 'user', 'content': 'q'},
                 {'role': 'assistant', 'content': '', 'reasoning_content': 'r1',
                  'tool_calls': tc(name='get_weather', arguments={'city': 'A'})},
                 {'role': 'tool', 'content': 'ok1'},
                 {'role': 'assistant', 'content': 'done', 'reasoning_content': 'r2'}],
    'kwargs': {'preserve_thinking': True}})
w('092-history-starts-with-tool', {'messages': [
    {'role': 'tool', 'content': 'boot result'},
    {'role': 'assistant', 'content': 'ok', 'reasoning_content': 'r'}],
    'kwargs': {'preserve_thinking': False}})
w('093-tool-response-plus-extra', {'messages': [
    {'role': 'user', 'content': 'q'},
    {'role': 'assistant', 'content': '', 'reasoning_content': 'r0',
     'tool_calls': tc(name='get_weather', arguments={'city': 'A'})},
    {'role': 'user', 'content': '<tool_response>ok</tool_response> and also this'},
    {'role': 'assistant', 'content': 'b', 'reasoning_content': 'r1'}],
    'kwargs': {'preserve_thinking': False}})

# ---------------- tools: XML ----------------
w('100-tools-xml-single', {'messages': [
    {'role': 'system', 'content': 'Be exact'}, {'role': 'user', 'content': 'hi'}],
    'tools': [TOOL]})
w('101-tools-xml-two', {'messages': [
    {'role': 'user', 'content': 'hi'}], 'tools': [TOOL2, TOOL]})
w('102-tools-xml-parallel', {'messages': [
    {'role': 'user', 'content': 'q'},
    {'role': 'assistant', 'content': 'checking',
     'tool_calls': tc(function={'name': 'get_weather', 'arguments': {'city': 'Paris'}})}],
    'tools': [TOOL]})
w('103-tools-xml-empty-args', {'messages': [
    {'role': 'user', 'content': 'q'},
    {'role': 'assistant', 'content': 'b',
     'tool_calls': tc(function={'name': 'get_weather', 'arguments': {}})}],
    'tools': [TOOL]})
w('104-tools-xml-string-args', {'messages': [
    {'role': 'user', 'content': 'q'},
    {'role': 'assistant', 'content': '',
     'tool_calls': tc(function={'name': 'get_weather', 'arguments': '{"city": "Paris"}'})}],
    'tools': [TOOL], 'oracle_only': True})
w('105-tools-xml-nonobject-string-args', {'messages': [
    {'role': 'user', 'content': 'q'},
    {'role': 'assistant', 'content': '',
     'tool_calls': tc(function={'name': 'get_weather', 'arguments': '[1, 2]'})}],
    'tools': [TOOL], 'oracle_only': True})
w('106-tools-xml-typed-args', {'messages': [
    {'role': 'user', 'content': 'q'},
    {'role': 'assistant', 'content': '',
     'tool_calls': tc(function={'name': 'f', 'arguments': {
         'zeta': 'last', 'alpha': 'first', 'num': 7, 'flag': True, 'nil': None,
         'arr': [1, 'two'], 'obj': {'b': 2, 'a': 1}}})}],
    'tools': [TOOL]})
w('107-tools-xml-arg-trunc', {'messages': [
    {'role': 'user', 'content': 'q'},
    {'role': 'assistant', 'content': '',
     'tool_calls': tc(function={'name': 'f', 'arguments': {'long': 'x' * 25}})}],
    'tools': [TOOL], 'kwargs': {'max_tool_arg_chars': 10}})
w('108-tools-xml-arg-trunc-boundary', {'messages': [
    {'role': 'user', 'content': 'q'},
    {'role': 'assistant', 'content': '',
     'tool_calls': tc(function={'name': 'f', 'arguments': {'edge': 'x' * 10}})}],
    'tools': [TOOL], 'kwargs': {'max_tool_arg_chars': 10}})
w('109-tools-xml-parallel-calls', {'messages': [
    {'role': 'user', 'content': 'q'},
    {'role': 'assistant', 'content': '',
     'tool_calls': [
         {'function': {'name': 'get_weather', 'arguments': {'city': 'A'}}},
         {'function': {'name': 'zeta_search', 'arguments': {'query': 'b'}}}]},
    {'role': 'tool', 'content': 'r1'},
    {'role': 'tool', 'content': 'r2'}],
    'tools': [TOOL, TOOL2]})
w('110-tools-unicode-args', {'messages': [
    {'role': 'user', 'content': 'q'},
    {'role': 'assistant', 'content': '',
     'tool_calls': tc(function={'name': 'f', 'arguments': {
         'msg': 'héllo 数据 "quoted" <tag> & more', 'name': "o'brien"}})}],
    'tools': [TOOL]})

# ---------------- tools: JSON ----------------
w('120-tools-json-single', {'messages': [
    {'role': 'system', 'content': 'Be exact'}, {'role': 'user', 'content': 'hi'}],
    'tools': [TOOL], 'kwargs': {'tool_call_format': 'json'}})
w('121-tools-json-args-mapping', {'messages': [
    {'role': 'user', 'content': 'q'},
    {'role': 'assistant', 'content': 'b',
     'tool_calls': tc(function={'name': 'f', 'arguments': {
         'zeta': 'last', 'alpha': 'first', 'num': 7}})}],
    'tools': [TOOL], 'kwargs': {'tool_call_format': 'json'}})
w('122-tools-json-args-string', {'messages': [
    {'role': 'user', 'content': 'q'},
    {'role': 'assistant', 'content': '',
     'tool_calls': tc(function={'name': 'f', 'arguments': '{"city": "Paris"}'})}],
    'tools': [TOOL], 'kwargs': {'tool_call_format': 'json'}})
w('123-tools-json-response-exempt', {'messages': [
    {'role': 'user', 'content': 'q'},
    {'role': 'assistant', 'content': '',
     'tool_calls': tc(function={'name': 'f', 'arguments': {}})},
    {'role': 'tool', 'content': '{' + 'x' * 50 + '}'}],
    'tools': [TOOL], 'kwargs': {'tool_call_format': 'json', 'max_tool_response_chars': 10}})
w('124-tools-json-unicode', {'messages': [
    {'role': 'user', 'content': 'q'},
    {'role': 'assistant', 'content': '',
     'tool_calls': tc(function={'name': "f <x> & 'q'", 'arguments': {
         'msg': 'héllo 数据 "quoted" <tag> & more'}})}],
    'tools': [TOOL], 'kwargs': {'tool_call_format': 'json'}})

# ---------------- tool responses: truncation / errors ----------------
w('130-tool-basic', {'messages': tool_turn('q', [('b0', 'r0')])})
w('131-tool-grouped', {'messages': [
    {'role': 'user', 'content': 'q'},
    {'role': 'assistant', 'content': '',
     'tool_calls': [
         {'function': {'name': 'get_weather', 'arguments': {'city': 'A'}}},
         {'function': {'name': 'zeta_search', 'arguments': {'query': 'b'}}}]},
    {'role': 'tool', 'content': 'first result'},
    {'role': 'tool', 'content': 'second result'}],
    'tools': [TOOL, TOOL2]})
w('132-tool-response-trunc', {'messages': [
    {'role': 'user', 'content': 'q'},
    {'role': 'assistant', 'content': '',
     'tool_calls': tc(function={'name': 'f', 'arguments': {}})},
    {'role': 'tool', 'content': 'y' * 30}],
    'tools': [TOOL], 'kwargs': {'max_tool_response_chars': 10}})
w('133-tool-response-trunc-boundary', {'messages': [
    {'role': 'user', 'content': 'q'},
    {'role': 'assistant', 'content': '',
     'tool_calls': tc(function={'name': 'f', 'arguments': {}})},
    {'role': 'tool', 'content': 'y' * 10}],
    'tools': [TOOL], 'kwargs': {'max_tool_response_chars': 10}})
w('134-error-weak', {'messages': [
    {'role': 'user', 'content': 'q'},
    {'role': 'assistant', 'content': '',
     'tool_calls': tc(function={'name': 'f', 'arguments': {}})},
    {'role': 'tool', 'content': 'Error: file not found'}],
    'tools': [TOOL]})
w('135-error-strong-traceback', {'messages': [
    {'role': 'user', 'content': 'q'},
    {'role': 'assistant', 'content': '',
     'tool_calls': tc(function={'name': 'f', 'arguments': {}})},
    {'role': 'tool', 'content': 'traceback (most recent call last):\n  File "x.py"\nBoom'}],
    'tools': [TOOL]})
w('136-error-consecutive-two', {'messages': [
    {'role': 'user', 'content': 'q'},
    {'role': 'assistant', 'content': '',
     'tool_calls': tc(function={'name': 'f', 'arguments': {'a': 1}})},
    {'role': 'tool', 'content': 'Error: one'},
    {'role': 'assistant', 'content': '',
     'tool_calls': tc(function={'name': 'f', 'arguments': {'a': 2}})},
    {'role': 'tool', 'content': 'Error: two'}],
    'tools': [TOOL]})
w('137-error-reset', {'messages': [
    {'role': 'user', 'content': 'q'},
    {'role': 'assistant', 'content': '',
     'tool_calls': tc(function={'name': 'f', 'arguments': {'a': 1}})},
    {'role': 'tool', 'content': 'Error: one'},
    {'role': 'assistant', 'content': '',
     'tool_calls': tc(function={'name': 'f', 'arguments': {'a': 2}})},
    {'role': 'tool', 'content': 'success'},
    {'role': 'assistant', 'content': '',
     'tool_calls': tc(function={'name': 'f', 'arguments': {'a': 3}})},
    {'role': 'tool', 'content': 'Error: again'}],
    'tools': [TOOL]})
w('138-error-suppressed-long', {'messages': [
    {'role': 'user', 'content': 'q'},
    {'role': 'assistant', 'content': '',
     'tool_calls': tc(function={'name': 'f', 'arguments': {}})},
    {'role': 'tool', 'content': 'error: ' + 'z' * 600}],
    'tools': [TOOL]})
w('139-error-suppressed-dollarsign', {'messages': [
    {'role': 'user', 'content': 'q'},
    {'role': 'assistant', 'content': '',
     'tool_calls': tc(function={'name': 'f', 'arguments': {}})},
    {'role': 'tool', 'content': '$ ls\nerror: no such file'}],
    'tools': [TOOL]})
w('140-error-field-null-ok', {'messages': [
    {'role': 'user', 'content': 'q'},
    {'role': 'assistant', 'content': '',
     'tool_calls': tc(function={'name': 'f', 'arguments': {}})},
    {'role': 'tool', 'content': '{"error": null, "data": [1, 2, 3]}'}],
    'tools': [TOOL]})
w('141-error-code-suppressed', {'messages': [
    {'role': 'user', 'content': 'q'},
    {'role': 'assistant', 'content': '',
     'tool_calls': tc(function={'name': 'f', 'arguments': {}})},
    {'role': 'tool', 'content': 'import os\nthrow new Error("boom")'}],
    'tools': [TOOL]})
w('142-error-exit-code-1', {'messages': [
    {'role': 'user', 'content': 'q'},
    {'role': 'assistant', 'content': '',
     'tool_calls': tc(function={'name': 'f', 'arguments': {}})},
    {'role': 'tool', 'content': 'exit code: 1'}],
    'tools': [TOOL]})
w('143-error-exit-code-0', {'messages': [
    {'role': 'user', 'content': 'q'},
    {'role': 'assistant', 'content': '',
     'tool_calls': tc(function={'name': 'f', 'arguments': {}})},
    {'role': 'tool', 'content': 'exit code: 0'}],
    'tools': [TOOL]})
w('144-user-resets-failures', {'messages': [
    {'role': 'user', 'content': 'q'},
    {'role': 'assistant', 'content': '',
     'tool_calls': tc(function={'name': 'f', 'arguments': {}})},
    {'role': 'tool', 'content': 'Error: one'},
    {'role': 'user', 'content': 'try again'},
    {'role': 'assistant', 'content': '',
     'tool_calls': tc(function={'name': 'f', 'arguments': {}})},
    {'role': 'tool', 'content': 'Error: two'}],
    'tools': [TOOL]})

# ---------------- renderer parity regressions ----------------
# Multiple close markers: jinja uses split(marker)[0] for reasoning and split(marker)[-1]
# for the body, so the LAST close marker delimits the body.
w('160-think-multi-close-explicit', {'messages': [
    {'role': 'user', 'content': 'q'},
    {'role': 'assistant', 'content': '<think>x</think>mid</think>tail',
     'reasoning_content': 'r'}]})
w('161-think-multi-close-content', {'messages': [
    {'role': 'user', 'content': 'q'},
    {'role': 'assistant', 'content': 'a\n</think>b\n</think>c'}]})
# Inline tags are stripped from the concatenated render, so a tag may span two parts and
# part-boundary whitespace must survive exactly as the oracle renders it.
w('162-tag-multipart-whitespace', {'messages': [
    {'role': 'user', 'content': [
        {'type': 'text', 'text': '<|think_off|>a '},
        {'type': 'text', 'text': ' b'}]}]})
w('163-tag-split-across-parts', {'messages': [
    {'role': 'user', 'content': [
        {'type': 'text', 'text': '<|think_'},
        {'type': 'text', 'text': 'off|>abc'}]}]})
# tojson parity: Python json.dumps(ensure_ascii=True) escapes C0 controls and DEL.
w('164-tojson-del-escape', {'messages': [
    {'role': 'user', 'content': 'q'},
    {'role': 'assistant', 'content': '', 'tool_calls': [
        {'function': {'name': 'f', 'arguments': {
            's': ''.join(chr(i) for i in range(0x20)) + chr(0x7f) + '~ \u00e9\U0001F600'}}}]}],
    'tools': [TOOL], 'kwargs': {'tool_call_format': 'json'}})
# Media placeholders in assistant and tool history must keep their rendered bytes and their
# placeholder metadata (the compiled renderer slices the rendered block, not a plain string).
w('165-assistant-media', {'messages': [
    {'role': 'user', 'content': 'hi'},
    {'role': 'assistant', 'content': [{'type': 'text', 'text': 'see '}, {'type': 'image'}]},
    {'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': 'now'}]}],
    'kwargs': {'add_vision_id': True}})
w('166-tool-media', {'messages': [
    {'role': 'user', 'content': 'capture'},
    {'role': 'assistant', 'content': '',
     'tool_calls': tc(function={'name': 'f', 'arguments': {}})},
    {'role': 'tool', 'content': [{'type': 'text', 'text': 'captured '}, {'type': 'image'}]}],
    'tools': [TOOL]})

# ---------------- long multi-step (>50 messages) ----------------
def long_tool_loop(n):
    msgs = [{'role': 'user', 'content': 'q'}]
    for i in range(n):
        msgs.append({'role': 'assistant', 'content': '', 'reasoning_content': 'r%d' % i,
                     'tool_calls': tc(function={'name': 'f', 'arguments': {'i': i}})})
        msgs.append({'role': 'tool', 'content': 'ok %d' % i})
    msgs.append({'role': 'assistant', 'content': 'done', 'reasoning_content': 'final'})
    return msgs
w('150-tool-loop-55-preserve-false', {'messages': long_tool_loop(27),
                                      'tools': [TOOL],
                                      'kwargs': {'preserve_thinking': False}})
w('151-tool-loop-5-preserve-false', {'messages': long_tool_loop(2),
                                     'tools': [TOOL],
                                     'kwargs': {'preserve_thinking': False}})
w('152-tool-loop-5-preserve-true', {'messages': long_tool_loop(2),
                                    'tools': [TOOL],
                                    'kwargs': {'preserve_thinking': True}})

print('wrote %d inputs to %s' % (len(os.listdir(OUT)), OUT))
