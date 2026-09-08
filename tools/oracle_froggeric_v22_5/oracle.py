"""Offline Jinja oracle for the froggeric v22.5 chat template.

Pinned environment (see tests/fixtures/frontend/froggeric_v22_5/PROVENANCE.md):
  Python 3.11, jinja2 3.1.6, StrictUndefined, keep_trailing_newline=True,
  lstrip_blocks=True, trim_blocks=True, global raise_exception.

Usage:
  oracle.py check       # verify fixture SHAs, render every input with BOTH the pretty and
                        # the oneline template (asserting byte parity), and compare against
                        # the committed goldens. Exits nonzero on any failure.
  oracle.py generate    # render and (re)write golden/ files after deliberate input changes.

The oracle must never reach the network; it only reads local fixture files.
"""
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURE_DIR = os.path.abspath(os.path.join(HERE, '..', '..', 'tests', 'fixtures', 'frontend',
                                           'froggeric_v22_5'))

EXPECTED_SHA = {
    'chat_template.jinja':
        'e57684bae4156211a55473c5a63be976a405a37ab5be5ae0e5abf1df5349c4b2',
    'chat_template_oneline.txt':
        'eecae0e068e60f9c8665f0085b589d3e1c41508d359776c62018512c40b5879b',
}


def load_jinja():
    import jinja2
    if jinja2.__version__ != '3.1.6':
        sys.exit('frozen oracle requires jinja2 3.1.6, found %s' % jinja2.__version__)
    if sys.version_info[:2] != (3, 11):
        sys.exit('frozen oracle requires Python 3.11, found %d.%d'
                 % sys.version_info[:2])
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(FIXTURE_DIR),
        undefined=jinja2.StrictUndefined,
        keep_trailing_newline=True,
        lstrip_blocks=True,
        trim_blocks=True,
    )
    env.globals['raise_exception'] = lambda msg: (_ for _ in ()).throw(Exception(msg))
    return {'chat_template.jinja': env.get_template('chat_template.jinja'),
            'chat_template_oneline.txt': env.get_template('chat_template_oneline.txt')}


def check_fixtures():
    for name, expected in EXPECTED_SHA.items():
        path = os.path.join(FIXTURE_DIR, name)
        with open(path, 'rb') as fh:
            digest = hashlib.sha256(fh.read()).hexdigest()
        if digest != expected:
            sys.exit('fixture drift: %s sha256 %s != %s' % (name, digest, expected))


def render_input(templates, path):
    with open(path, 'r', encoding='utf-8') as fh:
        spec = json.load(fh)
    kwargs = dict(spec.get('kwargs', {}))
    kwargs.setdefault('add_generation_prompt', True)
    messages = spec.get('messages', [])
    if 'tools' in spec:
        kwargs['tools'] = spec['tools']
    expect_error = spec.get('expect_error')
    results = {}
    for name, tpl in templates.items():
        if expect_error is not None:
            try:
                tpl.render(messages=messages, **kwargs)
            except Exception as exc:
                if expect_error in str(exc):
                    results[name] = None
                    continue
                return None, 'expected error %r, got %r' % (expect_error, str(exc))
            return None, 'expected error %r, render succeeded' % expect_error
        rendered = tpl.render(messages=messages, **kwargs)
        results[name] = rendered.encode('utf-8')
    pretty = results['chat_template.jinja']
    oneline = results['chat_template_oneline.txt']
    if pretty != oneline:
        for i, (a, b) in enumerate(zip(pretty, oneline)):
            if a != b:
                return None, 'pretty/oneline parity broken at byte %d: %r vs %r' % (
                    i, pretty[max(0, i - 30):i + 30], oneline[max(0, i - 30):i + 30])
        return None, 'pretty/oneline parity broken (length %d vs %d)' % (len(pretty), len(oneline))
    return pretty, None


def main(argv):
    mode = argv[1] if len(argv) > 1 else 'check'
    check_fixtures()
    templates = load_jinja()
    inputs = sorted(
        f for f in os.listdir(os.path.join(FIXTURE_DIR, 'inputs')) if f.endswith('.json'))
    if not inputs:
        sys.exit('no inputs found under ' + os.path.join(FIXTURE_DIR, 'inputs'))
    failures = 0
    for name in inputs:
        pretty, error = render_input(templates, os.path.join(FIXTURE_DIR, 'inputs', name))
        if error is not None:
            print('FAIL %s: %s' % (name, error))
            failures += 1
            continue
        golden_path = os.path.join(FIXTURE_DIR, 'golden', name[:-len('.json')] + '.expected')
        if pretty is None:
            # expect_error input: no golden payload; check mode validates the raise.
            print('skip %s (expect_error)' % name)
            continue
        if mode == 'generate':
            with open(golden_path, 'wb') as fh:
                fh.write(pretty)
            print('gen  %s (%d bytes)' % (name, len(pretty)))
        else:
            if not os.path.exists(golden_path):
                print('FAIL %s: no golden %s' % (name, os.path.basename(golden_path)))
                failures += 1
                continue
            with open(golden_path, 'rb') as fh:
                golden = fh.read()
            if golden != pretty:
                for i, (a, b) in enumerate(zip(pretty, golden)):
                    if a != b:
                        print('FAIL %s: first diff at byte %d' % (name, i))
                        print('  golden: ...%r' % golden[max(0, i - 40):i + 40])
                        print('  actual: ...%r' % pretty[max(0, i - 40):i + 40])
                        break
                else:
                    print('FAIL %s: length %d vs golden %d' % (name, len(pretty), len(golden)))
                failures += 1
                continue
            print('PASS %s (%d bytes)' % (name, len(pretty)))
    if failures:
        sys.exit('%d input(s) failed' % failures)
    print('all %d inputs %s' % (len(inputs), 'generated' if mode == 'generate' else 'pass'))


if __name__ == '__main__':
    main(sys.argv)
