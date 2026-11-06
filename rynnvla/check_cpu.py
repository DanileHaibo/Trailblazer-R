"""Run with CUDA_VISIBLE_DEVICES='' python -m rynnvla.check_cpu.
Does not instantiate the target, processor, renderer, or CUDA context.
"""
import ast
import hashlib
import importlib
import json
from pathlib import Path

def main():
    root = Path(__file__).resolve().parents[1]
    for path in (root/'rynnvla').glob('*.py'):
        ast.parse(path.read_text(), filename=str(path))
    for name in ['rynnvla', 'rynnvla.backend', 'rynnvla.gate', 'rynnvla.draft', 'rynnvla.runtime', 'rynnvla.evaluate']:
        importlib.import_module(name)
    import torch
    from .draft import TransformerDraft
    from .gate import AbsoluteGate
    from .evaluate import parser, make_plan
    torch.set_num_threads(2)
    args = parser().parse_args(['--base-path', 'BASE', '--official-repo', 'CODE',
        '--tokenizer-path', 'TEXT', '--image-tokenizer-path', 'VQ', '--output', 'OUTPUT'])
    plan = make_plan(args)
    assert len(plan) == 200 and sum(len(p['order']) for p in plan) == 400
    cfg = json.loads((root/'configs/rynnvla.json').read_text())
    gate = AbsoluteGate(cfg['gate'])
    row = [10004]+[10133]*7+[15004]
    assert gate(row, row)[0]
    opposite = row.copy(); opposite[-2] = 10006
    assert not gate(row, opposite)[0]
    displaced = row.copy(); displaced[1] = 10260
    assert not gate(row, displaced)[0]
    manifest = json.loads((root/'weights/rynnvla/manifest.json').read_text())
    results = []
    for name, expected in manifest['checkpoint_sha256'].items():
        path = root/'weights/rynnvla'/name
        assert path.stat().st_size < 100_000_000
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected
        ck = torch.load(path, map_location='cpu', weights_only=False)
        draft = TransformerDraft(ck['horizon'], ck['dims'], ck['architecture']).eval()
        draft.load_state_dict(ck['state_dict'], strict=True)
        with torch.inference_mode():
            out = draft(torch.zeros(1, 16392))
        assert out.shape == (1, ck['horizon'], ck['dims'], 256)
        results.append(dict(file=name, bytes=path.stat().st_size, shape=list(out.shape)))
    assert not torch.cuda.is_initialized()
    print(json.dumps(dict(syntax=True, imports=True, default_pairs=200, default_episodes=400,
                         gate_checks=True, checkpoints=results, cuda_initialized=False), indent=2))

if __name__ == '__main__':
    main()
