"""Maintainer-only export from the original experiment tree; never required at runtime."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import torch


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--lock-environment', choices=['nora', 'pi0'])
    args = parser.parse_args()
    release = Path(__file__).resolve().parents[1]
    if args.lock_environment:
        # Captures exact installed distributions without credentials or local editable paths.
        import importlib.metadata as metadata
        rows = sorted({f'{d.metadata["Name"]}=={d.version}' for d in metadata.distributions() if d.metadata['Name']})
        (release/'environments').mkdir(exist_ok=True)
        (release/'environments'/f'{args.lock_environment}-observed.txt').write_text('\n'.join(rows)+'\n')
        return
    root = args.source_root.resolve()
    config = {'thresholds': {}, 'nora_bounds': {}}
    tree = ast.parse((root/'third_party/nora/experiments/libero/nora_utils.py').read_text())
    bounds_node = next(n.value for n in ast.walk(tree) if isinstance(n, ast.Assign) and any(isinstance(t, ast.Attribute) and t.attr == 'libero_keys' for t in n.targets))
    import numpy as np
    bounds = eval(compile(ast.Expression(bounds_node), '<literal normalization bounds>', 'eval'), {'__builtins__': {}, 'np': np})
    suites = ['libero_object', 'libero_spatial', 'libero_goal', 'libero_10']
    config['nora_bounds'] = {s: [a.tolist() for a in bounds[s]] for s in suites}
    assets = {'nora': {'fast': {'repo': 'physical-intelligence/fast', 'revision': 'ec4d7aa71691cac0b8bed6942be45684db2110f4'}}, 'pi0': {}}
    revisions = ['7c205efb04d628b4e9bf507f794d8d2187b5b67a', 'ef6b673da0d0278d29b67658bf5764ef7d470029', '19f25d50c850d91b8fc100c21bf044e35c53c476', 'ddeffe36e738e407673371933b3fe7229e2287f1']
    for suite, revision in zip(suites, revisions):
        assets['nora'][suite] = {'repo': 'declare-lab/nora-long-finetuned-'+suite.replace('_', '-'), 'revision': revision}
    for key, repo, revision in [('checkpoint', 'lerobot/pi0fast-libero-v044', '2bb8d5599b9cf8023ce9c110ba65af8e818fb4d3'), ('tokenizer', 'jadechoghari/fast-libero-tokenizer-mean-std', '09072b523fb583dce9c16da1327d691cf0504b58'), ('text_tokenizer', 'google/paligemma-3b-pt-224', '35e4f46485b4d07967e7e9935bc3786aad50687c')]:
        assets['pi0'][key] = {'repo': repo, 'revision': revision}
    (release/'weights').mkdir(exist_ok=True)
    provenance = {}
    for name in ['nora', 'pi0']:
        source = root/'logs/speculation/conservative_draft_accept_20261009'/name
        path = source/'gated_best.pt'
        ck = torch.load(path, map_location='cpu', weights_only=True)
        exported = {k: ck[k] for k in ['kind', 'state_dict', 'model_config', 'selected_update']}
        exported['state_dict'] = {k: v.detach().cpu().contiguous() for k, v in exported['state_dict'].items()}
        dest = release/'weights'/f'{name}_gated.pt'
        torch.save(exported, dest)
        if dest.stat().st_size > 100*1024**2:
            raise RuntimeError('DO NOT git add: single exported file exceeds 100 MiB')
        thresholds = json.loads((root/'logs/speculation/relaxed_accept_sweep_20261009'/name/'thresholds.json').read_text())
        config['thresholds'][name] = thresholds['thresholds']['relax50']
        provenance[name] = {'source_relative': str(path.relative_to(root)), 'source_sha256': sha(path), 'export_sha256': sha(dest), 'bytes': dest.stat().st_size, 'selected_update': ck['selected_update'], 'training': json.loads((source/'gated_training.json').read_text()), 'calibration': thresholds, 'optimizer_included': False}
    (release/'configs').mkdir(exist_ok=True)
    for name, value in [('protocol', config), ('assets', assets), ('provenance', provenance)]:
        (release/'configs'/f'{name}.json').write_text(json.dumps(value, indent=2)+'\n')
    print(json.dumps({k: {'bytes': v['bytes'], 'sha256': v['export_sha256']} for k, v in provenance.items()}, indent=2))


if __name__ == '__main__':
    main()
