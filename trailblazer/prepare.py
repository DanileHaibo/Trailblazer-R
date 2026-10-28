"""Download official assets outside this repository; no token is serialized."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=['nora', 'pi0'], required=True)
    parser.add_argument('--suite', choices=['libero_object', 'libero_spatial', 'libero_goal', 'libero_10'], default='libero_object')
    parser.add_argument('--output', type=Path, required=True, help='Local asset manifest (outside the repository)')
    args = parser.parse_args()
    from huggingface_hub import snapshot_download
    specs = json.loads((ROOT/'configs/assets.json').read_text())[args.model]
    keys = ['fast', args.suite] if args.model == 'nora' else list(specs)
    paths = {}
    for key in keys:
        spec = specs[key]
        kwargs = {}
        if key == 'text_tokenizer':
            kwargs['allow_patterns'] = ['*.json', 'tokenizer.model', '*.txt']
        paths[key] = snapshot_download(repo_id=spec['repo'], revision=spec['revision'], **kwargs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        old = json.loads(args.output.read_text())
        old.update(paths); paths = old
    args.output.write_text(json.dumps(paths, indent=2)+'\n')
    print('Wrote local asset paths:', args.output)


if __name__ == '__main__':
    main()
