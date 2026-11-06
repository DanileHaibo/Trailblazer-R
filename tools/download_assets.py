"""Download pinned foundation assets, never draft weights or benchmark results."""
import argparse
from pathlib import Path

SUITES = ('libero_spatial', 'libero_object', 'libero_goal', 'libero_10')
WORLD = ('Alibaba-DAMO-Academy/WorldVLA', '5952d61278d52608b2f141ef5b155d42e237165b')
RYNN = ('Alibaba-DAMO-Academy/RynnVLA-002', 'be44787bfdf010799129dbd3752637f5c52ca957')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True, choices=('worldvla', 'rynnvla'))
    p.add_argument('--suite', nargs='+', choices=SUITES, default=list(SUITES))
    p.add_argument('--output', type=Path, required=True, help='Asset root; contains worldvla/ and rynnvla/')
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()
    jobs = [(WORLD, a.output / 'worldvla', ['base_model/*', 'chameleon/tokenizer/*'])]
    if a.model == 'worldvla':
        jobs[0][2].extend(f'model_256/{s}/*' for s in a.suite)
    else:
        jobs.append((RYNN, a.output / 'rynnvla', [f'VLA_model_256/{s}/*' for s in a.suite]))
    for (repo, revision), dest, patterns in jobs:
        print(f'{repo}@{revision} -> {dest}: {patterns}')
        if not a.dry_run:
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id=repo, revision=revision, local_dir=str(dest), allow_patterns=patterns)


if __name__ == '__main__':
    main()
