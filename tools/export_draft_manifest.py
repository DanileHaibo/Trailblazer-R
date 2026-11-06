"""Regenerate the portable draft checksum inventory from release weights."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    files = []
    for model in ('worldvla', 'rynnvla'):
        for p in sorted((ROOT/'weights'/model).glob('*.pt')):
            files.append(dict(path=str(p.relative_to(ROOT)), bytes=p.stat().st_size,
                              sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    if len(files) != 16:
        raise RuntimeError(f'Expected 16 draft files, found {len(files)}')
    (ROOT/'configs/draft_weights.json').write_text(json.dumps(dict(
        base_weights_included=False, total_bytes=sum(x['bytes'] for x in files), files=files), indent=2)+'\n')


if __name__ == '__main__':
    main()
