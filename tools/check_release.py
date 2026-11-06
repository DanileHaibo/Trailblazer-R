"""CPU-only publication checks; does not assert GPU numerical parity."""
import ast
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    tracked = subprocess.check_output(['git', 'ls-files', '-z'], cwd=ROOT).decode().split('\0')
    for name in filter(None, tracked):
        path = ROOT / name
        if not path.exists():
            continue
        assert path.stat().st_size < 100 * 1024**2, f'GitHub size limit: {name}'
        assert not path.is_symlink(), f'Nonportable symlink: {name}'
        if path.suffix == '.py':
            text = path.read_text()
            ast.parse(text, filename=name)
            if name != 'tools/check_release.py':
                assert '/home/xinhchen2' not in text and '/data2/' not in text and '/data3/' not in text, f'Host path: {name}'
        if path.suffix == '.json':
            json.loads(path.read_text())
    manifest = json.loads((ROOT / 'configs/draft_weights.json').read_text())
    for item in manifest['files']:
        path = ROOT / item['path']
        assert path.stat().st_size == item['bytes'], path
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item['sha256'], path
    assert len(manifest['files']) == 16, 'Expected World 4 checkpoints + Rynn 12 heads'
    print('PASS: tracked source syntax, JSON, host paths, file sizes and all 16 draft hashes')


if __name__ == '__main__':
    main()
