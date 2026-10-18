"""Download public DANDI assets by asset id with size verification and resume-safe skipping.

Usage: python download_dandi.py <assets.json> <output_dir> [path_substring ...]
The assets JSON is the DANDI API listing (results[].asset_id, path, size).
"""
import json
import shutil
import sys
import urllib.request
from pathlib import Path

API = 'https://api.dandiarchive.org/api/assets/{}/download/'


def main(listing, out, filters):
    data = json.loads(Path(listing).read_text())
    items = data['results'] if isinstance(data, dict) else data
    out = Path(out)
    for item in items:
        if filters and not any(f in item['path'] for f in filters):
            continue
        target = out / item['path']
        if target.exists() and target.stat().st_size == item['size']:
            print('present', item['path'], flush=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        part = target.with_suffix(target.suffix + '.part')
        with urllib.request.urlopen(API.format(item['asset_id']), timeout=120) as r, open(part, 'wb') as f:
            shutil.copyfileobj(r, f, length=8 << 20)
        if part.stat().st_size != item['size']:
            raise IOError(f"size mismatch for {item['path']}")
        part.replace(target)
        print('downloaded', item['path'], item['size'], flush=True)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3:])
