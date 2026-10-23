"""Run the archived official-data preprocessing with a configurable root."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--work-root', type=Path, required=True)
    args = ap.parse_args()
    root = args.work_root.resolve()
    sys.path.insert(0, str(Path(__file__).resolve().parent / 'original'))
    import train_area_qa_patch_v2 as patch
    import build_code_cache as code
    import build_scl_cache as scl
    import build_y7_500 as highres
    patch.ROOT = code.R = scl.ROOT = highres.R = root
    highres.OUT = root / 'y7_500'
    stage = root / 'area_qa_patch_v2'
    stage.mkdir(exist_ok=True)
    train, trainmap, trainqa = patch.make_records('训练集')
    val, valmap, valqa = patch.make_records('验证集')
    manifest = {'train': train, 'val': val, 'train_map': trainmap,
                'val_map': valmap, 'train_source': trainqa, 'val_source': valqa,
                'train_pairs': len(train), 'val_pairs': len(val),
                'train_qa': len(trainmap), 'val_qa': len(valmap),
                'channels': '12 optical + 2 SAR + optical validity + SAR validity',
                'labels': patch.CROPS + ['background'], 'epochs': 16}
    (stage / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf8')
    for split, records, label in [('train', train, '训练集'), ('val', val, '验证集')]:
        folder = stage / ('cache_' + split)
        folder.mkdir(exist_ok=True)
        jobs = [(record, str(folder), 256) for record in records]
        with ProcessPoolExecutor(max_workers=4) as pool:
            for count, _ in enumerate(pool.map(patch.prepare_one, jobs), 1):
                if count % 100 == 0 or count == len(jobs):
                    print('CACHE', split, count, len(jobs), flush=True)
        code.run(split, label, folder, root / 'code_cache' / ('cache_' + split))
        scl.build_split(manifest, split, root / 'code_cache_scl')
    old_argv = sys.argv
    sys.argv = [str(Path(highres.__file__)), '--out', str(root / 'y7_500')]
    try:
        highres.main()
    finally:
        sys.argv = old_argv


if __name__ == '__main__':
    main()
