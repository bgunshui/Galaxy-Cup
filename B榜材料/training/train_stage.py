"""Portable launcher for the actual v973 decoder fine-tuning stage."""
import argparse
import os
from pathlib import Path
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--work-root', type=Path, required=True)
    ap.add_argument('--model', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    here = Path(__file__).resolve().parent
    work = args.work_root.resolve()
    model = args.model.resolve()
    source = here / 'original'
    manifest = work / 'area_qa_patch_v2/manifest.json'
    if not manifest.is_file():
        raise FileNotFoundError('Prepare the official dataset/cache first: ' + str(manifest))
    os.environ.update({
        'DEC_RES': '500', 'AUG_GEOM': '0',
        'XINGHE_WORK_ROOT': str(work),
        'XINGHE_BASE_MODEL': str(model / 'base'),
        'XINGHE_TRAIN_OUTPUT': str(args.output.resolve()),
        'TRAIN_POOL_KEYS': str(source / 'v972_train_pool.json'),
        'VAL_KEYS': str(source / 'v972_val_keys.json'),
        'EVAL_KEYS': str(source / 'v972_val_keys.json'),
    })
    sys.path.insert(0, str(source))
    import v2_lib
    import v4_lib
    import v9_lib
    import v17_train
    for module in (v2_lib, v4_lib, v9_lib):
        module.R = work
    v17_train.base = str(model / 'base')
    v17_train.CodeDS4 = lambda split: v9_lib.CodeDSMix(
        split, manifest, work / 'code_cache_scl', work / 'unused_external_cache')
    sys.argv = [str(source / 'v18_train.py'),
        '--steps', '400', '--micro', '32', '--group', '32',
        '--reweight', '.5', '--teacher', '', '--lr_head', '5e-5',
        '--resume_adapter', str(model / 'v973_adapter'),
        '--resume_heads', str(here / 'initial/best_task_heads.pt'),
        '--out', str(args.output.resolve()), '--eval_every', '200',
        '--keep_every', '200', '--no_lm', '--freeze_season']
    v17_train.main()


if __name__ == '__main__':
    main()
