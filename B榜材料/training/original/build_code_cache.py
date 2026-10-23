# -*- coding: utf-8 -*-
"""构建码级 7 类缓存：复用已有 x(16通道)，重算 y7(7类，保留码24/26) + season
类别顺序 [玉米(1), 棉花(2), 水稻(3), 大豆(5), 冬小麦(24), 双季(26), 其他]
左侧 5 类作物面积由码级面积经官方季节映射得到：
  大豆 = c5 + (1-s)*c26 ;  冬小麦 = s*(c24 + c26)   (s=1 麦季)
"""
import json, re, csv, sys, time, os
import numpy as np, tifffile, torch
from pathlib import Path
from torch.nn import functional as F
from concurrent.futures import ProcessPoolExecutor
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
R = Path(os.environ.get('XINGHE_WORK_ROOT', '.'))
dec = lambda s: re.sub(r'#U([0-9a-fA-F]{4})', lambda m: chr(int(m[1], 16)), s)
CODES = [1, 2, 3, 5, 24, 26]          # -> 7 类（含其它）
CROPS = ['水稻', '玉米', '棉花', '大豆', '冬小麦']

def build_season(split, ds):
    seas = {}
    with (R / 'seg_data/MASK_逐影像类别解释.csv').open(encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            if row['数据集'] == ds:
                seas[Path(row['光学影像']).name] = 1 if row['码24当期类别'] == '冬小麦' else 0
    return seas

def job(args):
    rec, seas, src_cache, dst = args
    torch.set_num_threads(1)
    out = Path(dst) / (rec['key'] + '.npz')
    if out.exists(): return rec['key']
    with np.load(Path(src_cache) / (rec['key'] + '.npz')) as z:
        x = z['x'].astype(np.float16); areas = z['areas'].astype(np.float32)
    mask = tifffile.imread(rec['mask'])
    a = tifffile.imread(rec['opt'])
    if a.shape[0] == 12: a = np.moveaxis(a, 0, -1)
    valid = np.all(np.isfinite(a) & (a > 0), -1)
    lab = np.full(mask.shape, 6, np.int64)          # 6 = 其它/背景
    for i, code in enumerate(CODES): lab[mask == code] = i
    y7 = np.stack([(lab == k) & valid for k in range(7)]).astype(np.float32)
    # 自检：按官方季节映射从码级重算五类面积，应与缓存里的 areas 一致
    c = y7.reshape(7, -1).sum(1) * 0.0001
    s = seas[Path(rec['opt']).name]
    mapped = np.array([c[2], c[0], c[1], c[3] + (1 - s) * c[5], s * (c[4] + c[5])], np.float32)
    ok = np.allclose(mapped, areas, atol=2e-3)
    y7 = F.interpolate(torch.from_numpy(y7)[None], (256, 256), mode='area')[0].numpy().astype(np.float16)
    np.savez(out, x=x, y7=y7, areas=areas, mapped=mapped, season=np.float32(s), ok=np.uint8(ok))
    return rec['key']

def run(split, ds, src, dst):
    m = json.loads((R / 'area_qa_patch_v2/manifest.json').read_text())
    seas = build_season(split, ds)
    dst = Path(dst); dst.mkdir(parents=True, exist_ok=True)
    jobs = [(r, seas, src, str(dst)) for r in m[split]]
    t = time.time()
    with ProcessPoolExecutor(max_workers=8) as pool:
        for i, _ in enumerate(pool.map(job, jobs, chunksize=8), 1):
            if i % 500 == 0: print('  %s %d/%d %.0fs' % (split, i, len(jobs), time.time() - t), flush=True)
    print('%s 完成 %d 个 %.0fs' % (split, len(jobs), time.time() - t), flush=True)

if __name__ == '__main__':
    print('CODES =', CODES, '-> 7 类（末位为其它）', flush=True)
    run('train', '训练集', R / 'area_qa_patch_v2/cache_train', R / 'code_cache/cache_train')
    run('val', '验证集', R / 'area_qa_patch_v2/cache_val', R / 'code_cache/cache_val')
    # 汇总自检
    for sp in ['train', 'val']:
        fs = sorted((R / 'code_cache' / ('cache_' + sp)).glob('*.npz'))
        bad = 0; seas = 0
        for f in fs[::max(1, len(fs) // 200)]:
            with np.load(f) as z:
                if not z['ok']: bad += 1
                seas += float(z['season'])
        print('%s: 抽样 %d, 映射不一致 %d, 麦季占比 %.3f' % (sp, len(fs[::max(1, len(fs)//200)]), bad, seas / len(fs[::max(1, len(fs)//200)])))
