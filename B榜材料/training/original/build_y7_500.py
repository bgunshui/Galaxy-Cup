# -*- coding: utf-8 -*-
"""构建 500² 的码级分割标签（官方 MASK 1000² 的精确 2 倍抽取）

动机：当前分割监督只有 256²（39 米/像素），而官方 MASK 是 1000²（10 米/像素）
—— 我们只用了 1/15 的标签信息。测试集 T3 由小面积题主导（1-2 km² 档得分仅 0.193），
而小面积精度直接受监督分辨率限制。

做法：MASK[::2,::2] -> 500²（20 米/像素，精确抽取、无重采样模糊）
      有效像元掩膜同样抽取；码图存 uint8（每张 250KB，5029 张约 1.3 GB）
面积标签仍沿用已有的官方口径 areas（1000² 精度），此处只补分割监督。
"""
import csv, json, sys, time, os
from pathlib import Path
import numpy as np, tifffile
from concurrent.futures import ProcessPoolExecutor
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
R = Path(os.environ.get('XINGHE_WORK_ROOT', '.'))
OUT = Path(os.environ.get('XINGHE_Y7_OUT', str(R / 'y7_500')))
dec = lambda s: __import__('re').sub(r'#U([0-9a-fA-F]{4})', lambda m: chr(int(m[1], 16)), s)

def job(args):
    split, rec = args
    dst = OUT / split
    dst.mkdir(parents=True, exist_ok=True)
    out = dst / (rec['key'] + '.npz')
    if out.exists(): return 'skip'
    mask = tifffile.imread(rec['mask'])
    a = tifffile.imread(rec['opt'])
    if a.shape[0] == 12: a = np.moveaxis(a, 0, -1)
    valid = np.all(np.isfinite(a) & (a > 0), -1)
    cd = mask[::2, ::2].astype(np.uint8)                 # 500²
    va = valid[::2, ::2].astype(np.uint8)                # 500²
    # 截到 500（1000 可能不是偶数）
    cd = cd[:500, :500]; va = va[:500, :500]
    np.savez_compressed(out, cdl=cd, valid=va)
    return 'ok'

def main():
    man = json.loads((R / 'area_qa_patch_v2/manifest.json').read_text())
    jobs = [('train', r) for r in man['train']] + [('val', r) for r in man['val']]
    print('待处理 %d 张（train %d / val %d）' % (len(jobs), len(man['train']), len(man['val'])), flush=True)
    t0 = time.time(); n = 0
    with ProcessPoolExecutor(max_workers=8) as pool:
        for i, st in enumerate(pool.map(job, jobs, chunksize=16), 1):
            if st == 'ok': n += 1
            if i % 500 == 0: print('  %d/%d  %.0fs' % (i, len(jobs), time.time()-t0), flush=True)
    sz = sum(p.stat().st_size for p in OUT.rglob('*.npz'))
    print('完成 %d 张，用时 %.0f 分钟，占用 %.1f GB' % (n, (time.time()-t0)/60, sz/1e9), flush=True)

if __name__ == '__main__':
    main()
