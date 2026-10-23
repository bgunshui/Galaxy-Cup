# -*- coding: utf-8 -*-
"""混合数据集：官方 code_cache_scl + 外部同源 patch

外部 patch 与队友 code_cache_scl 的 schema 完全一致（x 18 通道 / y7 / areas / season），
因此可以像官方样本一样直接被训练循环使用；只有 val 始终只用官方，保证测量诚实。
"""
import json
import os
from pathlib import Path
import random
import numpy as np
import torch
from pathlib import Path
from v4_lib import IDX_SCL_CLOUD


_AUG = os.environ.get('AUG_GEOM', '0') == '1'
DEC_RES = int(os.environ.get('DEC_RES', '0'))       # >0 时用高分辨率分割监督
CODES7 = [1, 2, 3, 5, 24, 26]
_Y7_500 = None


R = Path(os.environ.get('XINGHE_WORK_ROOT', '.'))


class CodeDSMix:
    def __init__(self, split, manifest_path, off_cache, ext_root):
        self.m = json.loads(Path(manifest_path).read_text())
        self.split = split
        self.dir = Path(off_cache) / ('cache_' + split)
        self.ext = Path(ext_root)
        self.rows = [json.loads(l) for l in open(self.m[split + '_source'], encoding='utf8')]
        self.map = self.m[split + '_map']
        self.ext_index = {}
        if self.ext.exists():
            for p in self.ext.rglob('*.npz'):
                self.ext_index[p.stem] = p
        # 注意：容器有 64GB 内存上限，绝不能缓存整份样本（x+y7 每行约 7MB）
        self._cache = {}
        n_ext = sum(1 for r in self.rows if r['qa_id'].startswith('ext_'))
        print('[CodeDSMix:%s] 题目 %d（其中外部 %d），官方缓存 %s，外部索引 %d'
              % (split, len(self.rows), n_ext, self.dir, len(self.ext_index)), flush=True)

    def __len__(self):
        return len(self.rows)

    def areas_of(self, i):
        """只读 areas/season（小成员），供训练脚本启动期遍历使用：快且几乎不占内存"""
        r = self.rows[i]
        key = self.map[r['qa_id']]
        with np.load(self._path(key)) as z:
            return z['areas'].astype(np.float32), int(z['season'])

    def _path(self, key):
        p = self.dir / (key + '.npz')
        if p.exists():
            return p
        if key in self.ext_index:
            return self.ext_index[key]
        raise FileNotFoundError('%s (cache=%s, ext=%d)' % (key, self.dir, len(self.ext_index)))

    def load(self, i):
        r = self.rows[i]
        key = self.map[r['qa_id']]
        with np.load(self._path(key)) as z:
            x = z['x'].astype(np.float32)
            xt = torch.from_numpy(x)
            y7 = torch.from_numpy(z['y7'].astype(np.float32))
            if _AUG:
                # 几何增强：4 个 90° 旋转 × 2 个翻转 = 免费 8 倍数据。
                # 农田无规范朝向，旋转翻转不改变面积标签与 GSD，语义安全。
                k = random.randrange(8)
                if k % 4:
                    xt = torch.rot90(xt, k % 4, dims=(1, 2))
                    y7 = torch.rot90(y7, k % 4, dims=(1, 2))
                if k >= 4:
                    xt = torch.flip(xt, dims=(2,))
                    y7 = torch.flip(y7, dims=(2,))
            if DEC_RES:
                # 高分辨率分割监督：读 500² 的码图（官方 MASK 的精确 2 倍抽取）
                import v9_lib as _me
                p5 = _me.R / 'y7_500' / self.split / (key + '.npz')
                with np.load(p5) as z5:
                    cd = torch.from_numpy(z5['cdl'].astype(np.int64))
                    va = (torch.from_numpy(z5['valid'].astype(np.float32)) > 0.5)
                in7 = torch.stack([(cd == c) for c in CODES7])
                y7hi = torch.cat([in7, (~in7.any(0))[None]], 0).float() * va[None].float()
                if _AUG:
                    k2 = random.randrange(8)
                    if k2 % 4:
                        y7hi = torch.rot90(y7hi, k2 % 4, dims=(1, 2))
                    if k2 >= 4:
                        y7hi = torch.flip(y7hi, dims=(2,))
                y7 = y7hi
            return (r, key, xt, y7,
                    torch.from_numpy(z['areas'].astype(np.float32)),
                    int(z['season']), xt[IDX_SCL_CLOUD].float())
