# -*- coding: utf-8 -*-
"""V4：在 V2 基础上补三件事（不改队友任何文件，只读他们的 code_cache_scl）
 1) batch size 提升：支持 micro-batch（分层组仍为 2 麦季 + 6 豆季，拆成若干 micro-batch）
 2) SCL 用法修正为「测试安全」：
      - 缺失表示 = 该通道置为训练均值（归一化后恰为 0，即"无信息"）
      - 训练时按概率 p 随机置为缺失；推理时测试集无 SCL，恒为缺失 —— 训练/推理一致
      - SCL 作辅助监督（云/阴影分割头，不依赖推理输入）
      - SCL 指导云遮挡增强（光学通道盖云，标签与有效性口径不变）
 3) 面积注入：把面积头测得的五类面积写入 prompt，让语言头据此出答案（端到端，通用性）
"""
import json
import os, sys, random
import numpy as np, torch, torch.nn as nn
from pathlib import Path
from torch.nn import functional as F
sys.path.insert(0, str(Path(__file__).resolve().parent))
import v2_lib as L
from v2_lib import (patchify, CROPS, PSZ, MERGE, TPSZ, NCODE, SEG, find_visual,
                    make_multispectral, Season, official_areas, qtype, score_from_areas,
                    answer_from_areas, t3score, t4_crop, t4_options, t5_crops,
                    t1_answer, t2_answer, t3_answer, t4_answer, t5_answer, W)
from train_area_qa_patch_v2 import Block as _ABlock

R = Path(os.environ.get('XINGHE_WORK_ROOT', '.'))
NC4 = 18                       # 16 + scl_norm + cloud_shadow
IDX_SCL_NORM, IDX_SCL_CLOUD = 16, 17


# ---------------- 数据 ----------------
class CodeDS4:
    """读队友构建的 code_cache_scl（18 通道），只读"""
    def __init__(self, split):
        self.m = json.loads((R / 'area_qa_patch_v2/manifest.json').read_text())
        self.dir = R / 'code_cache_scl' / ('cache_' + split)
        self.rows = [json.loads(l) for l in open(self.m[split + '_source'], encoding='utf8')]
        self.split = split

    def __len__(self): return len(self.rows)

    def load(self, i):
        r = self.rows[i]; key = self.m[self.split + '_map'][r['qa_id']]
        with np.load(self.dir / (key + '.npz')) as z:
            return (r, key, torch.from_numpy(z['x'].astype(np.float32)),
                    torch.from_numpy(z['y7'].astype(np.float32)),
                    torch.from_numpy(z['areas'].astype(np.float32)),
                    int(z['season']), torch.from_numpy(z['x'][IDX_SCL_CLOUD].astype(np.float32)))


def channel_stats4(n=400):
    ds = CodeDS4('train'); xs = []
    for i in range(0, len(ds), max(1, len(ds) // n)):
        xs.append(ds.load(i)[2].reshape(NC4, -1).mean(1))
    x = torch.stack(xs)
    return x.mean(0).tolist(), (x.std(0) + 1e-6).tolist()


def to_model_input4(x18, mean, std, scl_absent=False):
    """归一化；scl_absent=True 时把两个 SCL 通道置为训练均值 -> 归一化后恰为 0（无信息）"""
    x = x18.clone()
    if scl_absent:
        x[IDX_SCL_NORM] = mean[IDX_SCL_NORM]
        x[IDX_SCL_CLOUD] = mean[IDX_SCL_CLOUD]
    m = torch.tensor(mean, dtype=torch.float32).view(NC4, 1, 1)
    s = torch.tensor(std, dtype=torch.float32).view(NC4, 1, 1)
    return (x - m) / s * 0.577


def cloud_occlude(x18, rng, p=0.25, frac=(0.05, 0.25), bright=(0.35, 0.60)):
    """云遮挡增强：在光学通道盖一块亮而无纹理的"云"。
    只改通道 0..11；通道 14（光学有效性）不动，因此官方面积口径与标签完全不变。"""
    if rng.random() > p:
        return x18
    _, H, Wd = x18.shape
    f = rng.uniform(*frac)
    h = int(H * np.sqrt(f)); w = int(Wd * np.sqrt(f))
    y0 = rng.randrange(0, H - h + 1); x0 = rng.randrange(0, Wd - w + 1)
    val = rng.uniform(*bright)
    x18 = x18.clone()
    x18[:12, y0:y0 + h, x0:x0 + w] = val
    return x18


# ---------------- 模型：解码器加一个云/阴影辅助头 ----------------
# 解码器宽度倍数：面积精度完全由这个小网络承担（原版仅约 0.5M 参数，
# 而 8B VLM 只给出 16x16 的语义特征）。加大宽度是提高面积估计能力最直接的改动。
DEC_W = int(os.environ.get('DEC_W', '1'))
DEC_RES = int(os.environ.get('DEC_RES', '0'))    # >0：解码器在该分辨率上输出


class Dec4(nn.Module):
    def __init__(self, cin=1152, nout=NCODE, nin=NC4, w=None):
        super().__init__()
        w = w or DEC_W
        c1, c2, c3 = 24 * w, 48 * w, 96 * w
        self.a = _ABlock(nin, c1); self.b = _ABlock(c1, c2); self.c = _ABlock(c2, c3)
        self.d = _ABlock(c3 + c2, c2); self.e = _ABlock(c2 + c1, c1)
        self.out = nn.Conv2d(c1, nout, 1)
        self.proj = nn.Sequential(nn.Conv2d(cin, c3, 1), nn.GroupNorm(8, c3), nn.GELU())
        self.cloud = nn.Conv2d(c1, 1, 1)          # 辅助头：云/阴影

    def forward(self, f, x_raw, want_cloud=False):
        if DEC_RES and x_raw.shape[-1] != DEC_RES:
            # 上采样输入即可把整个解码器的输出分辨率提到 DEC_RES
            # （卷积权重与分辨率无关，V7 学到的权重可直接复用）
            x_raw = F.interpolate(x_raw, size=(DEC_RES, DEC_RES), mode='bilinear', align_corners=False)
        a = self.a(x_raw)
        b = self.b(F.avg_pool2d(a, 2))
        c = self.c(F.avg_pool2d(b, 2))
        c = c + self.proj(F.interpolate(f, size=c.shape[-2:], mode='bilinear', align_corners=False))
        d = self.d(torch.cat([F.interpolate(c, size=b.shape[-2:], mode='bilinear', align_corners=False), b], 1))
        e = self.e(torch.cat([F.interpolate(d, size=a.shape[-2:], mode='bilinear', align_corners=False), a], 1))
        return (self.out(e), self.cloud(e)) if want_cloud else self.out(e)


class UnifiedV4(nn.Module):
    def __init__(self, vlm, calib=1.0):
        super().__init__()
        self.vlm = vlm
        vis = self.visual = find_visual(vlm)
        if vis.patch_embed.proj.weight.shape[1] == 3:
            make_multispectral(vis, NC4, calib)
        else:
            # 已合并多光谱入口（merge_lora.py 产出）时不能再替换卷积
            vis.patch_embed.in_channels = NC4
            if hasattr(vis.config, 'in_channels'):
                vis.config.in_channels = NC4
        self.dec = Dec4(vis.config.hidden_size).float()
        self.season = Season(vis.config.hidden_size).float()
        self.feat = None
        vis.register_forward_hook(self._hook)

    def _hook(self, mod, inp, out):
        self.feat = out.last_hidden_state if hasattr(out, 'last_hidden_state') else out[0]

    def forward_vision(self, pixel_values, grid_thw):
        self.feat = None
        self.visual(pixel_values, grid_thw=grid_thw)
        return self.feat


# ---------------- 面积注入的 prompt ----------------
AREA_HINT = ('\n（系统已测得本影像五类目标农作物的种植面积，单位平方千米：'
             '水稻 {a[0]:.2f}、玉米 {a[1]:.2f}、棉花 {a[2]:.2f}、大豆 {a[3]:.2f}、冬小麦 {a[4]:.2f}。'
             '请据此按题目要求的 JSON 格式回答。）')


def build_batch4(processor, row, x18, mean, std, images=1, scl_absent=True, areas=None,
                 cloud_map=None, want_cloud=False):
    """areas 非空时把面积提示追加进 user 文本；返回 inputs 与 prompt 长度（以及可选的云目标）"""
    H = W = x18.shape[-1]
    gh = H // PSZ
    ntok = (gh * gh) // (MERGE * MERGE)
    hint = AREA_HINT.format(a=areas) if areas is not None else ''
    msgs, pmsgs = [], []
    for msg in row['messages']:
        if msg['role'] == 'user':
            txt = msg['content'].replace('<image>', '') + hint
            c = [{'type': 'image'}] * images + [{'type': 'text', 'text': txt}]
            msgs.append({'role': 'user', 'content': c})
            pmsgs.append({'role': 'user', 'content': c})
        elif msg['role'] == 'assistant':
            msgs.append(msg)
        else:
            msgs.append(msg); pmsgs.append(msg)
    full = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
    prm = processor.apply_chat_template(pmsgs, tokenize=False, add_generation_prompt=True)
    rep = '<|image_pad|>' * ntok
    full = full.replace('<|image_pad|>', rep); prm = prm.replace('<|image_pad|>', rep)
    tok = processor.tokenizer
    fi = tok(full, return_tensors='pt', add_special_tokens=False)['input_ids']
    pi = tok(prm, return_tensors='pt', add_special_tokens=False)['input_ids']
    prefix = pi.shape[1]
    labels = fi.clone(); labels[:, :prefix] = -100
    pv = patchify(to_model_input4(x18, mean, std, scl_absent).numpy()).to(torch.bfloat16)
    grid = torch.tensor([[1, gh, gh]] * images)
    pv = pv.repeat(images, 1)
    mm = (fi == processor.image_token_id).long()
    out = dict(input_ids=fi.cuda(), attention_mask=torch.ones_like(fi).cuda(),
               pixel_values=pv.cuda(), image_grid_thw=grid.cuda(),
               mm_token_type_ids=mm.cuda(), labels=labels.cuda())
    return out, prefix
