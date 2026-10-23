# -*- coding: utf-8 -*-
"""V2 模型库：16 通道多光谱 patch embedding + 码级分割解码器 + 生长季头 + 官方面积映射
关键点
  1) patch_embed 从 Conv3d(3) 扩到 Conv3d(16)，按 3/16 缩放初始化以保持初始响应量级一致
  2) 直接读视觉编码器 last_hidden_state（合并前，16x16@256²），在其上解码，空间分辨率远高于挂 LLM token
  3) 分割输出码级 7 类 {1,2,3,5,24,26,其他}，再按官方口径映射成五类作物面积
     —— 互斥划分自动满足 Σ≤有效面积；码 26 只归一类；生长季是唯一隐变量
  4) 单个 16 通道"超图"，token 数与 3 通道 RGB 完全相同（64@256²），且不存在单/双影像不一致
"""
import json, re, csv, sys
import numpy as np, torch, torch.nn as nn
from pathlib import Path
from torch.nn import functional as F

R = Path(os.environ.get('XINGHE_WORK_ROOT', '.'))
CROPS = ['水稻', '玉米', '棉花', '大豆', '冬小麦']
NC, MERGE, PSZ, TPSZ = 16, 2, 16, 2
NCODE = 7
SEG = 256                     # 分割输出分辨率（与输入同分辨率，每格 0.001526 km²）

# ---------------- patchify（与 HF Qwen2VLImageProcessor.patchify 一致，已单元测试） -------------
def patchify(arr_chw: np.ndarray) -> torch.Tensor:
    C, H, W = arr_chw.shape
    gh, gw = H // PSZ, W // PSZ
    a = torch.from_numpy(np.ascontiguousarray(arr_chw, dtype=np.float32))
    a = a.reshape(C, gh // MERGE, MERGE, PSZ, gw // MERGE, MERGE, PSZ)
    a = a.permute(1, 4, 2, 5, 0, 3, 6)                       # (gh/m, gw/m, m, m, C, P, P)
    a = a.unsqueeze(5).expand(-1, -1, -1, -1, -1, TPSZ, -1, -1)
    return a.reshape(gh * gw, C * TPSZ * PSZ * PSZ)

# ---------------- 16 通道 patch embedding 替换 ----------------
def make_multispectral(vision, n_in=NC, calib=1.0):
    """把 Conv3d(3) 换成 Conv3d(n_in)。权重按 3/n_in 缩放，使"所有通道相同"时初始响应与
    3 通道一致；再乘 calib 做经验标定。新卷积保持 fp32（避免 bf16 参数更新丢精度），
    前向内部算 fp32 再转回原 dtype。"""
    pe = vision.patch_embed
    old = pe.proj
    assert old.weight.shape[1] == 3, old.weight.shape
    new = nn.Conv3d(n_in, old.out_channels, old.kernel_size, old.stride, old.padding, bias=True).float().to(old.weight.device)
    with torch.no_grad():
        for c in range(n_in):
            new.weight[:, c] = old.weight[:, c % 3].float() * (3.0 / n_in) * calib
        new.bias.copy_(old.bias.float())
    pe.proj = new
    pe.in_channels = n_in
    vision.config.in_channels = n_in
    psz, tpsz, dim = pe.patch_size, pe.temporal_patch_size, pe.embed_dim

    def fwd(hidden_states, _pe=pe):
        dt = hidden_states.dtype
        if dt not in (torch.bfloat16, torch.float16):
            dt = torch.bfloat16                      # 视觉块权重是 bf16，必须回低精度
        h = hidden_states.view(-1, n_in, tpsz, psz, psz)
        return _pe.proj(h.to(dtype=_pe.proj.weight.dtype)).view(-1, dim).to(dt)
    pe.forward = fwd
    return new

# ---------------- 解码器：细节分支用我们已训练的面积 CNN 结构初始化 ----------------
from train_area_qa_patch_v2 import Block as _ABlock

class Dec(nn.Module):
    """单一模型内的多尺度解码器：
       - 细节分支：16 通道 -> 5 个 Block 的 U-Net（结构与权重来自我们自己的面积 CNN）
       - 语义注入：视觉主干特征经 1x1 投影后加到最深一层
       输出 7 类码级分割（末层重新初始化，因为类别定义从 5 作物改为 6 码 + 背景）"""
    def __init__(self, cin=1152, nout=NCODE, nin=NC):
        super().__init__()
        self.a = _ABlock(nin, 24); self.b = _ABlock(24, 48); self.c = _ABlock(48, 96)
        self.d = _ABlock(96 + 48, 48); self.e = _ABlock(48 + 24, 24)
        self.out = nn.Conv2d(24, nout, 1)
        self.proj = nn.Sequential(nn.Conv2d(cin, 96, 1), nn.GroupNorm(8, 96), nn.GELU())
    def forward(self, f, x_raw):
        a = self.a(x_raw)
        b = self.b(F.avg_pool2d(a, 2))
        c = self.c(F.avg_pool2d(b, 2))
        c = c + self.proj(F.interpolate(f, size=c.shape[-2:], mode='bilinear', align_corners=False))
        d = self.d(torch.cat([F.interpolate(c, size=b.shape[-2:], mode='bilinear', align_corners=False), b], 1))
        e = self.e(torch.cat([F.interpolate(d, size=a.shape[-2:], mode='bilinear', align_corners=False), a], 1))
        return self.out(e)

def init_dec_from_areamodel(dec, ckpt=None):
    if ckpt is None:
        ckpt = str(R / 'area_qa_patch_v2/best.pt')
    """用我们已训练的面积 CNN 权重初始化细节分支（同一份官方数据训得，非外部模型）"""
    sd = torch.load(ckpt, map_location='cpu')['model']
    own = dec.state_dict()
    n = 0
    for k, v in sd.items():
        if k.startswith('out.'):
            continue                       # 末层类别数不同，保持随机初始化
        if k in own and own[k].shape == v.shape:
            own[k] = v; n += 1
    dec.load_state_dict(own)
    return n

class Season(nn.Module):
    def __init__(self, cin=1152):
        super().__init__()
        self.f = nn.Sequential(nn.LayerNorm(cin), nn.Linear(cin, 256), nn.GELU(), nn.Linear(256, 2))
    def forward(self, f):                       # f: (B, cin, gh, gw)
        return self.f(f.mean((2, 3)))

# ---------------- 模型 ----------------
def find_visual(vlm):
    """兼容裸模型与 PEFT 包装：返回视觉主干模块"""
    for path in ['model.visual', 'model.model.visual', 'visual',
                 'base_model.model.model.visual', 'base_model.model.visual']:
        o = vlm
        try:
            for part in path.split('.'):
                o = getattr(o, part)
            return o
        except AttributeError:
            continue
    raise AttributeError('找不到视觉主干')

class UnifiedV2(nn.Module):
    def __init__(self, vlm, calib=1.0):
        super().__init__()
        self.vlm = vlm
        vis = self.visual = find_visual(vlm)
        self.vis_hidden = vis.config.hidden_size
        make_multispectral(vis, NC, calib)
        self.dec = Dec(self.vis_hidden).float()
        self.season = Season(self.vis_hidden).float()
        self.feat = None
        vis.register_forward_hook(self._hook)
    def _hook(self, mod, inp, out):
        self.feat = out.last_hidden_state if hasattr(out, 'last_hidden_state') else out[0]
    def forward_vision(self, pixel_values, grid_thw):
        self.feat = None
        self.visual(pixel_values, grid_thw=grid_thw)   # 第一个位置参数是 hidden_states
        return self.feat                            # (sum patches, hidden)

def code_areas(logits, valid):
    """logits:(B,7,H,W) valid:(B,1,H,W) 0/1 -> (B,7) 码级面积 km²
    用输入自带的有效性通道（光学全波段>0），测试时可得，不依赖真值"""
    return (logits.softmax(1) * valid).mean((2, 3)) * 100.0

def official_areas(c: torch.Tensor, s_logits: torch.Tensor):
    """c: (7,) 码级面积(km²)  s_logits: (2,) 生长季 logits -> 五类作物面积 (5,)"""
    p_wheat = s_logits.softmax(-1)[1]              # 麦季概率
    return torch.stack([c[2], c[0], c[1],
                        c[3] + (1 - p_wheat) * c[5],
                        p_wheat * (c[4] + c[5])])

# ---------------- 数据 ----------------
class CodeDS:
    def __init__(self, split):
        self.m = json.loads((R / 'area_qa_patch_v2/manifest.json').read_text())
        self.dir = R / 'code_cache' / ('cache_' + split)
        self.rows = [json.loads(l) for l in open(self.m[split + '_source'], encoding='utf8')]
        self.split = split
    def __len__(self): return len(self.rows)
    def load(self, i):
        r = self.rows[i]; key = self.m[self.split + '_map'][r['qa_id']]
        with np.load(self.dir / (key + '.npz')) as z:
            return (r, key, torch.from_numpy(z['x'].astype(np.float32)),
                    torch.from_numpy(z['y7'].astype(np.float32)),
                    torch.from_numpy(z['areas'].astype(np.float32)),
                    int(z['season']))

# ---------------- 归一化统计 ----------------
def channel_stats(n=400, split='train'):
    ds = CodeDS(split); xs = []
    step = max(1, len(ds) // n)
    for i in range(0, len(ds), step):
        xs.append(ds.load(i)[2].reshape(NC, -1).mean(1))
    x = torch.stack(xs)
    return x.mean(0).tolist(), (x.std(0) + 1e-6).tolist()

def to_model_input(x16, mean, std):
    """x16: (16,H,W) -> 归一化到 patch_embed 期望的统计（≈零均值、std≈0.577）"""
    m = torch.tensor(mean).view(NC, 1, 1); s = torch.tensor(std).view(NC, 1, 1)
    return ((x16 - m) / s * 0.577)

# ---------------- 文本 + 输入构造 ----------------
def build_batch(processor, row, x16, mean, std, images=1):
    """返回 inputs(dict) 与 prompt 长度；labels 掩掉 prompt"""
    H = W = x16.shape[-1]
    gh = gw = H // PSZ
    ntok = (gh * gw) // (MERGE * MERGE)
    msgs, prompt_msgs = [], []
    for msg in row['messages']:
        if msg['role'] == 'user':
            txt = msg['content'].replace('<image>', '')
            c = [{'type': 'image'}] * images + [{'type': 'text', 'text': txt}]
            msgs.append({'role': 'user', 'content': c})
            prompt_msgs.append({'role': 'user', 'content': c})
        elif msg['role'] == 'assistant':
            msgs.append(msg)
        else:
            msgs.append(msg); prompt_msgs.append(msg)
    full = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
    prm = processor.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True)
    rep = '<|image_pad|>' * ntok
    full = full.replace('<|image_pad|>', rep); prm = prm.replace('<|image_pad|>', rep)
    tok = processor.tokenizer
    fi = tok(full, return_tensors='pt', add_special_tokens=False)['input_ids']
    pi = tok(prm, return_tensors='pt', add_special_tokens=False)['input_ids']
    prefix = pi.shape[1]
    labels = fi.clone(); labels[:, :prefix] = -100
    xn = to_model_input(x16, mean, std)
    pv = patchify(xn.numpy()).to(torch.bfloat16)     # (gh*gw, 16*T*P*P)，与视觉块权重同精度
    grid = torch.tensor([[1, gh, gw]] * images)
    pv = pv.repeat(images, 1)
    mm = (fi == processor.image_token_id).long()      # mm_token_type_ids: image_pad 处为 1
    return dict(input_ids=fi.cuda(), attention_mask=torch.ones_like(fi).cuda(),
                pixel_values=pv.cuda(), image_grid_thw=grid.cuda(),
                mm_token_type_ids=mm.cuda(), labels=labels.cuda()), prefix

# ---------------- 官方规则层（训练打分与推理出答案共用同一份） ----------------
import re as _re

def t3score(x, y):
    e = abs(x - y) / y if y > 0 else (0 if x == 0 else float('inf'))
    return 1.0 if e <= .05 else .8 if e <= .1 else .6 if e <= .2 else .4 if e <= .3 else 0.0

def t5_crops(q):
    nm = _re.findall('|'.join(CROPS), q.split('当两种')[0])
    return list(dict.fromkeys(nm))[:2]

def t4_crop(q):
    return _re.findall('|'.join(CROPS), q.split('A.')[0])[0]

def t4_options(q):
    return _re.findall(r'(?m)^\s*([A-Z])[.、．]\s*(.+)$', q)

def qtype(gt_or_q, q=None):
    if q is None:                                    # 测试集：只有题干
        s = gt_or_q
        if '估算' in s and '面积' in s: return 'T3'
        if '比较' in s: return 'T5'
        if '最大' in s: return 'T1'
        if '包含' in s or '哪些' in s: return 'T2'
        return 'T4'
    gt = gt_or_q
    if 'area_km2' in gt: return 'T3'
    if 'crops' in gt: return 'T2'
    if 'crop' in gt: return 'T1'
    return 'T5' if '比较' in q else 'T4'

def t1_answer(a): return {'crop': CROPS[int(np.argmax(a))]}
def t2_answer(a): return {'crops': [c for c, x in zip(CROPS, a) if x >= 1.0]}
def t3_answer(q, a): return {'crop': t4_crop(q), 'area_km2': round(float(a[CROPS.index(t4_crop(q))]), 2)}
def t4_answer(q, a):
    x = float(a[CROPS.index(t4_crop(q))]); fb = ''; hits = []
    for L_, txt in t4_options(q):
        if '以上都不是' in txt: fb = L_; continue
        nums = [float(v) for v in _re.findall(r'\d+(?:\.\d+)?', txt)]
        if '最接近' in q: ok = bool(nums) and abs(x - nums[0]) <= 1
        elif len(nums) >= 2: ok = nums[0] <= x < nums[1]
        elif len(nums) == 1: ok = (x < nums[0]) if '小于' in txt else (x >= nums[0])
        else: ok = False
        if ok: hits.append(L_)
    return {'choice': hits[0] if len(hits) == 1 else (fb or 'D')}
def t5_answer(q, a):
    c1, c2 = t5_crops(q); v1, v2 = float(a[CROPS.index(c1)]), float(a[CROPS.index(c2)])
    return {'choice': 'C' if abs(v1 - v2) <= 5.0 else ('A' if v1 > v2 else 'B')}

def answer_from_areas(t, q, a):
    return {'T1': t1_answer, 'T2': t2_answer, 'T3': t3_answer, 'T4': t4_answer,
            'T5': t5_answer}[t](q, a) if t in ('T3', 'T4', 'T5') else \
           ({'T1': t1_answer, 'T2': t2_answer}[t](a))

W = dict(T1=.25, T2=.15, T3=.30, T4=.20, T5=.10)
def score_from_areas(rows, areas_by_key):
    """rows: [(qa_id, key, t, q, gt)] ; areas_by_key: key -> (5,) np"""
    g = {}
    for qa, key, t, q, gt in rows:
        a = areas_by_key[key]
        p = answer_from_areas(t, q, a)
        if t == 'T1': s = float(p['crop'] == gt['crop'])
        elif t == 'T2':
            P, G = set(p['crops']), set(gt['crops'])
            s = 1.0 if not P and not G else (2 * len(P & G) / (len(P) + len(G)) if P or G else 0.0)
        elif t == 'T3': s = t3score(p['area_km2'], float(gt['area_km2']))
        else: s = float(p['choice'] == gt['choice'])
        g.setdefault(t, []).append(s)
    means = {k: float(np.mean(v)) for k, v in g.items()}
    return means, sum(W[k] * means[k] for k in means)
