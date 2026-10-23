# -*- coding: utf-8 -*-
"""V4 训练：micro-batch + SCL(测试安全) + 云遮挡增强 + 面积注入
每 step：分层取 8 样本（2 麦季+6 豆季），拆成 micro-batch
  pass1 视觉主干+解码器 -> 分割/面积/生长季/云 四项损失，并给出面积(detach)
  pass2 完整前向（带面积提示文本）-> L_json
验证两种模式：SCL 存在（可比队友 V3）与 SCL 缺失（= 测试条件）
"""
import json, sys, time, random, argparse
import numpy as np, torch
from pathlib import Path
from torch.nn import functional as F
sys.path.insert(0, str(Path(__file__).resolve().parent))
import v2_lib as L
import v4_lib as L4
from v4_lib import (CodeDS4, UnifiedV4, channel_stats4, to_model_input4, cloud_occlude,
                    AREA_HINT, NC4, IDX_SCL_NORM, IDX_SCL_CLOUD)
from v2_lib import patchify, official_areas, score_from_areas, qtype, code_areas, MERGE, PSZ
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from peft import PeftModel

base = os.environ.get('XINGHE_BASE_MODEL', 'weights/base')
OUT = Path(os.environ.get('XINGHE_TRAIN_OUTPUT', 'runs/v4_run'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=2500)
    ap.add_argument('--micro', type=int, default=4)
    ap.add_argument('--group', type=int, default=8)
    ap.add_argument('--calib', type=float, default=0.9357)
    ap.add_argument('--scl_drop', type=float, default=0.5)
    ap.add_argument('--cloud_aug', type=float, default=0.25)
    ap.add_argument('--w_cloud', type=float, default=0.1)
    ap.add_argument('--w_area', type=float, default=1.0)
    ap.add_argument('--w_distill', type=float, default=0.15)
    ap.add_argument('--w_season', type=float, default=0.2)
    ap.add_argument('--lr_head', type=float, default=2e-4)
    ap.add_argument('--lr_patch', type=float, default=1e-4)
    ap.add_argument('--lr_lora', type=float, default=1e-5)
    ap.add_argument('--eval_every', type=int, default=250)
    ap.add_argument('--keep_every', type=int, default=500, help='每多少步留档一个检查点（防止 task 峰值丢失）')
    ap.add_argument('--teacher', type=str, default='')
    ap.add_argument('--resume_adapter', type=str, default='weights/v973_adapter')
    ap.add_argument('--resume_heads', type=str, default='training/initial/best_task_heads.pt')
    ap.add_argument('--no_inject', action='store_true')
    ap.add_argument('--reweight', type=float, default=0.0, help='按面积重要性重加权的强度 alpha')
    ap.add_argument('--crop_balance', type=float, default=0.0,
                    help='按作物稀有度加权采样的强度（0=关闭；0.5~1.0 开启，让水稻等稀有作物被采到的频率接近常见作物）')
    ap.add_argument('--out', type=str, default='runs/v4_run')
    ap.add_argument('--no_lm', action='store_true', help='冻结LoRA/跳过语言前向（仅训解码器+季节头，快3-4倍）')
    ap.add_argument('--freeze_season', action='store_true', help='同时冻结季节头（避免退化）')
    ap.add_argument('--w_sub', type=float, default=0.0, help='子窗口面积损失权重（0=关闭）')
    ap.add_argument('--sub_scale', type=float, default=0.5, help='子窗口边长占比（0.5=半个边长=25km²）')
    ap.add_argument('--sub_n', type=int, default=2, help='每步随机子窗口数')
    a = ap.parse_args()
    global OUT
    OUT = Path(a.out)
    torch.set_num_threads(4); random.seed(41); np.random.seed(41); torch.manual_seed(41)
    OUT.mkdir(parents=True, exist_ok=True)
    print('=== V4 steps=%d micro=%d group=%d scl_drop=%.2f cloud_aug=%.2f inject=%s ==='
          % (a.steps, a.micro, a.group, a.scl_drop, a.cloud_aug, not a.no_inject), flush=True)

    import os as _os
    SIZE_IN = 256

    class MixedDS:
        """把多个（split 限定的）数据集拼成一个，供 train+val 混合训练"""

        def __init__(self, parts):
            self.parts = parts
            self.offs = []
            n = 0
            for p in parts:
                self.offs.append(n); n += len(p)
            self.rows = [r for p in parts for r in p.rows]
            self.map = {}
            for p in parts:
                self.map.update(p.map)
            self.m = parts[0].m

        def __len__(self):
            return len(self.rows)

        def _find(self, j):
            import bisect
            k = bisect.bisect_right(self.offs, j) - 1
            return self.parts[k], j - self.offs[k]

        def load(self, j):
            p, i = self._find(j)
            return p.load(i)

        def areas_of(self, j):
            p, i = self._find(j)
            return p.areas_of(i)

    class SubDS:
        """按数据集 key 过滤的包装（保持索引空间一致）"""

        def __init__(self, base, allowed):
            self.base = base
            self.m = base.m
            self.map = base.map
            if allowed is None:
                self.idx = list(range(len(base.rows)))
            else:
                self.idx = [i for i, r in enumerate(base.rows) if base.map[r['qa_id']] in allowed]
            self.rows = [base.rows[i] for i in self.idx]

        def __len__(self):
            return len(self.idx)

        def load(self, j):
            return self.base.load(self.idx[j])

        def areas_of(self, j):
            return self.base.areas_of(self.idx[j])

    def _keys(env):
        p = _os.environ.get(env, '')
        if p and _os.path.exists(p):
            return set(json.load(open(p)))
        return None
    KT, KE = _keys('TRAIN_KEYS'), _keys('EVAL_KEYS')
    KV, KP = _keys('VAL_KEYS'), _keys('TRAIN_POOL_KEYS')
    if KV is not None or KP is not None:            # 混合模式
        parts = []
        if KV is not None:
            parts.append(SubDS(CodeDS4('val'), KV))
        if KP is not None:
            parts.append(SubDS(CodeDS4('train'), KP))
        tr = MixedDS(parts) if len(parts) > 1 else parts[0]
        n_val = len(parts[0]) if KV is not None else 0
        n_train = len(parts[1]) if len(parts) > 1 and KP is not None else 0
        print('*** 混合训练：val 侧瓦片 %d / train 侧瓦片 %d ***' % (n_val, n_train), flush=True)
    else:
        base_tr = CodeDS4('val')
        tr = SubDS(base_tr, KT)
        print('*** 单分布训练：瓦片 %d ***' % len(tr), flush=True)
    ev_split = _os.environ.get('EVAL_SPLIT', 'val')
    va = SubDS(CodeDS4(ev_split), KE)
    print('*** 评估 split=%s：留出瓦片 %d ***' % (ev_split, len(va)), flush=True)
    # 启动期只遍历 areas/season：若数据集提供快路径则用之（容器仅 64GB 内存，不可全量缓存）
    def _A(i):
        return tr.areas_of(i)[0] if hasattr(tr, 'areas_of') else tr.load(i)[4].numpy()
    def _S(i):
        return tr.areas_of(i)[1] if hasattr(tr, 'areas_of') else tr.load(i)[5]
    mean, std = channel_stats4(n=400)
    (OUT / 'config.json').write_text(json.dumps(dict(vars(a), mean=mean, std=std), indent=2))
    print('SCL 通道均值 = %.4f / %.4f（缺失即置为此值 -> 归一化后为 0）'
          % (mean[IDX_SCL_NORM], mean[IDX_SCL_CLOUD]), flush=True)
    TEACHER = {}
    if a.teacher:
        z = np.load(a.teacher, allow_pickle=True)
        TEACHER = {k: v.astype(np.float32) for k, v in zip(z['keys'].tolist(), z['areas'])}
        print('蒸馏教师 %d 项', flush=True)

    proc = AutoProcessor.from_pretrained(base, local_files_only=True, max_pixels=65536)
    proc.tokenizer.padding_side = 'right'
    vlm = Qwen3VLForConditionalGeneration.from_pretrained(
        base, local_files_only=True, dtype=torch.bfloat16, device_map={'': 0},
        attn_implementation='sdpa', low_cpu_mem_usage=True)
    vlm = PeftModel.from_pretrained(vlm, a.resume_adapter, is_trainable=True)
    vlm.config.use_cache = False
    print('LoRA 起始 =', a.resume_adapter, flush=True)
    model = UnifiedV4(vlm, calib=a.calib)
    model.dec = model.dec.cuda(); model.season = model.season.cuda()
    pe = model.visual.patch_embed.proj
    if a.resume_heads:
        ck = torch.load(a.resume_heads, map_location='cpu')
        if 'patch_embed_proj' in ck:
            model.visual.patch_embed.proj.load_state_dict(ck['patch_embed_proj']); print('patch_embed 续载', flush=True)
        for nm, mod in [('dec', model.dec), ('season', model.season)]:
            if nm in ck:
                tgt = mod.state_dict()
                good = {k: v for k, v in ck[nm].items() if k in tgt and tgt[k].shape == v.shape}
                mod.load_state_dict(good, strict=False)
                print('%s 续载 %d/%d 项（形状不符的跳过并重新初始化）' % (nm, len(good), len(ck[nm])), flush=True)
    _pid = {id(pe.weight), id(pe.bias)}
    if a.no_lm:      # 只训头：冻结语言模型与 patch_embed，视觉塔不反传
        for p in model.vlm.parameters():
            p.requires_grad = False
        pe.weight.requires_grad = False; pe.bias.requires_grad = False
        print('*** no_lm 模式：冻结 LoRA 与 patch_embed，只训练 dec/season ***', flush=True)
    if a.freeze_season:
        model.season.requires_grad_(False)
        print('*** 季节头已冻结 ***', flush=True)
    lora_params = [p for p in model.vlm.parameters() if p.requires_grad and id(p) not in _pid]
    head_params = list(model.dec.parameters()) + ([] if a.freeze_season else list(model.season.parameters()))
    groups = [{'params': head_params, 'lr': a.lr_head}]
    if lora_params:
        groups.insert(0, {'params': lora_params, 'lr': a.lr_lora})
    if pe.weight.requires_grad:
        groups.append({'params': [pe.weight, pe.bias], 'lr': a.lr_patch})
    opt = torch.optim.AdamW(groups, weight_decay=0.01)

    rows = list(range(len(tr)))
    # ---- 逐作物面积分箱 -> 重要性权重（把训练分布拉向"对数面积均匀"，直接对收缩偏差）----
    WLUT = None
    if a.reweight > 0:
        import math
        nb = 24; lo, hi = math.log(0.1), math.log(150.0)
        cnt = np.zeros((5, nb))
        for i in rows:
            ar = _A(i)
            for ci in range(5):
                b = int((math.log(max(float(ar[ci]), 0.1)) - lo) / (hi - lo) * nb)
                cnt[ci, min(max(b, 0), nb - 1)] += 1
        wlut = 1.0 / np.maximum(cnt, 1.0) ** a.reweight
        WLUT = (wlut, nb, lo, hi)
        # 关键修正：归一化到"样本权重的均值 = 1"，避免整体削弱面积损失
        _w = []
        for i in rows[::max(1, len(rows) // 800)]:
            ar = _A(i)
            ws = []
            for ci in range(5):
                import math as _m
                b = int((_m.log(max(float(ar[ci]), 0.1)) - lo) / (hi - lo) * nb)
                ws.append(wlut[ci, min(max(b, 0), nb - 1)])
            _w.append(float(np.mean(ws)))
        scale = float(np.mean(_w))
        WLUT = (wlut / scale, nb, lo, hi)
        print('重加权 alpha=%.2f  样本权重均值已归一化(scale=%.4f)  范围 %.3f~%.3f'
              % (a.reweight, scale, wlut.min()/scale, wlut.max()/scale), flush=True)

    def sample_w(areas):
        if WLUT is None: return 1.0
        import math
        wlut, nb, lo, hi = WLUT; ws = []
        for ci in range(5):
            b = int((math.log(max(float(areas[ci]), 0.1)) - lo) / (hi - lo) * nb)
            ws.append(wlut[ci, min(max(b, 0), nb - 1)])
        return float(np.mean(ws))
    seas = np.array([_S(i) for i in range(len(tr))])   # 全长索引（rows 可能被缓存过滤过）
    wheat = [i for i in rows if seas[i] == 1]; bean = [i for i in rows if seas[i] == 0]
    print('训练样本 麦季 %d 豆季 %d' % (len(wheat), len(bean)), flush=True)

    # ---- 按作物稀有度加权采样：官方数据里玉米占 74% 影像、水稻仅 7%，
    #      均匀采样会让模型极少见到水稻（这正是水稻 T3 低估 39% 的根源）----
    CROPS5 = ['水稻', '玉米', '棉花', '大豆', '冬小麦']
    SW = None
    if a.crop_balance > 0:
        import collections, math as _m
        freq = collections.Counter()
        for i in rows:
            ar = _A(i)
            for ci in range(5):
                if ar[ci] >= 1.0:
                    freq[ci] += 1
        n = float(len(rows))
        w_c = {ci: (freq[ci] / n) for ci in range(5) if freq[ci] > 0}
        print('  作物出现率: ' + '  '.join('%s %.1f%%' % (CROPS5[ci], w_c[ci] * 100)
                                          for ci in sorted(w_c)), flush=True)
        base_w = max(w_c.values())
        SW = {}
        for i in rows:
            ar = _A(i)
            w = 1.0
            for ci in range(5):
                if ar[ci] >= 1.0:
                    # 出现率越低权重越高；alpha 控制强度
                    w = max(w, (base_w / w_c[ci]) ** a.crop_balance)
            SW[i] = min(w, 20.0)
        vals = list(SW.values())
        print('  采样权重: min %.2f 中位 %.2f max %.2f' % (
            min(vals), sorted(vals)[len(vals)//2], max(vals)), flush=True)
    gh = SIZE_IN // PSZ
    ntok = (gh * gh) // (MERGE ** 2)
    rng = random.Random(7)

    def vision_batch(items, aug, scl_absent):
        """items -> (pv, grid, xr, vc, xs, abs_)"""
        xs, pvs, vlds, abs_ = [], [], [], []
        for (r, key, x18, y7, areas, sea, cld) in items:
            x = cloud_occlude(x18, rng, p=a.cloud_aug) if aug else x18
            ab = True if scl_absent else (rng.random() < a.scl_drop if aug else False)
            xs.append(x); abs_.append(ab); vlds.append(x[14:15])
        pv = torch.cat([patchify(to_model_input4(x, mean, std, ab).numpy())
                        for x, ab in zip(xs, abs_)]).to(torch.bfloat16).cuda()
        xr = torch.stack([to_model_input4(x, mean, std, ab) for x, ab in zip(xs, abs_)]).cuda()
        vc = torch.stack(vlds).cuda().float()
        grid = torch.tensor([[1, gh, gh]] * len(items)).cuda()
        return dict(pv=pv, grid=grid, xr=xr, vc=vc, abs_=abs_)

    def text_inputs(items, hints):
        tf, tp, pref = [], [], []
        for (r, key, x18, y7, areas, sea, cld), h in zip(items, hints):
            hint = '' if h is None else AREA_HINT.format(a=h)
            msgs, pmsgs = [], []
            for msg in r['messages']:
                if msg['role'] == 'user':
                    cc = [{'type': 'image'}, {'type': 'text', 'text': msg['content'].replace('<image>', '') + hint}]
                    msgs.append({'role': 'user', 'content': cc}); pmsgs.append({'role': 'user', 'content': cc})
                elif msg['role'] == 'assistant':
                    msgs.append(msg)
                else:
                    msgs.append(msg); pmsgs.append(msg)
            f = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False).replace('<|image_pad|>', '<|image_pad|>' * ntok)
            p = proc.apply_chat_template(pmsgs, tokenize=False, add_generation_prompt=True).replace('<|image_pad|>', '<|image_pad|>' * ntok)
            tf.append(f); tp.append(p)
            pref.append(len(proc.tokenizer(p, add_special_tokens=False)['input_ids']))
        ef = proc.tokenizer(tf, return_tensors='pt', padding=True, add_special_tokens=False)
        ids = ef['input_ids']; am = ef['attention_mask']
        labels = ids.clone()
        for b, p in enumerate(pref): labels[b, :p] = -100
        labels[am == 0] = -100
        return dict(input_ids=ids.cuda(), attention_mask=am.cuda(), labels=labels.cuda(),
                    mm_token_type_ids=((ids == proc.image_token_id).long() * am).cuda())

    @torch.inference_mode()
    def eval_val(scl_absent, bs=4):
        """返回 (task, per_task, mae, season_acc, cloud_iou, areas_by_key)"""
        model.eval()
        areas_out = {}; mae = []; pos = []; sea_ok = 0; ciou = []; _pr = []; _tr = []
        for s in range(0, len(va), bs):
            items = [va.load(i) for i in range(s, min(s + bs, len(va)))]
            b = vision_batch(items, aug=False, scl_absent=scl_absent)
            if a.no_lm:
                with torch.no_grad():
                    feat = model.forward_vision(b['pv'], b['grid'])
                f2 = feat.reshape(len(items), gh, gh, -1).permute(0, 3, 1, 2).float().detach()
            else:
                feat = model.forward_vision(b['pv'], b['grid'])
                f2 = feat.reshape(len(items), gh, gh, -1).permute(0, 3, 1, 2).float()
            logits, cld_l = model.dec(f2, b['xr'], want_cloud=True)
            _vc = b['vc']
            if _vc.shape[-2:] != logits.shape[-2:]:
                _vc = F.interpolate(_vc, size=logits.shape[-2:], mode='nearest')
            c = code_areas(logits, _vc); sl = model.season(f2)
            ar = torch.stack([official_areas(c[i2], sl[i2]) for i2 in range(len(items))]).cpu()
            for i2, (r, key, x18, y7, tar, sea, cld) in enumerate(items):
                areas_out[key] = ar[i2].numpy()
                _pr.append(ar[i2].numpy()); _tr.append(tar.numpy())
                mae.append((ar[i2] - tar).abs().mean().item())
                if (tar >= 1).any(): pos.append((ar[i2] - tar).abs()[tar >= 1].mean().item())
                sea_ok += int(sl[i2].argmax().item() == sea)
                p = (cld_l[i2, 0].sigmoid() > 0.5); g = (cld[i2].cuda() > 0.5)
                if p.shape != g.shape:   # 高分辨率下预测是 500²，真值仍是 256²
                    p = F.interpolate(p[None].float(), size=g.shape[-2:], mode='nearest')[0] > 0.5
                un = (p | g).sum().item()
                ciou.append(((p & g).sum().item() / un) if un > 0 else 1.0)
        rv = []
        for r in va.rows:
            q = next(x['content'] for x in r['messages'] if x['role'] == 'user')
            gt = json.loads(r['messages'][-1]['content'])
            rv.append((r['qa_id'], va.map[r['qa_id']], qtype(gt, q), q, gt))
        means, task = score_from_areas(rv, areas_out)
        PR = np.stack(_pr); TR = np.stack(_tr)
        ratios = []
        for ci in range(5):
            m = TR[:, ci] >= 1
            ratios.append(float(PR[m, ci].mean() / max(TR[m, ci].mean(), 1e-6)) if m.sum() else float('nan'))
        model.train()
        return task, means, float(np.mean(mae)), sea_ok / len(va), float(np.mean(ciou)), areas_out, ratios

    def save(tag, step, m):
        model.vlm.save_pretrained(OUT / tag); proc.save_pretrained(OUT / tag)
        torch.save({'dec': model.dec.state_dict(), 'season': model.season.state_dict(),
                    'patch_embed_proj': model.visual.patch_embed.proj.state_dict(),
                    'n_in': NC4, 'step': step, 'metrics': m, 'calib': a.calib}, OUT / (tag + '_heads.pt'))

    step = 0; best_mae = float('inf'); best_task = -1.0; t0 = time.time()
    acc = dict(json=0., seg=0., area=0., sea=0., cloud=0., dist=0., sub=0., n=0)
    for _ in range(a.steps):
        if SW is None:
            group = [rng.choice(wheat) for _ in range(2)] + [rng.choice(bean) for _ in range(a.group - 2)]
        else:
            group = [rng.choices(wheat, weights=[SW[i] for i in wheat])[0] for _ in range(2)] +                     [rng.choices(bean, weights=[SW[i] for i in bean])[0] for _ in range(a.group - 2)]
        opt.zero_grad(set_to_none=True)
        nmb = (len(group) + a.micro - 1) // a.micro
        for mi in range(nmb):
            items = [tr.load(i) for i in group[mi * a.micro:(mi + 1) * a.micro]]
            b = vision_batch(items, aug=True, scl_absent=False)
            feat = model.forward_vision(b['pv'], b['grid'])
            f2 = feat.reshape(len(items), gh, gh, -1).permute(0, 3, 1, 2).float()
            logits, cld_l = model.dec(f2, b['xr'], want_cloud=True)
            y7 = torch.stack([it[3] for it in items]).cuda()
            tar = torch.stack([it[4] for it in items]).cuda()
            sea = torch.tensor([it[5] for it in items]).cuda()
            cldg = torch.stack([it[6] for it in items]).cuda()
            seg_l = -(y7 * logits.log_softmax(1)).sum(1).mean()
            _vc = b['vc']
            if _vc.shape[-2:] != logits.shape[-2:]:
                _vc = F.interpolate(_vc, size=logits.shape[-2:], mode='nearest')
            c = code_areas(logits, _vc); sl = model.season(f2)
            ar = torch.stack([official_areas(c[i2], sl[i2]) for i2 in range(len(items))])
            # V6: 相对误差导向的面积损失（原 log1p 在小面积区退化为绝对误差）
            area_l = F.smooth_l1_loss(torch.log(ar + 0.05), torch.log(tar + 0.05),
                                      beta=0.5, reduction='none').mean(1)
            area_l = (area_l * torch.tensor([sample_w(it[4]) for it in items], device=area_l.device)).mean()
            sea_l = F.cross_entropy(sl, sea)
            if cld_l.shape[-2:] != cldg.shape[-2:]:
                cld_l = F.interpolate(cld_l, size=cldg.shape[-2:], mode='bilinear', align_corners=False)
            cloud_l = F.binary_cross_entropy_with_logits(cld_l[:, 0], (cldg > 0.5).float())
            loss = seg_l + a.w_area * area_l + a.w_season * sea_l + a.w_cloud * cloud_l
            # ---- 子窗口面积监督：随机取标签子窗口，按官方口径（比例×100 km²）算面积 ----
            if a.w_sub > 0:
                H, W = logits.shape[-2], logits.shape[-1]
                S = max(32, int(H * a.sub_scale))
                for _w in range(max(1, a.sub_n)):
                    y0 = rng.randrange(0, H - S + 1); x0 = rng.randrange(0, W - S + 1)
                    cs = code_areas(logits[:, :, y0:y0 + S, x0:x0 + S], _vc[:, :, y0:y0 + S, x0:x0 + S])
                    ts = (y7[:, :, y0:y0 + S, x0:x0 + S] * _vc[:, :, y0:y0 + S, x0:x0 + S]).mean((2, 3)) * 100.0
                    sub_l = F.smooth_l1_loss(torch.log(cs + 0.05), torch.log(ts + 0.05), beta=0.5)
                    loss = loss + a.w_sub * sub_l / max(1, a.sub_n)
                    acc['sub'] += float(sub_l.item())
            if TEACHER:
                tv = torch.tensor(np.stack([TEACHER.get(it[1], np.zeros(5, np.float32)) for it in items])).cuda()
                dl = F.smooth_l1_loss(torch.log1p(ar), torch.log1p(tv))
                loss = loss + a.w_distill * dl; acc['dist'] += dl.item()
            if not a.no_lm:
                hints = [None if a.no_inject else [float(v) for v in ar[i2].detach().cpu()] for i2 in range(len(items))]
                inp2 = text_inputs(items, hints)
                inp2['pixel_values'] = b['pv']; inp2['image_grid_thw'] = b['grid']
                out = model.vlm(**inp2, use_cache=False)
                loss = (loss + out.loss) / nmb
                acc['json'] += out.loss.item()
                del out, inp2
            else:
                loss = loss / nmb
            loss.backward()
            acc['seg'] += seg_l.item(); acc['area'] += area_l.item()
            acc['sea'] += sea_l.item(); acc['cloud'] += cloud_l.item(); acc['n'] += 1
            del feat, f2, logits
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step(); step += 1
        if step == 1:
            print('梯度 patch=%s dec=%s cloud=%s' % (
                None if pe.weight.grad is None else '%.2e' % pe.weight.grad.norm().item(),
                None if model.dec.out.weight.grad is None else '%.2e' % model.dec.out.weight.grad.norm().item(),
                None if model.dec.cloud.weight.grad is None else '%.2e' % model.dec.cloud.weight.grad.norm().item()), flush=True)
        if step % 10 == 0 or step == 1:
            n = max(acc['n'], 1)
            print('STEP %4d json %.4f seg %.4f area %.4f sub %.4f season %.4f cloud %.4f dist %.4f | %.0fs gpu %.1fGiB'
                  % (step, acc['json'] / n, acc['seg'] / n, acc['area'] / n, acc['sub'] / n, acc['sea'] / n,
                     acc['cloud'] / n, acc['dist'] / n, time.time() - t0,
                     torch.cuda.max_memory_allocated() / 2**30), flush=True)
            acc = dict(json=0., seg=0., area=0., sea=0., cloud=0., dist=0., sub=0., n=0)
        if step % a.eval_every == 0:
            t_a, pa, mae_a, sea_a, ciou, areas_a, ratios = eval_val(scl_absent=True)   # 测试条件
            with (OUT / ('val_areas_step%d.json' % step)).open('w') as _f:
                json.dump({k: [round(float(x), 4) for x in v] for k, v in areas_a.items()}, _f)
            t_p, pp, mae_p, sea_p, _, _, _ = eval_val(scl_absent=False)         # 可比队友 V3
            m = dict(mae=mae_a, pos_mae=-1.0, season_acc=sea_a, cloud_iou=ciou, task=t_a,
                     per_task=pa, task_scl_present=t_p, mae_scl_present=mae_p,
                     per_task_scl_present=pp)
            print('VAL step=%d  [SCL缺失=测试] MAE=%.3f season=%.3f cloudIoU=%.3f task=%.4f %s'
                  % (step, mae_a, sea_a, ciou, t_a, {k: round(v, 4) for k, v in pa.items()}), flush=True)
            print('            逐作物收缩比(预测均值/真值均值) 水%.2f 玉%.2f 棉%.2f 豆%.2f 麦%.2f'
                  % tuple(ratios), flush=True)
            print('           [SCL存在=可比V3] MAE=%.3f task=%.4f %s'
                  % (mae_p, t_p, {k: round(v, 4) for k, v in pp.items()}), flush=True)
            with (OUT / 'metrics.jsonl').open('a') as f: f.write(json.dumps(dict(step=step, **m)) + '\n')
            save('last', step, m)
            # 定期留档：任何步都可能成为 task 峰值，不能只靠 best/last 两个槽位
            if step % a.keep_every == 0:
                save('ckpt-%d' % step, step, m)
            # 按 task 选模（这才是评分指标）；按 MAE 选模本轮三次选错检查点
            if t_a > best_task:
                best_task = t_a; save('best_task', step, m)
                (OUT / 'best_task.json').write_text(json.dumps(dict(step=step, **m), indent=2))
                print('           >>> 新 best_task: step=%d task=%.4f (MAE=%.3f)'
                      % (step, t_a, mae_a), flush=True)
            # 同时保留 MAE 最优，供对照
            if mae_a < best_mae:
                best_mae = mae_a; save('best_mae', step, m)
                (OUT / 'best_mae.json').write_text(json.dumps(dict(step=step, **m), indent=2))
    print('TRAIN_DONE steps=%d  best_task(SCL缺失)=%.4f  best_mae(SCL缺失)=%.3f'
          % (step, best_task, best_mae), flush=True)


if __name__ == '__main__':
    main()
