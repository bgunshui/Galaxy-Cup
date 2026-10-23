# -*- coding: utf-8 -*-
"""V6 推理：批处理（P0）——与 v4_infer 逐题版等价，但一次前向处理一批
只改工程不改模型：批内一次视觉前向算面积 → 面积注入 prompt → 批量 generate
答案经 schema 校验，不合法则回退到规则层（不用类别先验）
"""
import json, re, sys, time, functools, argparse
import numpy as np, torch, tifffile, os

# 设备无关：星上平台未必是 CUDA，逐级回退（可用 --device 覆盖）
def _pick_device(override=None):
    if override:
        return override
    try:
        if torch.cuda.is_available():
            return 'cuda'
    except Exception:
        pass
    for name in ('npu', 'xpu', 'mps'):          # 昇腾 / 昆仑 / Apple
        try:
            if getattr(torch, name, None) is not None and getattr(torch, name).is_available():
                return name
        except Exception:
            pass
    return 'cpu'
from pathlib import Path
from torch.nn import functional as F
sys.path.insert(0, str(Path(__file__).resolve().parent))
import v2_lib as L
import v4_lib as L4
from v4_lib import UnifiedV4, to_model_input4, AREA_HINT, NC4, IDX_SCL_NORM, IDX_SCL_CLOUD, CodeDS4
from v2_lib import (patchify, official_areas, code_areas, answer_from_areas, qtype, score_from_areas,
                    CROPS, PSZ, MERGE)
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

R = Path('.')
dec_name = lambda s: re.sub(r'#U([0-9a-fA-F]{4})', lambda m: chr(int(m[1], 16)), s)


def build_index(root):
    idx = {}
    for p in Path(root).rglob('*.tif'):
        k = (dec_name(p.parent.name), p.name)
        if k[0] in ('光学影像', 'SAR影像'):
            idx.setdefault(k, []).append(p)
    return idx


def load18(opt_path, sar_path, scl_path, size, mean):
    a = tifffile.imread(opt_path)
    if a.shape[0] == 12: a = np.moveaxis(a, 0, -1)
    a = a.astype(np.float32)
    valid = np.all(np.isfinite(a) & (a > 0), -1)
    s = np.zeros(a.shape[:2] + (2,), np.float32); sv = np.zeros(a.shape[:2], np.float32)
    if sar_path is not None:
        raw = tifffile.imread(sar_path)
        if raw.shape[0] == 2: raw = np.moveaxis(raw, 0, -1)
        raw = raw.astype(np.float32)
        good = np.all(np.isfinite(raw) & (raw != -9999), -1)
        s = np.where(good[..., None], np.clip((raw + 35) / 40, 0, 1), 0)
        sv = good.astype(np.float32)
    if scl_path is not None:
        rs = np.asarray(tifffile.imread(scl_path)).astype(np.float32)
        sn = np.clip(rs / 9.0, 0, 1); sc = np.isin(rs.astype(np.uint8), [3, 8, 9]).astype(np.float32)
    else:
        sn = np.full(a.shape[:2], mean[IDX_SCL_NORM], np.float32)
        sc = np.full(a.shape[:2], mean[IDX_SCL_CLOUD], np.float32)
    x = np.concatenate([np.clip(np.nan_to_num(a) / 10000, 0, 2), s, valid[..., None], sv[..., None],
                        sn[..., None], sc[..., None]], -1)
    x = torch.from_numpy(x).permute(2, 0, 1)[None]
    x = F.interpolate(x, (size, size), mode='area')[0].numpy().astype(np.float16).astype(np.float32)
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--head', default='best_task_heads.pt')
    ap.add_argument('--adapter', default='best_task')
    ap.add_argument('--size', type=int, default=256, choices=[256, 512], help='视觉输入分辨率（512=更细的token网格）')
    ap.add_argument('--merged', default='', help='合并后的完整权重目录（推荐，免 peft）')
    ap.add_argument('--peft_base', required=True, help='local Qwen3-VL base directory')
    ap.add_argument('--input', required=True, help='input question JSONL')
    ap.add_argument('--data-root', default=os.environ.get('B_MATERIAL_DATA_ROOT', '.'),
                    help='test data root; may contain test/ or qa_test.jsonl directly')
    ap.add_argument('--device', default='', help='强制设备，如 cuda / cpu / npu:0')
    ap.add_argument('--split', default='test', choices=['test', 'val'])
    ap.add_argument('--out', required=True)
    ap.add_argument('--out_rules', default='')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--bs', type=int, default=8)
    ap.add_argument('--no_inject', action='store_true')
    ap.add_argument('--out_areas', default='', help='同时保存每题的 5 维面积向量，供事后离线做决策层 A/B')
    a = ap.parse_args()
    global R
    data_root = Path(a.data_root).resolve()
    test_root = data_root / 'test' if (data_root / 'test').is_dir() else data_root
    R = data_root
    run = Path(a.run)
    cfg = json.loads((run / 'config.json').read_text())
    mean, std = cfg['mean'], cfg['std']
    ck = torch.load(run / a.head, map_location='cpu')
    print('载入 %s step=%s' % (a.head, ck.get('step')), flush=True)
    print('批大小 = %d ; 统一测试条件: SCL 缺失' % a.bs, flush=True)

    torch.set_num_threads(4)
    L4.DEC_RES = 500
    proc = AutoProcessor.from_pretrained(a.peft_base, local_files_only=True, max_pixels=65536)
    proc.tokenizer.padding_side = 'left'
    print('Using the checkpoint internal season head only', flush=True)

    DEV = _pick_device(a.device)
    print('推理设备 = %s' % DEV, flush=True)
    src_w = a.merged or str(run / 'merged')
    if a.peft_base:
        vlm = Qwen3VLForConditionalGeneration.from_pretrained(
            a.peft_base, local_files_only=True, dtype=torch.bfloat16,
            attn_implementation='sdpa', low_cpu_mem_usage=True)
        from peft import PeftModel
        vlm = PeftModel.from_pretrained(vlm, str(run / a.adapter), is_trainable=False)
        vlm = vlm.merge_and_unload().to(DEV)
        print('PEFT 模式: 底模=%s 适配器=%s' % (a.peft_base, run / a.adapter), flush=True)
    else:
        if not Path(src_w).exists():
            raise SystemExit('找不到合并权重 %s，请先跑 merge_lora.py（或显式传 --merged）' % src_w)
        vlm = Qwen3VLForConditionalGeneration.from_pretrained(
            src_w, local_files_only=True, dtype=torch.bfloat16,
            attn_implementation='sdpa', low_cpu_mem_usage=True).to(DEV)
    model = UnifiedV4(vlm, calib=ck['calib'])
    if not a.merged:      # 合并权重里已含 patch_embed，旧流程才需要单独载入
        model.visual.patch_embed.proj.load_state_dict(ck['patch_embed_proj'])
    model.dec = model.dec.to(DEV).eval(); model.season = model.season.to(DEV).eval()
    model.dec.load_state_dict(ck['dec']); model.season.load_state_dict(ck['season'])
    model.vlm.config.use_cache = True
    SIZE = a.size
    gh = SIZE // PSZ
    ntok = (gh * gh) // (MERGE ** 2)
    print('模型就绪', flush=True)

    # ---------- 取题目 ----------
    if a.split == 'val':
        va = CodeDS4('val')
        idx = build_index(R / 'val')
        items = []
        for i2, r in enumerate(va.rows):
            q = next(m['content'] for m in r['messages'] if m['role'] == 'user')
            gt = json.loads(r['messages'][-1]['content'])
            on = sn_ = None
            for rel in r['images']:
                nm = dec_name(Path(rel).parent.name)
                if nm == '光学影像': on = Path(rel).name
                elif nm == 'SAR影像': sn_ = Path(rel).name
            items.append(dict(qid=r['qa_id'], opt=on, sar=sn_, q=q, gt=gt))
        print('验证集 %d 题：原始影像按 --size=%d 渲染（索引 %d 项）' % (len(items), SIZE, len(idx)), flush=True)
        if a.limit: items = items[:a.limit]
    else:
        T = test_root
        rows = [json.loads(l) for l in Path(a.input).open(encoding='utf8') if l.strip()]
        if len({row['qa_id'] for row in rows}) != len(rows):
            raise ValueError('Duplicate qa_id in input')
        if a.limit: rows = rows[:a.limit]
        idx = build_index(T)
        print('题目 %d 道, 索引 %d 项' % (len(rows), len(idx)), flush=True)

        items = []
        for r in rows:
            on = sn_ = None
            for rel in r['images']:
                nm = dec_name(Path(rel).parent.name)
                if nm == '光学影像': on = Path(rel).name
                elif nm == 'SAR影像': sn_ = Path(rel).name
            q = next(m['content'] for m in r['messages'] if m['role'] == 'user')
            # 只记路径，影像按批惰性加载（测试集要读近 4000 个 tif，串行加载会成为瓶颈）
            items.append(dict(qid=r['qa_id'], opt=on, sar=sn_, q=q, gt=None))

    @functools.lru_cache(maxsize=32)
    def get18(opt_name, sar_name):
        o = idx.get(('光学影像', opt_name)); s = idx.get(('SAR影像', sar_name)) if sar_name else None
        assert o, opt_name
        return load18(o[0], s[0] if s else None, None, SIZE, mean)

    def ask(qtext, hint):
        h = '' if hint is None else AREA_HINT.format(a=hint)
        msgs = [{'role': 'user', 'content': [{'type': 'image'},
                 {'type': 'text', 'text': qtext.replace('<image>', '') + h}]}]
        return proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True
                                        ).replace('<|image_pad|>', '<|image_pad|>' * ntok)

    res_llm = {}; res_rule = {}; nbad = 0; res_areas = {}
    t0 = time.time(); t_vis = 0.0; t_gen = 0.0
    from concurrent.futures import ThreadPoolExecutor
    pool = ThreadPoolExecutor(max_workers=6) if a.split in ('test', 'val') else None
    fut = None
    if pool is not None:
        fut = pool.submit(lambda blk: list(pool.map(lambda it: get18(it['opt'], it['sar']), blk)), items[0:a.bs])
    with torch.inference_mode():
        for s in range(0, len(items), a.bs):
            blk = items[s:s + a.bs]; n = len(blk)
            if pool is not None:
                xs_loaded = fut.result()
                nxt = items[s + a.bs:s + 2 * a.bs]
                if nxt:
                    fut = pool.submit(lambda blk2: list(pool.map(lambda it: get18(it['opt'], it['sar']), blk2)), nxt)
                for it, x in zip(blk, xs_loaded):
                    it['x18'] = x
            tv = time.time()
            xn = [to_model_input4(torch.from_numpy(it['x18']), mean, std, scl_absent=True) for it in blk]
            pv = torch.cat([patchify(x.numpy()) for x in xn]).to(DEV).to(torch.bfloat16)
            xr = torch.stack(xn).to(DEV)
            vc = torch.stack([torch.from_numpy(it['x18'][14:15]) for it in blk]).to(DEV).float()
            grid = torch.tensor([[1, gh, gh]] * n).to(DEV)
            feat = model.forward_vision(pv, grid)
            f2 = feat.reshape(n, gh, gh, -1).permute(0, 3, 1, 2).float()
            lg = model.dec(f2, xr)
            if vc.shape[-2:] != lg.shape[-2:]:
                vc = F.interpolate(vc, size=lg.shape[-2:], mode='nearest')
            c = code_areas(lg, vc)
            sl = model.season(f2)
            ar = torch.stack([official_areas(c[b], sl[b]) for b in range(n)]).cpu().numpy()
            t_vis += time.time() - tv
            rules = []
            for b, it in enumerate(blk):
                t = qtype(it['q'])
                rr = answer_from_areas(t, it['q'], ar[b])
                res_rule[it['qid']] = rr; rules.append(rr)
                res_areas[it['qid']] = [float(v) for v in ar[b]]
            tg = time.time()
            hints = [None if a.no_inject else [float(v) for v in ar[b]] for b in range(n)]
            texts = [ask(it['q'], hints[b]) for b, it in enumerate(blk)]
            tk = proc.tokenizer(texts, return_tensors='pt', padding=True, add_special_tokens=False)
            ids = tk['input_ids'].to(DEV); am = tk['attention_mask'].to(DEV)
            mm = ((ids == proc.image_token_id).long() * am).to(DEV)
            with torch.autocast(DEV, dtype=torch.bfloat16):
                gen = model.vlm.generate(input_ids=ids, attention_mask=am, pixel_values=pv,
                                         image_grid_thw=grid, mm_token_type_ids=mm,
                                         max_new_tokens=48, do_sample=False)
            outs = proc.tokenizer.batch_decode(gen[:, ids.shape[1]:], skip_special_tokens=True)
            t_gen += time.time() - tg
            for b, it in enumerate(blk):
                t = qtype(it['q']); m = re.search(r'\{.*\}', outs[b], re.S); ans = None
                try:
                    obj = json.loads(m.group(0)) if m else None
                    if isinstance(obj, dict):
                        if t == 'T1' and obj.get('crop') in CROPS: ans = {'crop': obj['crop']}
                        elif t == 'T2' and isinstance(obj.get('crops'), list) and all(c in CROPS for c in obj['crops']):
                            ans = {'crops': list(dict.fromkeys(obj['crops']))}
                        elif t == 'T3' and obj.get('crop') in CROPS and isinstance(obj.get('area_km2'), (int, float)):
                            ans = {'crop': obj['crop'], 'area_km2': float(obj['area_km2'])}
                        elif t in ('T4', 'T5'):
                            opts = [x for x, _ in re.findall(r'(?m)^\s*([A-Z])[.、．]\s*(.+)$', it['q'])]
                            if obj.get('choice') in opts: ans = {'choice': obj['choice']}
                except Exception:
                    ans = None
                if ans is None:
                    ans = rules[b]; nbad += 1
                res_llm[it['qid']] = ans
            if (s // a.bs) % 20 == 0:
                print('PROGRESS %d/%d %.0fs (视觉 %.0fs 生成 %.0fs) fallback=%d'
                      % (min(s + a.bs, len(items)), len(items), time.time() - t0, t_vis, t_gen, nbad), flush=True)

    with open(a.out, 'w', encoding='utf8') as f:
        for it in items:
            f.write(json.dumps({'qa_id': it['qid'], 'answer': json.dumps(res_llm[it['qid']], ensure_ascii=False, separators=(',', ':'))}, ensure_ascii=False) + '\n')
    if a.out_rules:
        with open(a.out_rules, 'w', encoding='utf8') as f:
            for it in items:
                f.write(json.dumps({'qa_id': it['qid'], 'answer': json.dumps(res_rule[it['qid']], ensure_ascii=False, separators=(',', ':'))}, ensure_ascii=False) + '\n')
    if a.out_areas:
        with open(a.out_areas, 'w', encoding='utf8') as f:
            f.write(json.dumps(res_areas, ensure_ascii=False))
        print('面积向量已保存 ->', a.out_areas, flush=True)
    el = time.time() - t0
    print('DONE %s  非法回退=%d  总 %.0fs (视觉 %.0fs 生成 %.0fs)  %.3f s/题'
          % (a.out, nbad, el, t_vis, t_gen, el / max(len(items), 1)), flush=True)

    if a.split == 'val':
        def score(res, tag):
            g = {}
            for it in items:
                t = qtype(it['gt'], it['q']); p = res[it['qid']]; s = 0.0
                if t == 'T1': s = float(p['crop'] == it['gt']['crop'])
                elif t == 'T2':
                    P, G = set(p['crops']), set(it['gt']['crops'])
                    s = 1.0 if not P and not G else (2 * len(P & G) / (len(P) + len(G)) if P or G else 0.0)
                elif t == 'T3': s = L.t3score(p['area_km2'], float(it['gt']['area_km2']))
                else: s = float(p['choice'] == it['gt']['choice'])
                g.setdefault(t, []).append(s)
            m = {k: float(np.mean(v)) for k, v in g.items()}
            g2 = lambda k: m.get(k, float('nan'))
            sub = {k: v for k, v in m.items()}
            tot = sum(L.W[k] * sub[k] for k in sub) / sum(L.W[k] for k in sub) if sub else float('nan')
            print('  %-8s T1=%.4f T2=%.4f T3=%.4f T4=%.4f T5=%.4f task(已出现题型归一)=%.4f  n=%s'
                  % (tag, g2('T1'), g2('T2'), g2('T3'), g2('T4'), g2('T5'), tot,
                     {k: len(v) for k, v in g.items()}), flush=True)
        print('\n=== val 两条路径（批量 %d）===' % a.bs, flush=True)
        score(res_rule, '规则层'); score(res_llm, '语言头')


if __name__ == '__main__':
    main()
