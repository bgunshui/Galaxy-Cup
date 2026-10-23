"""Materialize the B-list candidate from one v973 checkpoint inference.

The only post-processing is the fixed, training-only calibration recorded in
calibration.json. T2 remains the official 1 km2 rule. No test labels or other
model outputs are read.
"""
import argparse
import json
import math
import re
from pathlib import Path

CROPS = ['水稻', '玉米', '棉花', '大豆', '冬小麦']

def qtext(row):
    return next(m['content'] for m in row['messages'] if m['role'] == 'user')

def kind(q):
    if '估算' in q and '面积' in q: return 'T3'
    if '比较' in q: return 'T5'
    if '最大' in q: return 'T1'
    if '包含' in q or '哪些' in q: return 'T2'
    return 'T4'

def crop(q):
    return next((c for c in CROPS if c in q.split('A.')[0]), None)

def option_letters(q):
    return [x for x, _ in re.findall(r'(?m)^\s*([A-Z])[.、．]\s*(.+)$', q)]

def t4(q, a):
    c = crop(q); x = float(a[CROPS.index(c)]); fallback = ''; hits = []
    for letter, text in re.findall(r'(?m)^\s*([A-Z])[.、．]\s*(.+)$', q):
        if '以上都不是' in text:
            fallback = letter; continue
        nums = [float(v) for v in re.findall(r'\d+(?:\.\d+)?', text)]
        if '最接近' in q: ok = bool(nums) and abs(x - nums[0]) <= 1
        elif len(nums) >= 2: ok = nums[0] <= x < nums[1]
        elif len(nums) == 1: ok = x < nums[0] if '小于' in text else x >= nums[0]
        else: ok = False
        if ok: hits.append(letter)
    return hits[0] if len(hits) == 1 else (fallback or 'D')

def t5(q, a):
    cs = list(dict.fromkeys(re.findall('|'.join(CROPS), q.split('当两种')[0])))[:2]
    left, right = (float(a[CROPS.index(c)]) for c in cs)
    return 'C' if abs(left - right) <= 5 else ('A' if left > right else 'B')

def load_jsonl(path):
    return [json.loads(x) for x in Path(path).read_text(encoding='utf8').splitlines() if x.strip()]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True)
    ap.add_argument('--raw', required=True, help='raw output from infer_v973_core.py')
    ap.add_argument('--areas', required=True, help='area vectors from infer_v973_core.py')
    ap.add_argument('--calibration', default=str(Path(__file__).resolve().parent.parent / 'calibration.json'))
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    questions = load_jsonl(args.input)
    raw_rows = {r['qa_id']: json.loads(r['answer']) for r in load_jsonl(args.raw)}
    areas = json.loads(Path(args.areas).read_text(encoding='utf8'))
    cal = json.loads(Path(args.calibration).read_text(encoding='utf8'))
    t2_scales = [cal['t2_crop_scales'][c] for c in CROPS]
    t3_scales = cal['t3_crop_scales']
    if set(raw_rows) != {r['qa_id'] for r in questions} or set(areas) != set(raw_rows):
        raise ValueError('prediction coverage mismatch')
    out = []
    for qrow in questions:
        qid = qrow['qa_id']; q = qtext(qrow); t = kind(q); raw = raw_rows[qid]; av = [float(x) for x in areas[qid]]
        if t == 'T1':
            ans = {'crop': CROPS[int(max(range(5), key=lambda i: av[i]))]}
        elif t == 'T2':
            calibrated = [x * s for x, s in zip(av, t2_scales)]
            ans = {'crops': [c for c, x in zip(CROPS, calibrated) if x >= 1.0]}
        elif t == 'T3':
            c = crop(q); value = raw.get('area_km2')
            if raw.get('crop') != c or isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 100:
                value = av[CROPS.index(c)]
            value = max(0.0, min(100.0, float(value) * t3_scales[c]))
            ans = {'crop': c, 'area_km2': round(value, 2)}
        elif t == 'T4':
            ans = {'choice': raw['choice']} if raw.get('choice') in option_letters(q) else {'choice': t4(q, av)}
        else:
            ans = {'choice': t5(q, av)}
        out.append({'qa_id': qid, 'answer': json.dumps(ans, ensure_ascii=False, separators=(',', ':'))})
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text('\n'.join(json.dumps(x, ensure_ascii=False) for x in out) + '\n', encoding='utf8')
    print(json.dumps({'rows': len(out), 'out': str(Path(args.out).resolve()), 't2_threshold_km2': 1.0,
                      'single_checkpoint': True, 'test_labels_used': False}, ensure_ascii=False))

if __name__ == '__main__': main()
