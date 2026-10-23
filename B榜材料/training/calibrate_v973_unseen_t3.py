import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

CROPS = ['水稻', '玉米', '棉花', '大豆', '冬小麦']


def question(row):
    return next(m['content'] for m in row['messages'] if m['role'] == 'user')


def kind(q):
    return 'T3' if '估算' in q and '面积' in q else 'other'


def score(x, y):
    if y == 0:
        return 1.0 if x == 0 else 0.0
    e = abs(x - y) / y
    return 1.0 if e <= .05 else .8 if e <= .10 else .6 if e <= .20 else .4 if e <= .30 else 0.0


def parse_t3(row, q):
    ans = json.loads(row['answer'])
    crops = [c for c in CROPS if c in q.split('A.')[0]]
    if len(crops) != 1 or ans.get('crop') != crops[0]:
        raise ValueError('T3 crop/schema mismatch: ' + row['qa_id'])
    value = ans.get('area_km2')
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError('T3 prediction is not finite: ' + row['qa_id'])
    return crops[0], float(value)


def geometric(lang, head, alpha, scale=1.0):
    # The floor is only numerical protection for tiny decoder outputs; it does
    # not create a new decision source.
    return max(0.0, min(100.0, scale * max(lang, .05) ** alpha * max(head, .05) ** (1.0 - alpha)))


def arithmetic(lang, head, alpha, scale=1.0):
    return max(0.0, min(100.0, scale * (alpha * lang + (1.0 - alpha) * head)))


def eval_rows(rows, spec):
    vals = []
    for r in rows:
        if spec['kind'] == 'raw':
            p = r['lang']
        elif spec['kind'] == 'head':
            p = r['head']
        elif spec['kind'] == 'geo':
            a = spec['alpha'][r['crop_i']]
            p = geometric(r['lang'], r['head'], a, spec.get('scale', 1.0))
        elif spec['kind'] == 'arith':
            a = spec['alpha'][r['crop_i']]
            p = arithmetic(r['lang'], r['head'], a, spec.get('scale', 1.0))
        else:
            raise ValueError(spec)
        vals.append(score(p, r['truth']))
    return float(np.mean(vals)) if vals else float('nan')


def candidates(train):
    # Keep the family small and interpretable. A per-crop alpha is selected
    # only inside each training fold; no test answer participates in fitting.
    alphas = np.arange(0.0, 1.0001, .05)
    scales = np.arange(.90, 1.1001, .025)
    out = [{'kind': 'raw'}, {'kind': 'head'}]
    for form in ('geo', 'arith'):
        for scale in scales:
            # A single alpha is a lower-variance baseline.
            for a in alphas:
                out.append({'kind': form, 'alpha': [float(a)] * 5, 'scale': float(scale)})
            # Per-crop alpha permits crop-specific bias while remaining one
            # deterministic transformation of the same checkpoint outputs.
            best = []
            for ci in range(5):
                rr = [r for r in train if r['crop_i'] == ci]
                if not rr:
                    best.append(.5)
                    continue
                scores = []
                for a in alphas:
                    spec = {'kind': form, 'alpha': [float(a)] * 5, 'scale': float(scale)}
                    scores.append(eval_rows(rr, spec))
                best.append(float(alphas[int(np.argmax(scores))]))
            out.append({'kind': form, 'alpha': best, 'scale': float(scale)})
    return out


def fit_fold(train):
    # Fitting all candidates is cheap: calibration has only a few hundred T3s.
    specs = candidates(train)
    scores = [eval_rows(train, s) for s in specs]
    # Prefer the simpler raw path for ties, then head, then global before
    # per-crop. This avoids unstable last-digit choices on the step score.
    order = {'raw': 0, 'head': 1, 'geo': 2, 'arith': 3}
    best_i = max(range(len(specs)), key=lambda i: (scores[i], -order[specs[i]['kind']], -i))
    return specs[best_i], float(scores[best_i])


def fit_full(rows):
    spec, training_score = fit_fold(rows)
    return spec, training_score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--qa-train', type=Path, default=Path('qa_train.jsonl'))
    ap.add_argument('--calib-language', type=Path, default=Path('v973_unseen_calib_t3.jsonl'))
    ap.add_argument('--calib-areas', type=Path, default=Path('v973_unseen_calib_t3.areas.json'))
    ap.add_argument('--test-questions', type=Path, default=Path('qa_test.jsonl'))
    ap.add_argument('--test-language', type=Path, default=Path('v973_language_test.jsonl'))
    ap.add_argument('--test-areas', type=Path, default=Path('candidate_v973_res500_t4.areas.json'))
    ap.add_argument('--parent', type=Path, default=Path('candidate_v973_submit_t2_10.jsonl'))
    ap.add_argument('--out', type=Path, default=Path('candidate_v973_t3_blend_unseen_t2_10.jsonl'))
    ap.add_argument('--report', type=Path, default=Path('results/v973_t3_blend_unseen.json'))
    args = ap.parse_args()

    qa = {r['qa_id']: r for r in (json.loads(x) for x in args.qa_train.read_text(encoding='utf8').splitlines())}
    lang = {r['qa_id']: r for r in (json.loads(x) for x in args.calib_language.read_text(encoding='utf8').splitlines())}
    areas = json.loads(args.calib_areas.read_text(encoding='utf8'))
    rows = []
    for qid, pred_row in lang.items():
        source = qa.get(qid)
        if source is None or kind(question(source)) != 'T3':
            continue
        crop, lp = parse_t3(pred_row, question(source))
        gold = json.loads(source['messages'][-1]['content'])
        crop_gold = gold.get('crop')
        if crop != crop_gold:
            raise ValueError('T3 crop differs from gold crop: ' + qid)
        ci = CROPS.index(crop)
        if qid not in areas:
            raise ValueError('Missing calibration area vector: ' + qid)
        hp = float(areas[qid][ci])
        rows.append({'qa_id': qid, 'group': tuple(source['images']), 'crop_i': ci,
                     'crop': crop, 'truth': float(gold['area_km2']), 'lang': lp, 'head': hp})
    if len(rows) < 100 or len({r['group'] for r in rows}) < 20:
        raise ValueError('Calibration set is unexpectedly small')

    groups = sorted({r['group'] for r in rows})
    rng = np.random.default_rng(20261016)
    rng.shuffle(groups)
    group_fold = {g: i % 5 for i, g in enumerate(groups)}
    folds = []
    for f in range(5):
        tr = [r for r in rows if group_fold[r['group']] != f]
        te = [r for r in rows if group_fold[r['group']] == f]
        spec, fit_score = fit_fold(tr)
        fold = {'fold': f, 'n': len(te), 'train_n': len(tr), 'selected': spec,
                'train_score': fit_score,
                'raw': eval_rows(te, {'kind': 'raw'}),
                'head': eval_rows(te, {'kind': 'head'}),
                'selected_score': eval_rows(te, spec)}
        folds.append(fold)

    full_spec, full_fit = fit_full(rows)
    crossfit_selected = float(np.mean([f['selected_score'] for f in folds]))
    crossfit_raw = float(np.mean([f['raw'] for f in folds]))
    crossfit_head = float(np.mean([f['head'] for f in folds]))

    # Apply the full-calibration rule to the test T3 language output. T2 and
    # all non-T3 rows are copied byte-for-byte from the compliant parent.
    test_q = {r['qa_id']: r for r in (json.loads(x) for x in args.test_questions.read_text(encoding='utf8').splitlines())}
    test_lang = {r['qa_id']: r for r in (json.loads(x) for x in args.test_language.read_text(encoding='utf8').splitlines())}
    test_areas = json.loads(args.test_areas.read_text(encoding='utf8'))
    parent = [json.loads(x) for x in args.parent.read_text(encoding='utf8').splitlines()]
    out = []
    changed = 0
    for row in parent:
        qid = row['qa_id']
        qrow = test_q[qid]
        q = question(qrow)
        ans = json.loads(row['answer'])
        if kind(q) == 'T3':
            crop = [c for c in CROPS if c in q.split('A.')[0]]
            if len(crop) != 1:
                raise ValueError('Test T3 crop parse failed: ' + qid)
            ci = CROPS.index(crop[0])
            lp_crop, lp = parse_t3(test_lang[qid], q)
            if lp_crop != crop[0]:
                raise ValueError('Test language crop differs from question: ' + qid)
            hp = float(test_areas[qid][ci])
            if full_spec['kind'] == 'raw':
                value = lp
            elif full_spec['kind'] == 'head':
                value = hp
            elif full_spec['kind'] == 'geo':
                value = geometric(lp, hp, full_spec['alpha'][ci], full_spec['scale'])
            else:
                value = arithmetic(lp, hp, full_spec['alpha'][ci], full_spec['scale'])
            ans = {'crop': crop[0], 'area_km2': round(float(value), 2)}
            changed += 1
        out.append({'qa_id': qid, 'answer': json.dumps(ans, ensure_ascii=False, separators=(',', ':'))})
    args.out.write_text('\n'.join(json.dumps(x, ensure_ascii=False) for x in out) + '\n', encoding='utf8')
    report = {
        'calibration_source': str(args.calib_language.resolve()),
        'calibration_images': len({r['group'] for r in rows}),
        'calibration_t3_rows': len(rows),
        'unseen_training_images_only': True,
        'test_labels_used': False,
        'single_checkpoint': 'weights/v973',
        'external_classifier': False,
        'task_file_splicing': False,
        't2_threshold_km2': 1.0,
        'crossfit_raw_t3': crossfit_raw,
        'crossfit_head_t3': crossfit_head,
        'crossfit_selected_t3': crossfit_selected,
        'full_calibration_fit_t3': full_fit,
        'selected_spec': full_spec,
        'folds': folds,
        'parent_submission': str(args.parent.resolve()),
        'only_t3_changed': True,
        'changed_rows': changed,
        'candidate': str(args.out.resolve()),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
