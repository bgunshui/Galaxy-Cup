import json
from pathlib import Path
import numpy as np

CROPS = ['水稻', '玉米', '棉花', '大豆', '冬小麦']

def qtext(r): return next(m['content'] for m in r['messages'] if m['role'] == 'user')

def score(pred, gold):
    p, g = set(pred), set(gold)
    return 1.0 if not p and not g else 2 * len(p & g) / max(1, len(p) + len(g))

def rows_from(scale_rows, area_map):
    qa = {r['qa_id']: r for r in (json.loads(x) for x in Path('qa_train.jsonl').read_text(encoding='utf8').splitlines())}
    rows = []
    for qid, r in scale_rows.items():
        qrow = qa.get(qid)
        if qrow is None: continue
        gold = json.loads(qrow['messages'][-1]['content'])
        if 'crops' not in gold or qid not in area_map: continue
        rows.append({'qid': qid, 'group': tuple(qrow['images']), 'area': np.asarray(area_map[qid], dtype=float),
                     'gold': set(gold['crops'])})
    return rows

def evaluate(rows, scales):
    vals = []
    for r in rows:
        pred = {c for c, v in zip(CROPS, r['area'] * np.asarray(scales)) if v >= 1.0}
        vals.append(score(pred, r['gold']))
    return float(np.mean(vals)) if vals else float('nan')

def fit(rows):
    grid = np.arange(.80, 1.251, .025)
    scales = [1.0] * 5
    for _ in range(2):
        for ci in range(5):
            best = (-1.0, scales[ci])
            for value in grid:
                trial = scales.copy(); trial[ci] = float(value)
                score_now = evaluate(rows, trial)
                if score_now > best[0] + 1e-12:
                    best = (score_now, float(value))
            scales[ci] = best[1]
    return evaluate(rows, scales), scales

def main():
    lang = {r['qa_id']: r for r in (json.loads(x) for x in Path('v973_unseen_calib_t3.jsonl').read_text(encoding='utf8').splitlines())}
    areas = json.loads(Path('v973_unseen_calib_t3.areas.json').read_text(encoding='utf8'))
    rows = rows_from(lang, areas)
    groups = sorted({r['group'] for r in rows})
    rng = np.random.default_rng(20261016); rng.shuffle(groups)
    fold_of = {g: i % 5 for i, g in enumerate(groups)}
    folds = []
    for f in range(5):
        tr = [r for r in rows if fold_of[r['group']] != f]
        te = [r for r in rows if fold_of[r['group']] == f]
        fit_score, scales = fit(tr)
        folds.append({'fold': f, 'train_rows': len(tr), 'held_rows': len(te), 'scales': scales,
                      'raw': evaluate(te, [1.0] * 5), 'scaled': evaluate(te, scales),
                      'fit_score': fit_score})
    fit_all, scales_all = fit(rows)
    report = {'rows': len(rows), 'groups': len(groups), 'threshold_km2': 1.0,
              'raw': evaluate(rows, [1.0] * 5), 'fit_all': fit_all,
              'scales_all': scales_all, 'crossfit_raw': float(np.mean([f['raw'] for f in folds])),
              'crossfit_scaled': float(np.mean([f['scaled'] for f in folds])), 'folds': folds,
              'unseen_training_images_only': True, 'test_labels_used': False,
              'single_checkpoint': 'weights/v973', 'external_classifier': False}
    Path('results/v973_t2_unseen_calibration.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')
    print(json.dumps(report, ensure_ascii=False, indent=2))

if __name__ == '__main__': main()
