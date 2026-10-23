import json
from pathlib import Path
import numpy as np

CROPS = ['水稻', '玉米', '棉花', '大豆', '冬小麦']
T2_SCALES = [1.2, .925, 1.225, 1.1, .95]

def qtext(row):
    return next(m['content'] for m in row['messages'] if m['role'] == 'user')

def main():
    questions = {r['qa_id']: r for r in (json.loads(x) for x in Path('qa_test.jsonl').read_text(encoding='utf8').splitlines())}
    parent = [json.loads(x) for x in Path('candidate_v973_t3_unseen_scale_t2_10.jsonl').read_text(encoding='utf8').splitlines()]
    area_raw = json.loads(Path('candidate_v973_res500_t4.areas.json').read_text(encoding='utf8'))
    effective = {}
    out = []
    changed_t2 = 0
    for row in parent:
        qid = row['qa_id']; q = qtext(questions[qid]); answer = json.loads(row['answer'])
        raw = np.asarray(area_raw[qid], dtype=float)
        calibrated = (raw * np.asarray(T2_SCALES)).tolist()
        effective[qid] = calibrated
        if ('包含' in q or '哪些' in q) and 'crops' in answer:
            new = {'crops': [c for c, v in zip(CROPS, calibrated) if v >= 1.0]}
            changed_t2 += int(new != answer)
            answer = new
        out.append({'qa_id': qid, 'answer': json.dumps(answer, ensure_ascii=False, separators=(',', ':'))})
    path = Path('candidate_v973_t3_scale_res500_t2_10.jsonl')
    path.write_text('\n'.join(json.dumps(x, ensure_ascii=False) for x in out) + '\n', encoding='utf8')
    Path('candidate_v973_t3_scale_res500_t2_10.areas.json').write_text(json.dumps(effective, ensure_ascii=False), encoding='utf8')
    report = {
        'candidate': str(path.resolve()),
        'parent': str(Path('candidate_v973_t3_unseen_scale_t2_10').resolve()),
        'source_checkpoint': 'weights/v973',
        't2_area_source': 'candidate_v973_res500_t4.areas.json from same v973 checkpoint',
        't2_crop_scales': dict(zip(CROPS, T2_SCALES)),
        't2_threshold_km2': 1.0,
        't2_changed_rows': changed_t2,
        't3_source': 'candidate_v973_t3_unseen_scale_t2_10; independent unseen-train crop scaling',
        'only_t2_replaced_after_parent': True,
        'test_labels_used': False,
        'external_classifier': False,
        'task_file_splicing': False,
        'single_checkpoint': True,
    }
    Path('results/v973_t3_scale_res500_t2_10.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')
    print(json.dumps(report, ensure_ascii=False, indent=2))

if __name__ == '__main__': main()
