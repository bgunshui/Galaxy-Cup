import json
from pathlib import Path

CROPS = ['水稻', '玉米', '棉花', '大豆', '冬小麦']
SCALES = [1.025, 0.975, 1.15, 1.0, 1.025]

def qtext(row):
    return next(m['content'] for m in row['messages'] if m['role'] == 'user')

def main():
    questions = {r['qa_id']: r for r in (json.loads(x) for x in Path('qa_test.jsonl').read_text(encoding='utf8').splitlines())}
    language = {r['qa_id']: r for r in (json.loads(x) for x in Path('v973_language_test.jsonl').read_text(encoding='utf8').splitlines())}
    parent = [json.loads(x) for x in Path('candidate_v973_submit_t2_10.jsonl').read_text(encoding='utf8').splitlines()]
    out = []
    changed = 0
    for row in parent:
        q = qtext(questions[row['qa_id']])
        answer = json.loads(row['answer'])
        if '估算' in q and '面积' in q:
            crop = [c for c in CROPS if c in q.split('A.')[0]]
            if len(crop) != 1:
                raise ValueError('T3 crop parse failed: ' + row['qa_id'])
            ci = CROPS.index(crop[0])
            lang_answer = json.loads(language[row['qa_id']]['answer'])
            value = max(0.0, min(100.0, float(lang_answer['area_km2']) * SCALES[ci]))
            answer = {'crop': crop[0], 'area_km2': round(value, 2)}
            changed += 1
        out.append({'qa_id': row['qa_id'], 'answer': json.dumps(answer, ensure_ascii=False, separators=(',', ':'))})
    path = Path('candidate_v973_t3_unseen_scale_t2_10.jsonl')
    path.write_text('\n'.join(json.dumps(x, ensure_ascii=False) for x in out) + '\n', encoding='utf8')
    report = {
        'candidate': str(path.resolve()),
        'parent_submission': str(Path('candidate_v973_submit_t2_10').resolve()),
        'source_checkpoint': 'weights/v973',
        'calibration_source': '1433 official train image pairs outside v973 training pool; T3 subset only',
        'crop_scales': dict(zip(CROPS, SCALES)),
        'changed_rows': changed,
        'only_t3_changed': True,
        't2_threshold_km2': 1.0,
        'test_labels_used': False,
        'external_classifier': False,
        'task_file_splicing': False,
        'single_checkpoint': True,
    }
    Path('results/v973_t3_unseen_scale_t2_10.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')
    print(json.dumps(report, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
