import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--seen-keys', type=Path, required=True)
    parser.add_argument('--keys-out', type=Path, required=True)
    parser.add_argument('--report-out', type=Path, required=True)
    parser.add_argument('--cache-dir', type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding='utf8'))
    seen = set(json.loads(args.seen_keys.read_text(encoding='utf8')))
    train = manifest['train']
    records = {row['key']: row for row in train}
    unknown = seen - set(records)
    if unknown:
        raise ValueError(f'{len(unknown)} seen keys are absent from manifest')
    held = [row for row in train if row['key'] not in seen]
    seen_records = [row for row in train if row['key'] in seen]
    seen_opt = {row['opt'] for row in seen_records}
    seen_sar = {row['sar'] for row in seen_records if row.get('sar')}
    held_opt = {row['opt'] for row in held}
    held_sar = {row['sar'] for row in held if row.get('sar')}
    optical_overlap = seen_opt & held_opt
    sar_overlap = seen_sar & held_sar
    missing_cache = [row['key'] for row in held if not (args.cache_dir / f"{row['key']}.npz").is_file()]
    if optical_overlap or sar_overlap or missing_cache:
        raise ValueError(json.dumps({'optical_overlap': len(optical_overlap),
                                     'sar_overlap': len(sar_overlap),
                                     'missing_cache': len(missing_cache)}))
    keys = [row['key'] for row in held]
    args.keys_out.write_text(json.dumps(keys, ensure_ascii=False, indent=2), encoding='utf8')
    report = {'fit_keys_from_v973_train_pool': len(seen),
              'calibration_keys': len(keys),
              'optical_overlap': 0, 'sar_overlap': 0,
              'missing_cache': 0, 'test_labels_read': False}
    args.report_out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
