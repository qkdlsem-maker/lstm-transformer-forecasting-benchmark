"""Read-only audit of completed manifests; does not alter training or results."""
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

def main():
    root = Path(__file__).resolve().parent
    h = hashlib.sha256()
    for name in ['data_pipeline.py', 'models.py', 'study.py']:
        h.update(name.encode()); h.update((root / name).read_bytes())
    expected_hash = h.hexdigest()
    selected = json.loads((root / 'runs/full/selected_lr.json').read_text())
    assert selected['code_sha256'] == expected_hash
    conditions = {('ETTh1', l, 24, 1.) for l in [24,48,96,168,336]}
    conditions |= {(d,96,24,1.) for d in ['ETTh1','ETTm1','Exchange','Traffic']}
    conditions |= {('ETTh1',96,24,r) for r in [.3,.5,.7,1.]}
    conditions |= {('ETTh1',96,h,1.) for h in [12,24,48,96,192]}
    expected = {c+(m,s) for c in conditions for m in ['LSTM','Transformer']
                for s in [42,123,456,789,2024,7,88,314,1004,9999]}
    assert len(expected) == 300
    baseline_path = root/'runs/full/baselines.json'
    baselines = json.loads(baseline_path.read_text()) if baseline_path.exists() else []
    baseline_by_condition = defaultdict(list)
    for b in baselines:
        bc = tuple(b['condition'][k] for k in ['dataset','lookback','horizon','ratio'])
        baseline_by_condition[bc].append(b)
    seen = set(); hashes = defaultdict(set); counts = Counter()
    origins = defaultdict(set)
    for path in (root / 'runs/full/main').glob('*.json'):
        r = json.loads(path.read_text()); c = r['config']; d = r['data']
        key = tuple(c[k] for k in ['dataset','lookback','horizon','ratio','model','seed'])
        assert key in expected, ('unexpected condition or seed', key)
        assert key not in seen, ('duplicate', key)
        seen.add(key)
        assert r['status'] == 'complete' and not c['smoke']
        assert c['code_sha256'] == expected_hash
        assert c['data_sha256'] == d['sha256']
        assert c['lr'] == selected['learning_rates'][c['dataset']+'/'+c['model']]
        assert (root / 'checkpoints/full' / (r['run_id']+'.pt')).is_file()
        assert 1 <= r['best_epoch'] <= r['epochs'] <= 100
        assert len(r['history']) == r['epochs']
        for metric in ['mae', 'rmse']:
            assert math.isfinite(r['test'][metric]) and r['test'][metric] >= 0
        assert r['test']['rmse'] + 1e-10 >= r['test']['mae']
        test = r['test']; ref = d['reference_scale']
        mae = sum(a/s for a,s in zip(test['channel_mae'],ref))/7
        rmse = math.sqrt(sum((a/s)**2 for a,s in zip(test['channel_rmse'],ref))/7)
        assert abs(mae-test['mae']) < 1e-10
        assert abs(rmse-test['rmse']) < 1e-10
        assert test['target_count_per_channel'] == r['test_windows']*c['horizon']
        for b in baseline_by_condition[key[:4]]:
            assert b['code_sha256'] == expected_hash
            for field in ['sha256','selected_indices','train_start','train_end','mean','scale',
                          'reference_scale','validation_origin_range','test_origin_range']:
                assert b['data'][field] == d[field], ('baseline mismatch', key, field)
            assert b['test']['target_count_per_channel'] == test['target_count_per_channel']
        hashes[c['dataset']].add(d['sha256'])
        origins[c['dataset']].add((r['validation_windows'],r['test_windows']))
        counts[c['dataset']] += 1
    assert all(len(v) == 1 for v in hashes.values()), 'Mixed dataset versions'
    assert all(len(v) == 1 for v in origins.values()), 'Evaluation window counts differ'
    report = dict(main_completed=len(seen), expected_main=300,
                  tuning_completed=len(list((root/'runs/full/tune').glob('*.json'))),
                  by_dataset=dict(counts), code_sha256=expected_hash,
                  data_hashes={k:list(v)[0] for k,v in hashes.items()},
                  audit='passed for all completed manifests',
                  baseline_records=len(baselines),
                  full_complete=seen==expected)
    print(json.dumps(report, indent=2))

if __name__ == '__main__':
    main()
