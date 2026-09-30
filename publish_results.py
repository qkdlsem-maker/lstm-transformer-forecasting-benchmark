"""Verify final statistical CSVs independently and export manuscript tables."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

def main():
    records = [json.loads(p.read_text()) for p in Path('runs/full/main').glob('*.json')]
    if len(records) != 300:
        raise RuntimeError(f'Final publication requires 300 runs, found {len(records)}')
    keys = ['dataset','lookback','horizon','ratio']
    raw = pd.DataFrame([{k:r['config'][k] for k in keys+['model','seed']} |
                       {k:r['test'][k] for k in ['mae','rmse']} |
                       {k:r[k] for k in ['parameters','train_seconds','peak_gpu_bytes','best_epoch']}
                       for r in records])
    summary = raw.groupby(keys+['model']).agg(n=('seed','count'),mae_mean=('mae','mean'),
        mae_sd=('mae','std'),rmse_mean=('rmse','mean'),rmse_sd=('rmse','std'),
        parameters=('parameters','first'),train_seconds_mean=('train_seconds','mean'),
        peak_gpu_bytes_max=('peak_gpu_bytes','max'),best_epoch_mean=('best_epoch','mean')).reset_index()
    assert len(summary) == 30 and (summary.n == 10).all()
    published = pd.read_csv('results/paired_statistics.csv').set_index(keys)
    assert len(published) == 15 and published.complete.all()
    reconstructed = []
    for condition,g in raw.groupby(keys):
        paired = g.pivot(index='seed', columns='model', values='mae').dropna()
        assert len(paired) == 10
        a = paired.LSTM.to_numpy(); b = paired.Transformer.to_numpy(); d = a-b
        t = d.mean()/(d.std(ddof=1)/np.sqrt(10))
        p = 2*stats.t.sf(abs(t),9)
        w = stats.wilcoxon(d,method='exact').pvalue
        width = stats.t.ppf(.975,9)*d.std(ddof=1)/np.sqrt(10)
        computed = dict(lstm_mae=a.mean(),lstm_sd=a.std(ddof=1),transformer_mae=b.mean(),
            transformer_sd=b.std(ddof=1),paired_difference=d.mean(),ci_low=d.mean()-width,
            ci_high=d.mean()+width,paired_t_p=p,wilcoxon_p=w,dz=d.mean()/d.std(ddof=1),
            improvement_pct=100*d.mean()/a.mean())
        for name,value in computed.items():
            assert np.isclose(published.loc[condition,name],value,atol=1e-12,rtol=1e-9), (condition,name)
        reconstructed.append(dict(zip(keys,condition))|computed)
    audit = pd.DataFrame(reconstructed)
    # Direct Holm step-down calculation independent of statsmodels implementation.
    for source,target in [('paired_t_p','t_holm_p'),('wilcoxon_p','wilcoxon_holm_p')]:
        order = np.argsort(audit[source].to_numpy())
        adjusted_sorted = np.minimum(1,np.maximum.accumulate(audit[source].to_numpy()[order]*np.arange(15,0,-1)))
        adjusted = np.empty(15); adjusted[order] = adjusted_sorted
        audit[target] = adjusted
        for _,r in audit.iterrows():
            condition = tuple(r[k] for k in keys)
            assert np.isclose(published.loc[condition,target],r[target],atol=1e-12,rtol=1e-9)
    baselines = json.loads(Path('runs/full/baselines.json').read_text())
    assert len(baselines) == 44
    baseline_table = pd.DataFrame([b['condition'] | dict(model=b['model'],mae=b['test']['mae'],
        rmse=b['test']['rmse'],alpha=b['alpha']) for b in baselines])
    assert not baseline_table.duplicated(keys+['model']).any()
    summary.to_csv('results/manuscript_model_summary.csv',index=False)
    baseline_table.to_csv('results/manuscript_baselines.csv',index=False)
    audit.to_csv('results/independent_statistics_audit.csv',index=False)
    report = dict(status='passed',main_runs=300,unique_conditions=15,seeds_per_model=10,
                  baseline_records=44,statistical_checks='means, SD, paired differences, CI, t, Wilcoxon, dz, Holm',
                  caveat='Uncertainty is across training seeds on one fixed split; no future-regime guarantee.')
    Path('results/publication_audit.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))

if __name__ == '__main__':
    main()
