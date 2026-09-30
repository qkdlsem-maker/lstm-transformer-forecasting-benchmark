from pathlib import Path
import json, warnings
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.tsa.stattools import adfuller,kpss,acf
from data_pipeline import load

def summarize():
    rows=[]
    for p in Path('runs/full/main').glob('*.json'):
        r=json.loads(p.read_text());c=r['config']
        rows.append({k:c[k] for k in ['dataset','lookback','horizon','ratio','model','seed']}|dict(mae=r['test']['mae'],rmse=r['test']['rmse'],run_id=r['run_id'],code_sha256=c['code_sha256'],best_epoch=r['best_epoch']))
    out=Path('results');out.mkdir(exist_ok=True)
    df=pd.DataFrame(rows)
    if df.empty:print('No completed main runs; results pending.');return
    if df.code_sha256.nunique()!=1:raise ValueError('Mixed code versions')
    keys=['dataset','lookback','horizon','ratio']
    if df.duplicated(keys+['model','seed']).any():raise ValueError('Duplicate condition/model/seed')
    df.to_csv(out/'raw_results.csv',index=False)
    summary=[]
    for key,g in df.groupby(keys):
        pv=g.pivot(index='seed',columns='model',values='mae')
        if not {'LSTM','Transformer'}.issubset(pv.columns):continue
        pv=pv.dropna();a=pv.LSTM.to_numpy();b=pv.Transformer.to_numpy();n=len(a)
        if n<2:continue
        d=a-b;se=d.std(ddof=1)/np.sqrt(n);width=stats.t.ppf(.975,n-1)*se
        summary.append(dict(zip(keys,key))|dict(n=n,complete=n==10,lstm_mae=a.mean(),lstm_sd=a.std(ddof=1),transformer_mae=b.mean(),transformer_sd=b.std(ddof=1),paired_difference=d.mean(),ci_low=d.mean()-width,ci_high=d.mean()+width,improvement_pct=100*d.mean()/a.mean(),paired_t_p=stats.ttest_rel(a,b).pvalue,wilcoxon_p=stats.wilcoxon(d,method='exact').pvalue if np.any(d) else 1.,dz=d.mean()/d.std(ddof=1) if d.std(ddof=1)>0 else np.nan))
    s=pd.DataFrame(summary)
    if not s.empty:
        # Full-family adjustment only when every one of the 15 comparisons is complete.
        if len(s)==15 and s.complete.all():
            s['t_holm_p']=multipletests(s.paired_t_p,method='holm')[1];s['wilcoxon_holm_p']=multipletests(s.wilcoxon_p,method='holm')[1]
        s.to_csv(out/'paired_statistics.csv',index=False)
    print(f'{len(df)}/300 main runs complete; full inferential family requires 300.')

def diagnostics():
    rows=[]
    for ds in ['ETTh1','ETTm1','Exchange','Traffic']:
        arr,meta=load(ds);a=arr[:meta['train_end']]
        for channel in range(7):
            x=a[:,channel].astype(np.float64)
            # Standardization is affine and train-only; makes numeric conditioning explicit.
            x=(x-x.mean())/x.std()
            period=meta['period'];season_acf=float(acf(x,nlags=period,fft=True)[period]) if period else None
            for regression in ['c','ct']:
                ad=adfuller(x,regression=regression,autolag='AIC')
                with warnings.catch_warnings(record=True) as captured:
                    kp=kpss(x,regression=regression,nlags='auto')
                rows.append(dict(dataset=ds,channel=meta['selected_names'][channel],n=len(x),regression=regression,adf_stat=ad[0],adf_p=ad[1],adf_lags=ad[2],kpss_stat=kp[0],kpss_p=kp[1],kpss_lags=kp[2],kpss_p_boundary_warning='; '.join(str(w.message) for w in captured),seasonal_acf=season_acf,data_sha256=meta['sha256']))
    Path('results').mkdir(exist_ok=True);pd.DataFrame(rows).to_csv('results/stationarity_diagnostics.csv',index=False)

if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--diagnostics',action='store_true');args=ap.parse_args()
    if args.diagnostics:diagnostics()
    else:summarize()
