from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':8, 'axes.labelsize':8, 'legend.fontsize':7})

def main():
    root=Path('results');figdir=Path('figures');figdir.mkdir(exist_ok=True)
    df=pd.read_csv(root/'raw_results.csv')
    if len(df)!=300:raise RuntimeError('Final figures require all 300 neural runs')
    b=json.loads(Path('runs/full/baselines.json').read_text())
    baseline=pd.DataFrame([x['condition']|dict(model=x['model'],mae=x['test']['mae'],rmse=x['test']['rmse']) for x in b])
    baseline.to_csv(root/'baseline_results.csv',index=False)
    definitions=[('input_length','lookback',dict(dataset='ETTh1',horizon=24,ratio=1.)),('training_history','ratio',dict(dataset='ETTh1',lookback=96,horizon=24)),('forecast_horizon','horizon',dict(dataset='ETTh1',lookback=96,ratio=1.)),('datasets','dataset',dict(lookback=96,horizon=24,ratio=1.))]
    for fname,var,fixed in definitions:
        g=df.copy();bg=baseline.copy()
        for k,v in fixed.items():g=g[g[k]==v];bg=bg[bg[k]==v]
        labels=sorted(g[var].unique());fig,ax=plt.subplots(figsize=(3.2,2.7))
        for model in ['LSTM','Transformer']:
            a=g[g.model==model].groupby(var).mae.agg(['mean','std']).reindex(labels)
            ax.errorbar(range(len(labels)),a['mean'],yerr=a['std'],marker='o',capsize=3,label=model)
        for model in sorted(bg.model.unique()):
            a=bg[bg.model==model].set_index(var).reindex(labels)
            ax.plot(range(len(labels)),a.mae,marker='x',linestyle='--',label=model)
        ax.set_xticks(range(len(labels)),labels);ax.set_ylabel('Reference-standardized MAE');ax.set_xlabel({'lookback':'Input length (steps)','horizon':'Forecast horizon (steps)','ratio':'Training-history ratio','dataset':'Dataset'}[var]);ax.legend(fontsize=6.5);ax.grid(axis='y',alpha=.2);fig.tight_layout();fig.savefig(figdir/(fname+'.png'),dpi=300);plt.close(fig)
    sensitivity=root/'gradient_curves.csv'
    if sensitivity.exists():
        g=pd.read_csv(sensitivity);fig,ax=plt.subplots(figsize=(3.2,2.7))
        for model in ['LSTM','Transformer']:
            a=g[g.model==model].groupby('time_index').mass.agg(['mean','std']);ax.plot(a.index,a['mean'],label=model);ax.fill_between(a.index,a['mean']-a['std'],a['mean']+a['std'],alpha=.15)
        ax.axhline(1/96,color='gray',linestyle=':',label='Uniform mass');ax.set_xlabel('Input time step');ax.set_ylabel('Normalized absolute gradient');ax.legend();fig.tight_layout();fig.savefig(figdir/'gradient_sensitivity.png',dpi=300);plt.close(fig)

if __name__=='__main__':main()
