"""Deterministic, train/validation-only selected baselines on identical origins."""
from pathlib import Path
import json
import numpy as np
from sklearn.linear_model import Ridge
from data_pipeline import prepare, finalize
from study import grid, atomic_json, code_hash

def batches(arr,origins,L,H,size=128):
    for start in range(0,len(origins),size):
        oo=origins[start:start+size]
        yield np.stack([arr[o-L:o] for o in oo]),np.stack([arr[o:o+H] for o in oo])

def metrics(z,origins,L,H,predict,meta):
    sums=dict(abs=np.zeros(7),sq=np.zeros(7),count=0)
    for x,y in batches(z,origins,L,H):
        e=(predict(x)-y).astype(np.float64)*np.asarray(meta['scale'])
        sums['abs']+=np.abs(e).sum((0,1));sums['sq']+=(e*e).sum((0,1));sums['count']+=len(x)*H
    return finalize(sums,meta['reference_scale'])

def fit_ridge(z,origins,L,H,alpha):
    # Shared map across channels; accumulated normal equations keep memory bounded.
    xx=np.zeros((L+1,L+1));xy=np.zeros((L+1,H));n=0
    for x,y in batches(z,origins,L,H):
        a=x.transpose(0,2,1).reshape(-1,L).astype(np.float64)
        b=y.transpose(0,2,1).reshape(-1,H).astype(np.float64)
        a=np.column_stack([a,np.ones(len(a))]);xx+=a.T@a;xy+=a.T@b;n+=len(a)
    penalty=np.eye(L+1)*alpha;penalty[-1,-1]=0
    coef=np.linalg.solve(xx+penalty,xy)
    def predict(x):
        a=x.transpose(0,2,1).reshape(-1,L).astype(np.float64)
        a=np.column_stack([a,np.ones(len(a))]);return (a@coef).reshape(len(x),7,H).transpose(0,2,1)
    return predict

def main():
    rows=[]
    for c in grid():
        _,z,meta,vo,to=prepare(c['dataset'],c['ratio']);L=c['lookback'];H=c['horizon'];period=meta['period']
        origins=np.arange(meta['train_start']+L,meta['train_end']-H+1)
        candidates={}
        candidates['Persistence']=lambda x:np.repeat(x[:,-1:,:],H,axis=1)
        if period and L>=period:candidates['SeasonalNaive']=lambda x,p=period:x[:,[-p+i%p for i in range(H)],:]
        validation=[]
        for alpha in [.01,.1,1.,10.,100.]:
            fn=fit_ridge(z,origins,L,H,alpha);score=metrics(z,vo,L,H,fn,meta)['rmse']**2;validation.append((score,alpha))
        alpha=min(validation)[1];candidates['Ridge']=fit_ridge(z,origins,L,H,alpha)
        for name,predict in candidates.items():
            row=dict(condition=c,model=name,test=metrics(z,to,L,H,predict,meta),data=meta,code_sha256=code_hash(),alpha=alpha if name=='Ridge' else None)
            rows.append(row);print(c,name,row['test']['mae'],flush=True)
        atomic_json('runs/full/baselines.json',rows)

if __name__=='__main__':main()
