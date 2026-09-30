from pathlib import Path
import gzip, hashlib, json, urllib.request
import numpy as np
import pandas as pd

SOURCES = {
 'ETTh1': ('https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv',24,'hour'),
 'ETTm1': ('https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTm1.csv',96,'15 minutes'),
 'Exchange': ('https://raw.githubusercontent.com/laiguokun/multivariate-time-series-data/master/exchange_rate/exchange_rate.txt.gz',None,'day'),
 'Traffic': ('https://raw.githubusercontent.com/laiguokun/multivariate-time-series-data/master/traffic/traffic.txt.gz',24,'hour'),
}

def load(name,root='data'):
    url,period,unit=SOURCES[name];root=Path(root);root.mkdir(exist_ok=True,parents=True)
    path=root/url.rsplit('/',1)[-1]
    if not path.exists():
        temporary=path.with_suffix(path.suffix+'.partial')
        urllib.request.urlretrieve(url,temporary);temporary.replace(path)
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    if path.suffix=='.gz':
        with gzip.open(path,'rt') as f: arr=np.loadtxt(f,delimiter=',',dtype=np.float32)
        names=[str(i) for i in range(arr.shape[1])]
    else:
        df=pd.read_csv(path);names=[x for x in df.columns if x!='date'];arr=df[names].to_numpy(np.float32)
    if arr.ndim!=2 or not np.isfinite(arr).all():raise ValueError('Invalid/nonfinite dataset '+name)
    if name=='Traffic' and arr.shape[1]!=862:raise ValueError('Expected original 862-channel Traffic')
    n=len(arr);end=int(n*.7);val_end=end+int(n*.15)
    selected=np.arange(arr.shape[1]) if arr.shape[1]==7 else np.sort(np.argsort(arr[:end].var(0))[-7:])
    meta=dict(dataset=name,url=url,sha256=digest,original_shape=list(arr.shape),selected_indices=selected.tolist(),selected_names=[names[i] for i in selected],train_end=end,val_end=val_end,period=period,step_unit=unit)
    return arr[:,selected].copy(),meta

def prepare(name,ratio=1.,root='data'):
    arr,meta=load(name,root);end=meta['train_end'];ve=meta['val_end']
    start=end-int(end*ratio)
    mean=arr[start:end].mean(0,dtype=np.float64);scale=arr[start:end].std(0,dtype=np.float64)
    scale=np.maximum(scale,1e-8);ref=np.maximum(arr[:end].std(0,dtype=np.float64),1e-8)
    normalized=((arr-mean)/scale).astype(np.float32)
    # Forecast origin is first predicted index. Shared across all lookbacks/horizons.
    val_origins=np.arange(end+336,ve-192+1);test_origins=np.arange(ve+336,len(arr)-192+1)
    if not len(val_origins) or not len(test_origins):raise ValueError('Insufficient evaluation history')
    meta.update(train_start=start,ratio=ratio,mean=mean.tolist(),scale=scale.tolist(),reference_scale=ref.tolist(),validation_origin_range=[int(val_origins[0]),int(val_origins[-1])],test_origin_range=[int(test_origins[0]),int(test_origins[-1])])
    return arr,normalized,meta,val_origins,test_origins

def metric_sums(pred,target,ref):
    err=np.asarray(pred,dtype=np.float64)-np.asarray(target,dtype=np.float64)
    return dict(abs=np.abs(err).sum((0,1)),sq=(err*err).sum((0,1)),count=err.shape[0]*err.shape[1])

def finalize(sums,ref):
    mae=sums['abs']/sums['count'];mse=sums['sq']/sums['count'];ref=np.asarray(ref)
    return dict(mae=float(np.mean(mae/ref)),rmse=float(np.sqrt(np.mean(mse/ref**2))),channel_mae=mae.tolist(),channel_rmse=np.sqrt(mse).tolist(),target_count_per_channel=int(sums['count']))
