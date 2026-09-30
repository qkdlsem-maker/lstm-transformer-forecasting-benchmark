"""Matched input-gradient sensitivity; not IG or explanation faithfulness."""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
import torch
from data_pipeline import prepare
from study import make_model,seed_all,code_hash

def main(device='cuda:0'):
    rows=[];curves=[]
    _,z,meta,_,origins=prepare('ETTh1')
    chosen=origins[np.linspace(0,len(origins)-1,100,dtype=int)]
    x0=np.stack([z[o-96:o] for o in chosen])
    manifests=[]
    for p in Path('runs/full/main').glob('*.json'):
        r=json.loads(p.read_text());c=r['config']
        if (c['dataset'],c['lookback'],c['horizon'],c['ratio'])==('ETTh1',96,24,1.):manifests.append(r)
    if len(manifests)!=20:raise RuntimeError(f'Need 20 default ETTh1 model/seed checkpoints, found {len(manifests)}')
    for r in manifests:
        c=r['config']
        if c['code_sha256']!=code_hash():raise ValueError('Training code hash mismatch')
        seed_all(c['seed']);m=make_model(c['model'],24).to(device).eval()
        ck=torch.load(Path('checkpoints/full')/(r['run_id']+'.pt'),map_location='cpu',weights_only=True)
        m.load_state_dict(ck['state_dict'])
        all_imp=[]
        for start in range(0,len(x0),10):
            x=torch.tensor(x0[start:start+10],device=device,requires_grad=True)
            with torch.backends.cudnn.flags(enabled=False):
                score=m(x).mean(dim=(1,2)).sum()
                grad=torch.autograd.grad(score,x)[0]
            imp=grad.abs().mean(-1).detach().cpu().numpy()
            all_imp.append(imp/np.clip(imp.sum(1,keepdims=True),1e-12,None))
        imps=np.concatenate(all_imp);rr=imps[:,-19:].sum(1)
        rows.append(dict(model=c['model'],seed=c['seed'],recency_mean=float(rr.mean()),n_origins=len(chosen),uniform_reference=19/96,method='absolute_gradient_of_mean_standardized_prediction',run_id=r['run_id']))
        for t,value in enumerate(imps.mean(0)):curves.append(dict(model=c['model'],seed=c['seed'],time_index=t,mass=float(value)))
    Path('results').mkdir(exist_ok=True);pd.DataFrame(rows).to_csv('results/gradient_sensitivity.csv',index=False);pd.DataFrame(curves).to_csv('results/gradient_curves.csv',index=False)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--device',default='cuda:0');args=ap.parse_args();main(args.device)
