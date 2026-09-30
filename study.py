"""Per-run atomic manifests; tune on validation only, then evaluate frozen grid."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
from pathlib import Path
import argparse, hashlib, json, platform, random, subprocess, time
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from data_pipeline import prepare, finalize
from models import LSTMForecaster, TransformerForecaster, count_params

SEEDS=[42,123,456,789,2024,7,88,314,1004,9999]

def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(exist_ok=True,parents=True)
    tmp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');tmp.replace(path)

def grid():
    jobs=set()
    for l in [24,48,96,168,336]:jobs.add(('ETTh1',l,24,1.))
    for ds in ['ETTh1','ETTm1','Exchange','Traffic']:jobs.add((ds,96,24,1.))
    for r in [.3,.5,.7,1.]:jobs.add(('ETTh1',96,24,r))
    for h in [12,24,48,96,192]:jobs.add(('ETTh1',96,h,1.))
    return [dict(dataset=d,lookback=l,horizon=h,ratio=r) for d,l,h,r in sorted(jobs)]

class Windows(Dataset):
    def __init__(self,arr,origins,L,H):self.arr=torch.from_numpy(arr);self.origins=origins;self.L=L;self.H=H
    def __len__(self):return len(self.origins)
    def __getitem__(self,i):
        o=int(self.origins[i]);return self.arr[o-self.L:o],self.arr[o:o+self.H]

def loader(arr,origins,L,H,batch,shuffle=False,seed=42):
    gen=torch.Generator().manual_seed(seed)
    return DataLoader(Windows(arr,origins,L,H),batch_size=batch,shuffle=shuffle,num_workers=0,generator=gen)

def make_model(name,H):
    return (LSTMForecaster if name=='LSTM' else TransformerForecaster)(n_features=7,horizon=H)

def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False

def code_hash():
    h=hashlib.sha256()
    for name in ['data_pipeline.py','models.py','study.py']:
        p=Path(__file__).parent/name;h.update(p.name.encode());h.update(p.read_bytes())
    return h.hexdigest()

def environment(device):
    return dict(python=platform.python_version(),os=platform.platform(),torch=torch.__version__,numpy=np.__version__,cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),device=device,gpu=torch.cuda.get_device_name(device) if device.startswith('cuda') else None)

@torch.no_grad()
def evaluate(model,batches,device,scale,ref):
    model.eval();sums=dict(abs=np.zeros(7),sq=np.zeros(7),count=0)
    for x,y in batches:
        pred=model(x.to(device)).cpu().numpy();target=y.numpy()
        # Mean cancels; restore errors to raw units using this run's scaler.
        e=(pred.astype(np.float64)-target)*np.asarray(scale)
        sums['abs']+=np.abs(e).sum((0,1));sums['sq']+=(e*e).sum((0,1));sums['count']+=len(x)*y.shape[1]
    return finalize(sums,ref)

def run(config,args,stage):
    arr,z,meta,vo,to=prepare(config['dataset'],config['ratio'])
    config=dict(config,max_epochs=2 if args.smoke else 100,patience=1 if args.smoke else 10,batch_size=32,grad_clip=1.,smoke=args.smoke,stage=stage,code_sha256=code_hash(),data_sha256=meta['sha256'])
    cid=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()[:20]
    root=Path('runs')/('smoke' if args.smoke else 'full')/stage
    path=root/(cid+'.json')
    if path.exists():return json.loads(path.read_text(encoding='utf-8'))
    seed_all(config['seed']);device=args.device
    torch.set_num_threads(args.cpu_threads)
    L=config['lookback'];H=config['horizon']
    train_origins=np.arange(meta['train_start']+L,meta['train_end']-H+1)
    if args.smoke:train_origins=train_origins[:128];vo=vo[:64];to=to[:64]
    tr=loader(z,train_origins,L,H,32,True,config['seed']);va=loader(z,vo,L,H,args.eval_batch)
    model=make_model(config['model'],H).to(device);opt=torch.optim.Adam(model.parameters(),lr=config['lr'])
    print(json.dumps(dict(event='run_started',run_id=cid,stage=stage,dataset=config['dataset'],model=config['model'],seed=config['seed'],lr=config['lr'],lookback=L,horizon=H,ratio=config['ratio'])),flush=True)
    best=float('inf');best_state=None;best_epoch=0;bad=0;history=[];start=time.perf_counter()
    if device.startswith('cuda'):torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(config['max_epochs']):
        model.train();loss_total=0.;elements=0
        for x,y in tr:
            x=x.to(device);y=y.to(device);opt.zero_grad(set_to_none=True)
            pred=model(x);loss=((pred-y)**2).mean()
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
            loss_total+=loss.item()*y.numel();elements+=y.numel()
        vm=evaluate(model,va,device,meta['scale'],meta['reference_scale']);score=vm['rmse']**2
        history.append(dict(epoch=epoch+1,train_mse=loss_total/elements,val_reference_mse=score))
        if (epoch+1)%10==0:print(json.dumps(dict(event='epoch',run_id=cid,epoch=epoch+1,val_reference_mse=score)),flush=True)
        if score<best-1e-6:
            best=score;best_epoch=epoch+1;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};bad=0
        else:bad+=1
        if bad>=config['patience']:break
    if best_state is None:raise RuntimeError('No valid checkpoint')
    model.load_state_dict(best_state)
    result=dict(run_id=cid,status='complete',config=config,data=meta,environment=environment(device),parameters=count_params(model),best_epoch=best_epoch,epochs=len(history),train_seconds=time.perf_counter()-start,peak_gpu_bytes=torch.cuda.max_memory_allocated(device) if device.startswith('cuda') else None,history=history,val_reference_mse=best,train_windows=len(train_origins),validation_windows=len(vo),test_windows=len(to))
    if stage=='main':
        result['test']=evaluate(model,loader(z,to,L,H,args.eval_batch),device,meta['scale'],meta['reference_scale'])
        ck=Path('checkpoints')/('smoke' if args.smoke else 'full');ck.mkdir(exist_ok=True,parents=True)
        torch.save(dict(state_dict=best_state,config=config,data=meta),ck/(cid+'.pt'))
    atomic_json(path,result)
    print(json.dumps(dict(run=cid,dataset=config['dataset'],model=config['model'],seed=config['seed'],stage=stage,val=best,test=result.get('test',{}).get('mae'),seconds=result['train_seconds'])),flush=True)
    return result

def select_tuning(smoke=False):
    root=Path('runs')/('smoke' if smoke else 'full')/'tune';records=[json.loads(p.read_text()) for p in root.glob('*.json')]
    chosen={}
    expected=1 if smoke else 3
    for ds in ['ETTh1','ETTm1','Exchange','Traffic']:
        for m in ['LSTM','Transformer']:
            candidates=[]
            for lr in [1e-4,1e-3]:
                rr=[r for r in records if r['config']['dataset']==ds and r['config']['model']==m and r['config']['lr']==lr and r['config']['code_sha256']==code_hash()]
                if len(rr)!=expected:raise RuntimeError(f'Incomplete or ambiguous tuning {ds} {m} {lr}: {len(rr)}')
                candidates.append((float(np.mean([r['val_reference_mse'] for r in rr])),lr))
            chosen[ds+'/'+m]=min(candidates)[1]
    atomic_json(Path('runs')/('smoke' if smoke else 'full')/'selected_lr.json',dict(code_sha256=code_hash(),selection='minimum mean validation reference MSE',learning_rates=chosen))
    return chosen

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['tune','select','main','manifest'],default='manifest');ap.add_argument('--device',default='cuda:0');ap.add_argument('--worker',type=int,default=0);ap.add_argument('--workers',type=int,default=1);ap.add_argument('--smoke',action='store_true');ap.add_argument('--dataset');ap.add_argument('--eval-batch',type=int,default=64);ap.add_argument('--cpu-threads',type=int,default=4);ap.add_argument('--limit',type=int)
    args=ap.parse_args()
    if not 0<=args.worker<args.workers:raise ValueError('Invalid worker index')
    if args.stage=='manifest':print(json.dumps(dict(conditions=grid(),unique_conditions=len(grid()),main_neural_runs=len(grid())*2*10,tuning_neural_runs=48),indent=2));return
    if args.stage=='select':print(select_tuning(args.smoke));return
    seeds=SEEDS[:1] if args.smoke else SEEDS
    jobs=[]
    if args.stage=='tune':
        for ds in ['ETTh1','ETTm1','Exchange','Traffic']:
            for model in ['LSTM','Transformer']:
                for lr in [1e-4,1e-3]:
                    for seed in seeds[:3]:jobs.append(dict(dataset=ds,lookback=96,horizon=24,ratio=1.,model=model,lr=lr,seed=seed))
    else:
        selected=json.loads((Path('runs')/('smoke' if args.smoke else 'full')/'selected_lr.json').read_text())
        if selected['code_sha256']!=code_hash():raise RuntimeError('Code changed after tuning; revalidate version')
        for c in grid():
            for seed in seeds:
                for model in ['LSTM','Transformer']:jobs.append(dict(c,model=model,seed=seed,lr=selected['learning_rates'][c['dataset']+'/'+model]))
    if args.dataset:jobs=[x for x in jobs if x['dataset']==args.dataset]
    jobs=jobs[args.worker::args.workers]
    if args.limit:jobs=jobs[:args.limit]
    for config in jobs:run(config,args,args.stage)

if __name__=='__main__':main()
