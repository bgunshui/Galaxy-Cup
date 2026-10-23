"""Official QA-paired crop segmentation and area regression, patch-aware."""
import argparse, csv, json, os, re, time, random
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import tifffile
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

CROPS = ['水稻','玉米','棉花','大豆','冬小麦']
ROOT = Path(os.environ.get('XINGHE_WORK_ROOT', '.'))

def decode_name(s):
    return re.sub(r'#U([0-9a-fA-F]{4})', lambda m: chr(int(m[1],16)), s)

def catalog(root):
    result = {}
    for p in root.rglob('*.tif'):
        kind = decode_name(p.parent.name)
        if kind not in ('光学影像','SAR影像','MASK'): continue
        result.setdefault((kind,p.name),p)
    return result

def make_records(split):
    base=ROOT/('train' if split=='训练集' else 'val')
    idx=catalog(base)
    qa=list(base.rglob('qa_*.jsonl'))
    if len(qa)!=1: raise RuntimeError(f'QA ambiguous: {qa}')
    seasons={}
    with (ROOT/'seg_data'/'MASK_逐影像类别解释.csv').open(encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            if row['数据集']==split:
                seasons[Path(row['光学影像']).name]=[row['码24当期类别'],row['码26当期类别']]
    records={}; qa_to_pair={}
    for line in qa[0].open(encoding='utf-8'):
        row=json.loads(line); mods={}
        for rel in row.get('images',[]):
            p=Path(rel); kind=decode_name(p.parent.name)
            if kind not in ('光学影像','SAR影像'): raise ValueError(rel)
            if kind in mods: raise ValueError('Repeated modality '+row['qa_id'])
            mods[kind]=idx[(kind,p.name)]
        if '光学影像' not in mods: raise ValueError('Missing optical '+row['qa_id'])
        op=mods['光学影像']; sr=mods.get('SAR影像')
        key=op.stem+'_'+(sr.stem if sr else 'none')
        records[key]={'key':key,'opt':str(op),'sar':str(sr) if sr else None,
                      'mask':str(idx[('MASK',op.name)]),'season':seasons[op.name]}
        qa_to_pair[row['qa_id']]=key
    return list(records.values()), qa_to_pair, str(qa[0])

def band_array(path,n):
    a=tifffile.imread(path)
    if a.shape==(n,1000,1000): a=np.moveaxis(a,0,-1)
    if a.shape!=(1000,1000,n): raise ValueError(f'{path}: {a.shape}, expected 1000x1000x{n}')
    return a

def prepare_one(job):
    record,folder,size=job
    torch.set_num_threads(1)
    target=Path(folder)/(record['key']+'.npz')
    if target.exists(): return str(target)
    a=band_array(record['opt'],12).astype(np.float32)
    valid=np.all(np.isfinite(a)&(a>0),axis=-1)
    mask=tifffile.imread(record['mask'])
    if mask.shape!=(1000,1000): raise ValueError(record['mask'])
    label=np.full(mask.shape,5,np.int64) # explicit non-target background
    for code,ci in [(3,0),(1,1),(2,2),(5,3)]: label[mask==code]=ci
    for code,name in zip((24,26),record['season']):
        if name in CROPS: label[mask==code]=CROPS.index(name)
    soft=np.stack([(label==i)&valid for i in range(6)]).astype(np.float32)
    areas=soft[:5].sum((1,2))*0.0001
    s=np.zeros((1000,1000,2),np.float32); sv=np.zeros((1000,1000),np.float32)
    if record['sar']:
        raw=band_array(record['sar'],2).astype(np.float32)
        good=np.all(np.isfinite(raw)&(raw!=-9999),axis=-1)
        s=np.where(good[...,None],np.clip((raw+35)/40,0,1),0)
        sv=good.astype(np.float32)
    x=np.concatenate([np.clip(np.nan_to_num(a)/10000,0,2),s,valid[...,None],sv[...,None]],-1)
    x=torch.from_numpy(x).permute(2,0,1)[None]
    x=F.interpolate(x,(size,size),mode='area')[0].numpy().astype(np.float16)
    y=F.interpolate(torch.from_numpy(soft)[None],(size,size),mode='area')[0].numpy().astype(np.float16)
    np.savez(target,x=x,y=y,areas=areas)
    return str(target)

class Cached(Dataset):
    def __init__(self,records,folder,augment): self.records=records;self.folder=Path(folder);self.augment=augment
    def __len__(self):return len(self.records)
    def __getitem__(self,i):
        with np.load(self.folder/(self.records[i]['key']+'.npz')) as z:
            x=torch.from_numpy(z['x'].astype(np.float32)); y=torch.from_numpy(z['y'].astype(np.float32)); a=torch.from_numpy(z['areas'].astype(np.float32))
        if self.augment:
            for axis in (1,2):
                if random.random()<.5: x=x.flip(axis);y=y.flip(axis)
        return x,y,a

class Block(nn.Sequential):
    def __init__(self,ci,co):
        super().__init__(nn.Conv2d(ci,co,3,padding=1),nn.GroupNorm(8,co),nn.GELU(),nn.Conv2d(co,co,3,padding=1),nn.GroupNorm(8,co),nn.GELU())
class AreaNet(nn.Module):
    def __init__(self):
        super().__init__();self.a=Block(16,24);self.b=Block(24,48);self.c=Block(48,96);self.d=Block(96+48,48);self.e=Block(48+24,24);self.out=nn.Conv2d(24,6,1)
    def forward(self,x):
        a=self.a(x);b=self.b(F.avg_pool2d(a,2));c=self.c(F.avg_pool2d(b,2))
        d=self.d(torch.cat([F.interpolate(c,size=b.shape[-2:],mode='bilinear',align_corners=False),b],1))
        return self.out(self.e(torch.cat([F.interpolate(d,size=a.shape[-2:],mode='bilinear',align_corners=False),a],1)))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--epochs',type=int,default=16);args=ap.parse_args()
    torch.set_num_threads(4);torch.manual_seed(23);np.random.seed(23);random.seed(23)
    run=ROOT/'area_qa_patch_v2';run.mkdir(exist_ok=True)
    train,trainmap,trainqa=make_records('训练集');val,valmap,valqa=make_records('验证集')
    meta={'train_pairs':len(train),'val_pairs':len(val),'train_qa':len(trainmap),'val_qa':len(valmap),'train':train,'val':val,'train_map':trainmap,'val_map':valmap,'train_source':trainqa,'val_source':valqa,'channels':'12 optical + 2 SAR + optical validity + SAR validity','labels':CROPS+['background'],'epochs':args.epochs}
    (run/'manifest.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
    print('DATA', {k:meta[k] for k in ('train_pairs','val_pairs','train_qa','val_qa')},flush=True)
    for split,recs in [('train',train),('val',val)]:
        folder=run/('cache_'+split);folder.mkdir(exist_ok=True)
        jobs=[(r,str(folder),256) for r in recs]
        with ProcessPoolExecutor(max_workers=4) as pool:
            for i,_ in enumerate(pool.map(prepare_one,jobs),1):
                if i%100==0 or i==len(jobs): print('CACHE',split,i,len(jobs),flush=True)
    dl=DataLoader(Cached(train,run/'cache_train',True),batch_size=8,shuffle=True,num_workers=2,pin_memory=True)
    vl=DataLoader(Cached(val,run/'cache_val',False),batch_size=8,num_workers=2,pin_memory=True)
    m=AreaNet().cuda();opt=torch.optim.AdamW(m.parameters(),lr=3e-4,weight_decay=1e-3)
    best=float('inf');bad=0
    for epoch in range(args.epochs):
        m.train();total=0
        for step,(x,y,a) in enumerate(dl):
            x,y,a=x.cuda(),y.cuda(),a.cuda();opt.zero_grad(set_to_none=True)
            logits=m(x);prob=logits.softmax(1);valid=y.sum(1)
            pred=(prob[:,:5]*valid[:,None]).mean((2,3))*100
            seg=-(y*logits.log_softmax(1)).sum()/valid.sum().clamp_min(1)
            weights=1+(a>=1).float()*2
            area=(F.smooth_l1_loss(torch.log1p(pred),torch.log1p(a),reduction='none')*weights).mean()
            loss=seg+area;loss.backward();nn.utils.clip_grad_norm_(m.parameters(),1);opt.step();total+=loss.item()
            if step%50==0:print('TRAIN',epoch,step,'loss',round(loss.item(),5),flush=True)
        m.eval();preds=[];golds=[]
        with torch.inference_mode():
            for x,y,a in vl:
                p=m(x.cuda()).softmax(1)[:,:5];p=(p*y.sum(1).cuda()[:,None]).mean((2,3))*100
                preds.append(p.cpu());golds.append(a)
        pred=torch.cat(preds);gt=torch.cat(golds);mae=(pred-gt).abs().mean().item()
        pos=gt>=1;positive=(pred-gt).abs()[pos].mean().item()
        msg={'epoch':epoch,'train_loss':total/len(dl),'official_val_area_mae':mae,'positive_area_mae':positive}
        with (run/'metrics.jsonl').open('a') as f:f.write(json.dumps(msg)+chr(10))
        print('EPOCH',json.dumps(msg),flush=True)
        if mae<best:
            best=mae;bad=0;torch.save({'model':m.state_dict(),'optimizer':opt.state_dict(),'epoch':epoch,'metrics':msg},run/'best.pt')
            np.savez(run/'best_val_areas.npz',pred=pred.numpy(),gold=gt.numpy(),keys=np.array([r['key'] for r in val]))
        else: bad+=1
        if bad>=5: print('EARLY_STOP',epoch,flush=True);break
    print('TRAIN_DONE',best,flush=True)
if __name__=='__main__':main()
