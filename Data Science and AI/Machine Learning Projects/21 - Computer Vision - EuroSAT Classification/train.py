import argparse, os, random
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import StratifiedShuffleSplit

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# ---------- utility ----------
def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

class GaussianNoise(object):
    def _init_(self,std=0.01): self.std=std
    def _call_(self,x): return x+torch.randn_like(x)*self.std

# ---------- model ----------
def conv3x3(inp,out,stride=1): return nn.Conv2d(inp,out,3,stride,1,bias=False)

class BasicBlock(nn.Module):
    expansion=1
    def _init_(self,inplanes,planes,stride=1,downsample=None):
        super()._init_()
        self.conv1=conv3x3(inplanes,planes,stride); self.bn1=nn.BatchNorm2d(planes)
        self.relu=nn.ReLU(inplace=True)
        self.conv2=conv3x3(planes,planes); self.bn2=nn.BatchNorm2d(planes)
        self.downsample=downsample
    def forward(self,x):
        id=x
        o=self.relu(self.bn1(self.conv1(x)))
        o=self.bn2(self.conv2(o))
        if self.downsample is not None: id=self.downsample(x)
        return self.relu(o+id)

class ResNet_CIFAR(nn.Module):
    def _init_(self,block,layers,num_classes=10,width=1.0,dropout=0.2):
        super()._init_()
        w=[int(64*width),int(128*width),int(256*width),int(512*width)]
        self.inplanes=w[0]
        self.conv1=nn.Conv2d(3,w[0],3,1,1,bias=False); self.bn1=nn.BatchNorm2d(w[0]); self.relu=nn.ReLU(True)
        self.layer1=self._make_layer(block,w[0],layers[0],1)
        self.layer2=self._make_layer(block,w[1],layers[1],2)
        self.layer3=self._make_layer(block,w[2],layers[2],2)
        self.layer4=self._make_layer(block,w[3],layers[3],2)
        self.avg=nn.AdaptiveAvgPool2d(1)
        self.drop=nn.Dropout(dropout) if dropout>0 else nn.Identity()
        self.fc=nn.Linear(w[3],num_classes)
    def _make_layer(self,block,p,b,stride):
        down=None
        if stride!=1 or self.inplanes!=p:
            down=nn.Sequential(nn.Conv2d(self.inplanes,p,1,stride,bias=False),nn.BatchNorm2d(p))
        layers=[block(self.inplanes,p,stride,down)]; self.inplanes=p
        for _ in range(1,b): layers.append(block(self.inplanes,p))
        return nn.Sequential(*layers)
    def forward(self,x):
        x=self.relu(self.bn1(self.conv1(x)))
        for l in [self.layer1,self.layer2,self.layer3,self.layer4]: x=l(x)
        x=self.avg(x).flatten(1); x=self.drop(x); return self.fc(x)

def resnet18_cifar(num_classes,width=1.0,dropout=0.2):
    return ResNet_CIFAR(BasicBlock,[2,2,2,2],num_classes,width,dropout)

# ---------- data ----------
def make_tf(aug=True):
    if aug:
        t=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(0.25,0.25,0.25,0.02),
            transforms.ToTensor(),
            GaussianNoise(0.01)
        ])
    else:
        t=transforms.ToTensor()
    return t, transforms.ToTensor()

def split_80_10_10(dataset, seed=42):
    """Returns (train,val,test subsets) + (train_idx,val_idx,test_idx) using stratified splits."""
    targets=[s[1] for s in dataset.samples]
    # first: 80% train, 20% temp
    sss1=StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=seed)
    train_idx, temp_idx = next(sss1.split(np.zeros(len(targets)), targets))
    # second: split temp (20%) into val (10%) and test (10%) → so 50/50 of temp
    temp_tgts=[targets[i] for i in temp_idx]
    sss2=StratifiedShuffleSplit(n_splits=1,test_size=0.5,random_state=seed)
    val_subidx, test_subidx = next(sss2.split(np.zeros(len(temp_tgts)), temp_tgts))
    val_idx  = [temp_idx[i] for i in val_subidx]
    test_idx = [temp_idx[i] for i in test_subidx]
    return Subset(dataset,train_idx), Subset(dataset,val_idx), Subset(dataset,test_idx), train_idx, val_idx, test_idx

# ---------- training helpers ----------
def train_epoch(model,loader,opt,dev,crit,scaler):
    model.train(); tl,acc,cnt=0,0,0
    for x,y in tqdm(loader,leave=False,desc="train"):
        x,y=x.to(dev),y.to(dev)
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda",enabled=(dev=="cuda")):
            out=model(x); loss=crit(out,y)
        if scaler:
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        else:
            loss.backward(); opt.step()
        tl+=loss.item()*x.size(0); acc+=(out.argmax(1)==y).sum().item(); cnt+=x.size(0)
    return tl/cnt, acc/cnt

@torch.no_grad()
def evaluate(model,loader,dev,crit):
    model.eval(); tl,acc,cnt=0,0,0
    for x,y in tqdm(loader,leave=False,desc="val"):
        x,y=x.to(dev),y.to(dev); out=model(x); l=crit(out,y)
        tl+=l.item()*x.size(0); acc+=(out.argmax(1)==y).sum().item(); cnt+=x.size(0)
    return tl/cnt, acc/cnt

# ---------- main ----------
def main():
    a=argparse.ArgumentParser()
    a.add_argument("--data_dir",required=True)
    a.add_argument("--epochs",type=int,default=80)
    a.add_argument("--batch_size",type=int,default=64)
    a.add_argument("--lr",type=float,default=1e-3)
    a.add_argument("--weight_decay",type=float,default=1e-4)
    a.add_argument("--width",type=float,default=1.0)
    a.add_argument("--dropout",type=float,default=0.2)
    a.add_argument("--seed",type=int,default=42)
    a.add_argument("--out_dir",default="outputs")
    args=a.parse_args()

    os.makedirs(args.out_dir,exist_ok=True)
    set_seed(args.seed)
    dev="cuda" if torch.cuda.is_available() else "cpu"
    print("device:",dev)

    tr_tf, te_tf = make_tf(True)
    full = datasets.ImageFolder(args.data_dir, transform=tr_tf)

    train,val,test,idx_tr,idx_va,idx_te = split_80_10_10(full, seed=args.seed)
    # ensure val/test use only test-time transforms
    val.dataset.transform = te_tf
    test.dataset.transform = te_tf

    # save split indices for reproducibility
    np.savez(os.path.join(args.out_dir,"splits_80_10_10.npz"),
             train=np.array(idx_tr), val=np.array(idx_va), test=np.array(idx_te))

    print(f"Split sizes: train={len(train)}, val={len(val)}, test={len(test)}")

    dl_tr = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    dl_va = DataLoader(val,   batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    ncls = len(full.classes)
    model = resnet18_cifar(ncls, width=args.width, dropout=args.dropout).to(dev)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-6)
    crit = nn.CrossEntropyLoss(label_smoothing=0.05)
    scaler = torch.cuda.amp.GradScaler(enabled=(dev=="cuda"))

    best, best_ep = 0.0, -1
    for ep in range(1, args.epochs+1):
        print(f"\nEpoch {ep}/{args.epochs}")
        tr_l,tr_a = train_epoch(model, dl_tr, opt, dev, crit, scaler)
        va_l,va_a = evaluate(model, dl_va, dev, crit)
        sch.step()
        print(f"train {tr_l:.4f}|{tr_a:.4f}  val {va_l:.4f}|{va_a:.4f}")

        if va_a > best:
            best, best_ep = va_a, ep
            torch.save({"model":model.state_dict(),"classes":full.classes},
                       os.path.join(args.out_dir,"best.pt"))
            print("✓ saved best model")

    print(f"\nBest val acc {best:.4f} (epoch {best_ep})")
    torch.save(model.state_dict(), os.path.join(args.out_dir,"last.pt"))

if __name__=="__main__":
    main()