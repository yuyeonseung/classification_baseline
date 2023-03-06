import random
import pandas as pd
import numpy as np
import os
import cv2

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.distributed as dist

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
from sklearn import metrics 

from dataset import CustomDataset

from model import BaseModel
import datetime

import argparse
import warnings
import wandb
 
warnings.filterwarnings(action='ignore') 

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def validation(model, criterion, valid_loader, device, weight_path):
    model.eval()
    val_loss = []
    preds, trues = [], []
    
    with torch.no_grad():
        for videos, labels in tqdm(iter(valid_loader),disable=global_rank not in [-1, 0]):
            videos = videos.to(device)
            labels = labels.to(device)
            
            logit = model(videos)
            
            loss = criterion(logit, labels)
            
            val_loss.append(loss.item())
            
            preds += logit.argmax(1).detach().cpu().numpy().tolist()
            trues += labels.detach().cpu().numpy().tolist()
        
        _val_loss = np.mean(val_loss)
    
    if global_rank in [-1,0]:
        print(metrics.classification_report(trues,preds,digits=3))

    _val_score = metrics.f1_score(trues, preds, average='macro')    
    _val_acc = metrics.accuracy_score(trues, preds)

    return _val_loss, _val_score,_val_acc



def train(model, optimizer, train_loader, valid_loader, scheduler, device, weight_path):
    model.to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    
    best_val_score = 0
    best_model = None
    
    for epoch in range(1, args.epochs+1):
        model.train()
        train_loss = []
        pbar =  tqdm(iter(train_loader),
                    disable=global_rank not in [-1, 0])
        for imgs, labels in pbar :
            imgs = imgs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            output = model(imgs)
            loss = criterion(output, labels)
            pbar.set_description(f"train_loss : {loss:.5f}")
            
            loss.backward()
            optimizer.step()
            
            train_loss.append(loss.item())
                    
        _val_loss, _val_score, _val_acc = validation(model, criterion, valid_loader, device, weight_path)
        _train_loss = np.mean(train_loss)
        
        if global_rank in [-1,0]:
            print(f'Epoch [{epoch}], Train Loss : [{_train_loss:.5f}] Val Loss : [{_val_loss:.5f}] Val F1 : [{_val_score:.5f}] Val Acc : [{_val_acc:.5f}]')
            
        if global_rank in [-1,0]and args.use_wandb:
            wandb.log({'train_loss': _train_loss, 'valid_loss': _val_loss, 'val_acc': _val_acc,
                'lr': optimizer.param_groups[0]['lr'], 'val_score': _val_score})

        if scheduler is not None:
            scheduler.step(_val_score)
            
        if best_val_score < _val_score:
            best_val_score = _val_score
            best_model = model
            
        if global_rank in [-1,0]:
            torch.save(best_model.state_dict(), os.path.join(weight_path,'best.pth'))
            
    if global_rank in [-1,0]:
        torch.save(best_model.state_dict(), os.path.join(weight_path,'last.pth'))
    return best_model

def main(args, device, weight_path):
    df = pd.read_csv(args.data)
    train_data, valid_data, _, _ = train_test_split(df, df['label'], test_size=0.2, random_state=args.seed)
    train_transform = A.Compose([
                                A.Resize(args.img_size,args.img_size),
                                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False, p=1.0),
                                ToTensorV2()
                                ])

    test_transform = A.Compose([
                                A.Resize(args.img_size,args.img_size),
                                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False, p=1.0),
                                ToTensorV2()
                                ])

    train_dataset = CustomDataset(train_data['image_path'].values, train_data['label'].values, train_transform)
    valid_dataset = CustomDataset(valid_data['image_path'].values, valid_data['label'].values, test_transform)

    if global_rank  == -1:
        train_sampler = torch.utils.data.RandomSampler(train_dataset)
        valid_sampler = torch.utils.data.SequentialSampler(valid_dataset)
        train_loader = DataLoader(dataset=train_dataset, sampler=train_sampler, shuffle=False, batch_size=args.batch_size)
        valid_loader = DataLoader(dataset=valid_dataset, sampler=valid_sampler, shuffle=False, batch_size=args.batch_size)
    else:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset)
        train_loader = DataLoader(dataset=train_dataset, sampler=train_sampler, shuffle=False, batch_size=args.batch_size)
        valid_loader = DataLoader(dataset=valid_dataset, sampler=valid_sampler, shuffle=False, batch_size=args.batch_size)

    model = BaseModel(num_classes=args.num_class)



    if global_rank in [-1,0] and args.use_wandb:
        wandb.watch(model)
    
    if global_rank in [-1,0]:
        print("parameter: ", sum(p.numel() for p in model.parameters() if p.requires_grad))
    if args.pretrain != '':
        model.load_state_dict(torch.load(args.pretrain))

    optimizer = torch.optim.Adam(params = model.parameters(), lr = args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2,threshold_mode='abs',min_lr=1e-8, verbose=True)
    train(model, optimizer, train_loader, valid_loader, scheduler, device, weight_path)





if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--seed', type=int, default=41)
    parser.add_argument('--pretrain', type=str, default='')
    parser.add_argument('--num_class', type=int, default=2)
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument("--local_rank", type=int, default=-1)

    args = parser.parse_args()
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    weight_path =  f'weight/{datetime.datetime.now()}'
    os.makedirs(weight_path, exist_ok = True)

    seed_everything(args.seed)


    if args.local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl',
                                             timeout=datetime.timedelta(minutes=60))
        args.n_gpu = 1
        global_rank = torch.distributed.get_rank()




    if global_rank in [-1,0] and args.use_wandb:
        wandb.init(
            project="block",
            config=vars(args)
        )
        wandb.watch(model)
        wandb.config.update(config)
    main(args, device, weight_path)