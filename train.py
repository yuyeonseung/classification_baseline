import random
import pandas as pd
import numpy as np
import os
import cv2

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

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
 
warnings.filterwarnings(action='ignore') 

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def main(args, device, weight_path):
    df = pd.read_csv(args.data)
    train, val, _, _ = train_test_split(df, df['label'], test_size=0.2, random_state=CFG['SEED'])
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
    train_dataset = CustomDataset(train['image_path'].values, train['label'].values, train_transform)
    train_loader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle=True, num_workers=0)

    val_dataset = CustomDataset(val['image_path'].values, val['label'].values, test_transform)
    val_loader = DataLoader(val_dataset, batch_size = args.batch_size, shuffle=False, num_workers=0)



    model = BaseModel(num_classes=num_class)

    print("parameter: ", sum(p.numel() for p in model.parameters() if p.requires_grad))
    if args.pretrain != '':
        model.load_state_dict(torch.load(args.pretrain))

    optimizer = torch.optim.Adam(params = model.parameters(), lr = args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2,threshold_mode='abs',min_lr=1e-8, verbose=True)
    train(model, optimizer, train_loader, val_loader, scheduler, device, weight_path)



def train(model, optimizer, train_loader, val_loader, scheduler, device, weight_path):
    model.to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    
    best_val_score = 0
    best_model = None
    
    for epoch in range(1, args.epochs+1):
        model.train()
        train_loss = []
        pbar =  tqdm(iter(train_loader))
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
                    
        _val_loss, _val_score = validation(model, criterion, val_loader, device, weight_path)
        _train_loss = np.mean(train_loss)
        print(f'Epoch [{epoch}], Train Loss : [{_train_loss:.5f}] Val Loss : [{_val_loss:.5f}] Val F1 : [{_val_score:.5f}]')
        
        if scheduler is not None:
            scheduler.step(_val_score)
            
        if best_val_score < _val_score:
            best_val_score = _val_score
            best_model = model
        torch.save(best_model.state_dict(), os.path.join(weight_path,'best.pth'))
    torch.save(best_model.state_dict(), os.path.join(weight_path,'last.pth'))
    return best_model

def validation(model, criterion, val_loader, device, weight_path):
    model.eval()
    val_loss = []
    preds, trues = [], []
    
    with torch.no_grad():
        for videos, labels in tqdm(iter(val_loader)):
            videos = videos.to(device)
            labels = labels.to(device)
            
            logit = model(videos)
            
            loss = criterion(logit, labels)
            
            val_loss.append(loss.item())
            
            preds += logit.argmax(1).detach().cpu().numpy().tolist()
            trues += labels.detach().cpu().numpy().tolist()
        
        _val_loss = np.mean(val_loss)
    
    print(metrics.classification_report(trues,preds,digits=3))
    _val_score = metrics.f1_score(trues, preds, average='macro')    
    _val_acc = metrics.accuracy_score(trues, preds)
    return _val_loss, _val_score,_val_acc


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
    args = parser.parse_args()
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    weight_path =  f'weight/{case}/{datetime.datetime.now()}'
    os.makedirs(weight_path, exist_ok = True)

    seed_everything(args.seed)
    main(args, device, weight_path)