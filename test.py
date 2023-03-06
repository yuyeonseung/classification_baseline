import random
import pandas as pd
import numpy as np
import os, shutil
import cv2

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import torchvision.models as models

from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

from dataset import CustomDataset
from model import BaseModel
import datetime

import argparse
import warnings
warnings.filterwarnings(action='ignore') 


def test(model, test_loader, device):
    model.to(device)
    model.eval()
    predictions = []
    with torch.no_grad():
        for imgs in tqdm(iter(test_loader)):
            imgs = imgs.float().to(device)
            
            probs = model(imgs)

            probs  = probs.cpu().detach().numpy()
            preds = probs > 0.5
            preds = preds.astype(int)
            predictions += preds.tolist()

    return predictions



def main(args,device) :
    test_data = os.listdir(args.data_root)
    test_data = [os.path.join(args.data_root, x) for x in test]
    test_dataset = CustomDataset(test_data, None, None, CFG)
    test_loader = DataLoader(test_dataset, batch_size = CFG['BATCH_SIZE'], shuffle=False, num_workers=0)

    model = BaseModel(num_classes=2)
    model.load_state_dict(torch.load(crash_path))

    predictions = test(model, test_loader, device)
    print(zip(test_data,predictions))
    if args.output != '':
        os.makedirs(args.output, exist_ok = True)
        for path, result in zip(test_data,predictions) :
            if args.class_list == '':
                shutil.copyfile(test_data,os.path.join(args.output,str(result),test_data.split('/')[-1]))
            else:
                shutil.copyfile(test_data,os.path.join(args.output,args.class_list[result],test_data.split('/')[-1]))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--output', type=str, default='')
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--seed', type=int, default=41)
    parser.add_argument('--pretrain', type=str, default='')
    parser.add_argument('--num_class', type=int, default=2)
    parser.add_argument('--class_list', type=list, default='')

    args = parser.parse_args()
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    main(args, device)