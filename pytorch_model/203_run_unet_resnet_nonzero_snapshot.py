import os
import logging
import pickle

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision

from IPython.core.debugger import set_trace
import matplotlib.pyplot as plt

from model import UNetResNet34
#from my_model import UNetResNet34
from metrics import iou_pytorch
from utils import save_checkpoint, load_checkpoint, set_logger, save_checkpoint_snapshot
from dataset import TgsDataSet
from augmentation import get_seed
import imgaug as ia


######### Load TGS salt data #########
def prepare_data():
    # read numpy format data
    with open('../data/processed/dataset_%d.pkl'%SEED, 'rb') as f:
        ids_train, ids_valid, x_train, x_valid, y_train, y_valid, cov_train, cov_test, depth_train, depth_test = pickle.load(f)
    y_train = y_train.astype(np.uint8)
    y_valid = y_valid.astype(np.uint8)
    if debug:
        x_train, y_train = x_train[:100], y_train[:100]
        x_valid, y_valid = x_valid[:10], y_valid[:10]
    print('Count of trainset: ', x_train.shape[0])
    print('Count of validset: ', x_valid.shape[0])

    # make pytorch.data.Dataset
    train_ds = TgsDataSet(x_train, y_train, transform=True)
    val_ds = TgsDataSet(x_valid, y_valid, transform=False)#False
    
    train_dl = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,#True
        #sampler=StratifiedSampler(),
        num_workers=NUM_WORKERS,
    )
    
    val_dl = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        #sampler=StratifiedSampler(),
        num_workers=NUM_WORKERS,
    )
    
    return train_dl, val_dl

######### Run the training process #########
def run_check_net(train_dl, val_dl):
    set_logger(LOG_PATH)
    logging.info('\n\n')
    #---
    net = UNetResNet34(pretrained=True).cuda(device=device)#debug=False
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, net.parameters()),
                      lr=LearningRate, momentum=0.9, weight_decay=0.0001)
    
    if warm_start:
        logging.info('warm_start: '+last_checkpoint_path)
        net, _ = load_checkpoint(last_checkpoint_path, net)

    net = nn.DataParallel(net, device_ids=[0, 1])

    optimizer.zero_grad()
    for i in range(CYCLES):
        # each cycle is a new model
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS_PER_CYCLE, eta_min=MIN_LR, last_epoch=-1)
        best_val_iou = 0.0
        for j in range(EPOCHS_PER_CYCLE):
            scheduler.step(epoch=j)
            logging.info('lr: %f'%scheduler.get_lr()[0])
            # iterate through trainset
            net.module.set_mode('train')
            train_loss_list1, train_iou_list1 = [], []

            seed = get_seed()
            np.random.seed(seed)
            #ia.imgaug.seed(i//10)
            for input_data, truth in train_dl:
                #set_trace()
                input_data, truth = input_data.to(device=device, dtype=torch.float), truth.to(device=device, dtype=torch.float)
                logit = net(input_data)

                # calculate metrics separately on data with zero_mask_coverage and nonzero_mask_coverage
                is_zero_cov = (truth.sum(dim=2, keepdim=False).sum(dim=2, keepdim=False)==0).reshape(-1)#.type(torch.FloatTensor)
                if (is_zero_cov==1).all():
                    continue
                logit_nonzero_cov = logit[1 - is_zero_cov, ]
                truth_nonzero_cov = truth[1 - is_zero_cov, ]

                _train_loss1 = net.module.criterion(logit_nonzero_cov, truth_nonzero_cov)
                _train_iou1 = net.module.metric(logit_nonzero_cov, truth_nonzero_cov)
                train_loss_list1.append(_train_loss1.detach())
                train_iou_list1.append(_train_iou1.detach())

                _train_loss1.backward()#_train_loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            train_loss1, train_iou1 = np.mean(train_loss_list1), np.mean(train_iou_list1)

            # compute valid loss & iou (for memory efficiency, use batch)
            net.module.set_mode('valid')
            with torch.no_grad():
                val_loss_list1, val_iou_list1 = [], []
                for input_data, truth in val_dl:
                    input_data, truth = input_data.to(device=device, dtype=torch.float), truth.to(device=device, dtype=torch.float)
                    logit = net(input_data)

                    # calculate metrics separately on data with zero_mask_coverage and nonzero_mask_coverage
                    is_zero_cov = (truth.sum(dim=2, keepdim=False).sum(dim=2, keepdim=False)==0).reshape(-1)#.type(torch.FloatTensor)
                    if (is_zero_cov==1).all():
                        continue
                    logit_nonzero_cov = logit[1 - is_zero_cov, ]
                    truth_nonzero_cov = truth[1 - is_zero_cov, ]

                    _val_loss1 = net.module.criterion(logit_nonzero_cov, truth_nonzero_cov)
                    _val_iou1 = net.module.metric(logit_nonzero_cov, truth_nonzero_cov)
                    val_loss_list1.append(_val_loss1.detach())
                    val_iou_list1.append(_val_iou1.detach())

                val_loss1, val_iou1 = np.mean(val_loss_list1), np.mean(val_iou_list1)
            
            logging.info('[CYCLE %d EPOCH %03d][mask coverage not zero] train_loss, train_iou: %0.5f, %0.5f; val_loss, val_iou: %0.5f, %0.5f'%(i, j, train_loss1.item(), train_iou1.item(), val_loss1.item(), val_iou1.item()))

            # try save best epoch?
            if val_iou1 > best_val_iou:
                logging.info('========save best epoch: EPOCH %03d========'%j)
                best_val_iou = val_iou1
                #save checkpoint
                checkpoint_dict = \
                {
                    'cycle': i,
                    'state_dict': net.module.state_dict(),
                    'optim_dict' : optimizer.state_dict(),
                    'metrics': {'train_loss1': train_loss1, 
                                'val_loss1': val_loss1, 
                                'train_iou1': train_iou1, 
                                'val_iou1': val_iou1}
                }
                save_checkpoint_snapshot(checkpoint_dict, checkpoint=checkpoint_path, cycle_id=i)



os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"#"0, 1, 2, 3, 4, 5"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

SEED = 6789#5678#4567#3456#2345#1234
debug = False# if True, load 100 samples
BATCH_SIZE = 16#32
NUM_WORKERS = 20
warm_start, last_checkpoint_path = True, 'checkpoint/1006_v1_seed6789/best.pth.tar'
checkpoint_path = 'checkpoint/1008_v1_seed%d_snapshot'%SEED#seed%d_phase2, seed%d ;;; seed%d-hypercol, seed%d-hypercol-phase2
LOG_PATH = 'logging/1008_v1_seed%d_snapshot.log'%SEED#seed%d.log
torch.cuda.manual_seed_all(SEED)

#NUM_EPOCHS = 500#200#500
#early_stopping_round = 20#500#30
LearningRate = 0.01#phase1: 0.005, phase2: 0.001
EPOCHS_PER_CYCLE = 50#50
MIN_LR = 0.001
CYCLES = 6

train_dl, val_dl = prepare_data()

run_check_net(train_dl, val_dl)
