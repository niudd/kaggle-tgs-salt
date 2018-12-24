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

from model_classify_zero_mask import criterion, metric
from model import UNetResNet34
#from metrics import iou_pytorch
from utils import save_checkpoint, load_checkpoint, set_logger
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
        x_train, y_train = x_train[:500], y_train[:500]
        x_valid, y_valid = x_valid[:100], y_valid[:100]
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
#     for param in net.named_parameters():
#         if param[0][:8] in ['decoder5']:#'decoder5', 'decoder4', 'decoder3', 'decoder2'
#             param[1].requires_grad = False
    #net = SaltLinkNet(num_classes=1, dropout_2d=0.0, pretrained=True, is_deconv=False).cuda(device=device)

    # dummy sgd to see if it can converge ...
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, net.parameters()),
                      lr=LearningRate, momentum=0.9, weight_decay=0.0001)
    #optimizer = optim.Adam(net.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', 
                                                           factor=0.5, patience=8, 
                                                           verbose=False, threshold=0.0001, 
                                                           threshold_mode='rel', cooldown=0, 
                                                           min_lr=0, eps=1e-08)
    if warm_start:
        logging.info('warm_start: '+last_checkpoint_path)
        net, _ = load_checkpoint(last_checkpoint_path, net)
    
    net = nn.DataParallel(net, device_ids=[0, 1])

    diff = 0
    best_val_iou = 0.0
    optimizer.zero_grad()
    
    #seed = get_seed()
    #seed = SEED
    #logging.info('aug seed: '+str(seed))
    #ia.imgaug.seed(seed)
    #np.random.seed(seed)
    
    for i in range(NUM_EPOCHS):
        # iterate through trainset
        net.module.set_mode('train')
        train_loss_list, train_iou_list = [], []
        logit_list, truth_list = [], []
        #for seed in [1]:#[1, SEED]:#augment raw data with a duplicate one (augmented)
        seed = get_seed()
        np.random.seed(seed)
        #ia.imgaug.seed(i//10)
        for input_data, truth in train_dl:
            #set_trace()
            input_data, truth = input_data.to(device=device, dtype=torch.float), truth.to(device=device, dtype=torch.float)
            # for classify zero mask modelling, 1: zero 0: nonzero
            truth = truth.reshape(-1, 256*256).sum(dim=1, keepdim=True)==0
            
            logit = net(input_data)
            logit = logit.reshape(-1, 256*256).sum(dim=1, keepdim=True)==0
            logit = logit.float()

            _train_loss  = criterion(logit, truth)
            #_train_iou  = net.module.metric(logit, truth)
            train_loss_list.append(_train_loss.detach())
            #train_iou_list.append(_train_iou)
            logit_list.append(logit)
            truth_list.append(truth)

            _train_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        train_loss = np.mean(train_loss_list)
        #train_iou = np.mean(train_iou_list)
        train_iou = metric(torch.cat(logit_list, dim=0), torch.cat(truth_list, dim=0))

        # compute valid loss & iou (for memory efficiency, use batch)
        net.module.set_mode('valid')
        with torch.no_grad():
            val_loss_list, val_iou_list = [], []
            logit_list, truth_list = [], []
            for input_data, truth in val_dl:
                input_data, truth = input_data.to(device=device, dtype=torch.float), truth.to(device=device, dtype=torch.float)
                # for classify zero mask modelling, 1: zero 0: nonzero
                truth = truth.reshape(-1, 256*256).sum(dim=1, keepdim=True)==0
                
                logit = net(input_data)
                logit = logit.reshape(-1, 256*256).sum(dim=1, keepdim=True)==0
                logit = logit.float()
                
                _val_loss = criterion(logit, truth)
                #_val_iou = net.module.metric(logit, truth)
                val_loss_list.append(_val_loss)
                #val_iou_list.append(_val_iou)
                logit_list.append(logit)
                truth_list.append(truth)
            val_loss = np.mean(val_loss_list)
            #val_iou = np.mean(val_iou_list)
            val_iou = metric(torch.cat(logit_list, dim=0), torch.cat(truth_list, dim=0))

        # Adjust learning_rate
        scheduler.step(val_iou)
        #
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            is_best = True
            diff = 0
        else:
            is_best = False
            diff += 1
            if diff > early_stopping_round:
                logging.info('Early Stopping: val_iou does not increase %d rounds'%early_stopping_round)
                #print('Early Stopping: val_iou does not increase %d rounds'%early_stopping_round)
                break
        
        #save checkpoint
        checkpoint_dict = \
        {
            'epoch': i,
            'state_dict': net.module.state_dict(),
            'optim_dict' : optimizer.state_dict(),
            #'metrics': {'train_loss': train_loss, 'val_loss': val_loss, 'train_iou': train_iou, 'val_iou': val_iou}
            'metrics': {'train_loss1': train_loss, 
                        'val_loss1': val_loss, 
                        'train_iou1': train_iou, 
                        'val_iou1': val_iou}
        }
        save_checkpoint(checkpoint_dict, is_best=is_best, checkpoint=checkpoint_path)

        #if i%20==0:
        if i>-1:
            #logging.info('[EPOCH %05d][mask coverage zero] train_loss, train_iou: %0.5f, %0.5f; val_loss, val_iou: %0.5f, %0.5f'%(i, train_loss0.item(), train_iou0.item(), val_loss0.item(), val_iou0.item()))
            logging.info('[EPOCH %05d]train_loss, train_iou: %0.5f, %0.5f; val_loss, val_iou: %0.5f, %0.5f'%(i, train_loss.item(), train_iou.item(), val_loss.item(), val_iou.item()))
            #logging.info('[EPOCH %05d] train_loss, train_iou: %0.5f,%0.5f; val_loss, val_iou: %0.5f,%0.5f'%(i, train_loss.item(), train_iou.item(), val_loss.item(), val_iou.item()))


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"#"0, 1, 2, 3, 4, 5"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
#torch.set_num_threads(20)

SEED = 1234#5678#4567#3456#2345#1234
debug = True# if True, load 100 samples
BATCH_SIZE = 16#32
NUM_WORKERS = 20
warm_start, last_checkpoint_path = True, 'checkpoint/1006_v1_seed1234_849/best.pth.tar'
checkpoint_path = 'checkpoint/1010_binary_mask_classifier_v1_seed%d'%SEED
LOG_PATH = 'logging/1010_binary_mask_classifier_v1_seed%d.log'%SEED
torch.cuda.manual_seed_all(SEED)

NUM_EPOCHS = 500#200#500
early_stopping_round = 20#500#30
LearningRate = 0.005#phase1: 0.005, phase2: 0.001

train_dl, val_dl = prepare_data()

run_check_net(train_dl, val_dl)
