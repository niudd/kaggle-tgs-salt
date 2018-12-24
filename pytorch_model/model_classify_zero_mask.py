import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from IPython.core.debugger import set_trace
import matplotlib.pyplot as plt

import lovasz_losses as L
#from metrics import iou_pytorch
from sklearn.metrics import roc_auc_score, confusion_matrix


class ConvBn2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3,3), stride=(1,1), padding=(1,1)):
        super(ConvBn2d, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        #self.dropout = nn.Dropout2d(p=0.1, inplace=False)
        self.bn = nn.BatchNorm2d(out_channels)
        #self.bn = SynchronizedBatchNorm2d(out_channels)


    def forward(self, z):
        x = self.conv(z)
        #x = self.dropout(x)
        x = self.bn(x)
        return x

class Decoder(nn.Module):
    def __init__(self, in_channels, channels, out_channels):
        super(Decoder, self).__init__()
        self.conv1 =  ConvBn2d(in_channels,  channels, kernel_size=3, padding=1)
        self.conv2 =  ConvBn2d(channels, out_channels, kernel_size=3, padding=1)
        self.spa_cha_gate = SCSE(out_channels)

    def forward(self, x, e=None):
        x = F.upsample(x, scale_factor=2, mode='bilinear', align_corners=True)#False
        if e is not None:
            x = torch.cat([x, e], 1)
        x = F.relu(self.conv1(x),inplace=True)
        x = F.relu(self.conv2(x),inplace=True)
        x = self.spa_cha_gate(x)
        return x

class SCSE(nn.Module):
    def __init__(self, in_ch):
        super(SCSE, self).__init__()
        self.spatial_gate = SpatialGate2d(in_ch, 16)#16
        self.channel_gate = ChannelGate2d(in_ch)
    
    def forward(self, x):
        g1 = self.spatial_gate(x)
        g2 = self.channel_gate(x)
        x = g1 + g2 #x = g1*x + g2*x
        return x

class SpatialGate2d(nn.Module):
    def __init__(self, in_ch, r=16):
        super(SpatialGate2d, self).__init__()

        self.linear_1 = nn.Linear(in_ch, in_ch//r)
        self.linear_2 = nn.Linear(in_ch//r, in_ch)

    def forward(self, x):
        input_x = x

        x = x.view(*(x.shape[:-2]),-1).mean(-1)
        x = F.relu(self.linear_1(x), inplace=True)
        x = self.linear_2(x)
        x = x.unsqueeze(-1).unsqueeze(-1)
        x = F.sigmoid(x)

        x = input_x * x

        return x

class ChannelGate2d(nn.Module):
    def __init__(self, in_ch):
        super(ChannelGate2d, self).__init__()

        self.conv = nn.Conv2d(in_ch, 1, kernel_size=1, stride=1)

    def forward(self, x):
        input_x = x

        x = self.conv(x)
        x = F.sigmoid(x)

        x = input_x * x

        return x

class ZeroMaskClassifier(nn.Module):
    # PyTorch U-Net model using ResNet(34, 50 , 101 or 152) encoder.
    def load_pretrain(self, pretrain_file):
        self.encoder.load_state_dict(torch.load(pretrain_file, map_location=lambda storage, loc: storage))

    def __init__(self, pretrained=True, debug=False):
        super().__init__()
        self.resnet = torchvision.models.resnet34(pretrained=pretrained)
        self.debug = debug

        self.conv1 = nn.Sequential(
            self.resnet.conv1,
            self.resnet.bn1,
            self.resnet.relu,
            #self.resnet.maxpool,
        )# 64
        self.encoder2 = nn.Sequential(self.resnet.layer1, SCSE(64))
        self.encoder3 = nn.Sequential(self.resnet.layer2, SCSE(128))
        self.encoder4 = nn.Sequential(self.resnet.layer3, SCSE(256))
        self.encoder5 = nn.Sequential(self.resnet.layer4, SCSE(512))

        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(51200, 1)

    def forward(self, x):
        #batch_size,C,H,W = x.shape

        mean=[0.485, 0.456, 0.406]
        std =[0.229, 0.224, 0.225]
        x = torch.cat([
            (x-mean[0])/std[0],
            (x-mean[1])/std[1],
            (x-mean[2])/std[2],
        ],1)
        #x = add_depth_channels(x)
        
        if self.debug:
            print('input: ', x.size())

        x = self.conv1(x)
        if self.debug:
            print('e1',x.size())
        e2 = self.encoder2(x)
        if self.debug:
            print('e2',e2.size())
        e3 = self.encoder3(e2)
        if self.debug:
            print('e3',e3.size())
        e4 = self.encoder4(e3)
        if self.debug:
            print('e4',e4.size())
        e5 = self.encoder5(e4)
        if self.debug:
            print('e5',e5.size())

        f = self.avgpool(e5)
        if self.debug:
            print('avgpool: ',f.size())
        f = F.dropout(f, p=0.5)
        
        f = f.view(f.size(0), -1)
        logit = self.fc(f)
        if self.debug:
            print('fc: ', logit.size())
        return logit

        ##-----------------------------------------------------------------
    def set_mode(self, mode):
        self.mode = mode
        if mode in ['eval', 'valid', 'test']:
            self.eval()
        elif mode in ['train']:
            self.train()
        else:
            raise NotImplementedError

def criterion(logit, truth):
    """Define the (customized) loss function here."""
    loss = L.binary_xloss(logit, truth, ignore=255)
    return loss

def metric(logit, truth):
    """Define metrics for evaluation especially for early stoppping."""
    auc = roc_auc_score(truth.detach(), logit.detach())
    #tn, fp, fn, tp = confusion_matrix(truth.detach(), logit.detach()).ravel()
    return auc#, [tn, fp, fn, tp]

def predict_proba(net, test_dl, device):
    y_pred = None
    net.set_mode('test')
    with torch.no_grad():
        for i, (input_data, truth) in enumerate(test_dl):
            #if i > 10:
            #    break
            input_data, truth = input_data.to(device=device, dtype=torch.float), truth.to(device=device, dtype=torch.float)
            logit = net(input_data).cpu().numpy()
            if y_pred is None:
                y_pred = logit
            else:
                y_pred = np.concatenate([y_pred, logit], axis=0)
    return y_pred


def add_depth_channels(image_tensor):
    _, _, h, w = image_tensor.size()
    x_depth_channel = torch.ones(image_tensor.size(), dtype=torch.float64, device=image_tensor.device)
    for row, const in enumerate(np.linspace(0, 1, h)):
        x_depth_channel[:, 0, row, :] = const
    x_depth_channel = x_depth_channel.float()
    x_depth_channel_mul = image_tensor * x_depth_channel
    image_tensor = torch.cat([image_tensor, x_depth_channel, x_depth_channel_mul], 1)
    return image_tensor

