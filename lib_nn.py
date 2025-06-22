# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 12:27:05 2025

@author: User
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 15:32:52 2025

@author: User
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


cfg_feat = {
    'NN-1': [512],
    'NN-2': [512,512],
    
}


class Res_NN(nn.Module):
    def __init__(self, input_channels, block, num_blocks=[1,1], num_classes=10,normalize=False):
        super(Res_NN, self).__init__()
        self.in_planes = 512 
        self.conv1 = nn.Conv2d(input_channels, 512, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(512)
        self.layer1 = self._make_layer(block, 512, num_blocks[0])
        self.layer2 = self._make_layer(block, 512, num_blocks[1])
        self.linear = nn.Linear(512, num_classes)
        self.normalize = normalize 
        
        # self.drop = nn.Dropout(0.2)
        
        
    def _make_layer(self, block, planes, num_blocks, stride=1):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes 
        return nn.Sequential(*layers)

    def forward(self, x):

        out = F.relu(self.bn1(self.conv1(x)))

        out = self.layer1(out)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out,0
    
    
class FC_NN(nn.Module):
    def __init__(self, nn_name, in_channels = 1, num_classes = 10):
        super(FC_NN, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.network = self._make_layers(cfg_feat[nn_name])
        

    def forward(self, x):
        out = self.network(x)
        return out

    def _make_layers(self, in_channels, cfg):
        layers = []

        for i in range(len(cfg)):
            layers += [nn.Linear(in_channels, cfg[i]),
                       nn.BatchNorm1d(cfg[i]),
                       nn.ReLU(inplace=True)]
            in_channels = cfg[i]
        layers += [nn.Linear(in_channels, self.num_classes)]
        return nn.Sequential(*layers)
    


def test():
    net = FC_NN('NN-1')