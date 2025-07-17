# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 15:32:52 2025

@author: User
"""
import torch
import torch.nn as nn


cfg_feat = {
    'ShallowMNIST': [64,'M', 128, 'M', 128, 'MP1', 128, 'M4', 'GA'],
    'ShallowCIFAR10': [64,'M', 128, 'M', 128, 'M', 128, 'M4', 'GA'],
    'ShallowfatterMNIST': [64,'M', 128, 'M', 256, 'MP1', 512, 'M4', 'GA'],
    'ShallowfatterCIFAR10': [64,'M', 128, 'M', 256, 'M', 512, 'M4', 'GA'],

    
    'GA_VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M', 'GA'],
    'GA_VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M', 'GA'],
    'GA_VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M', 'GA'],
    'GA_VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M', 'GA'],
}

cfg_FC = {
    'ShallowMNIST': [128],
    'ShallowfatterMNIST': [512],
    
    'ShallowCIFAR10': [128],
    'ShallowfatterCIFAR10': [512],
    
    'GA_VGG11': [512],
    'GA_VGG13': [512],
    'GA_VGG16': [512],
    'GA_VGG19': [512],
}


class CNN(nn.Module):
    def __init__(self, vgg_name, in_channels=1, num_classes=10, img_size=32, batchnorm=True):
        super(CNN, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.batchnorm = batchnorm

        # Build convolutional feature extractor
        self.features, out_channels = self._make_layers(cfg_feat[vgg_name])

        # Dynamically compute flattened feature size
        flatten_size = self._get_flatten_size(img_size, out_channels)

        # Build classifier
        self.classifier = self._make_FC(flatten_size, cfg_FC[vgg_name])
        
    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _get_flatten_size(self, img_size, out_channels):
        """
        Pass dummy input to compute size of flattened feature map.
        """
        dummy_input = torch.zeros(1, self.in_channels, img_size, img_size)
        dummy_output = self.features(dummy_input)
        return dummy_output.view(1, -1).size(1)

    def _make_layers(self, cfg):
        layers = []
        in_channels = self.in_channels
        for x in cfg:
            if type(x) != int:
                if x[0] == 'M':
                    if len(x)==1:
                        layers += [nn.MaxPool2d(kernel_size=2)]
                    else:
                        kernel_size = 2
                        padding = 0 
                        if x[1].isnumeric():
                            kernel_size = int(x[1])
                            if len(x)>2:
                                padding = int(x[3])
                        else:
                            padding = int(x[2])
                        layers += [nn.MaxPool2d(kernel_size=kernel_size, padding=padding)]
                        
            elif x == 'GA':
                layers += [nn.AdaptiveAvgPool2d((1, 1))]
                
            else:
                if self.batchnorm: 
                    layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                               nn.BatchNorm2d(x),
                               nn.ReLU(inplace=True)]
                else:
                    layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                               nn.ReLU(inplace=True)]
                in_channels = x
                
        return nn.Sequential(*layers), in_channels
    
    def _make_FC(self, in_channels, cfg):
        layers = []
        for i in range(len(cfg)):
            if self.batchnorm:
                layers += [nn.Linear(in_channels, cfg[i]),
                           nn.BatchNorm1d(cfg[i]),
                           nn.ReLU(inplace=True)]
            else:
                layers += [nn.Linear(in_channels, cfg[i]),
                           nn.ReLU(inplace=True)]
                
            in_channels = cfg[i]
        layers += [nn.Linear(in_channels, self.num_classes)]
        return nn.Sequential(*layers)
    


def test():
    net = CNN('GA_VGG11')
    x = torch.randn(2,3,32,32)
    y = net(x)
    print(y.size())