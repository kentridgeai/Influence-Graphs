# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 15:32:52 2025

@author: User
"""
import torch
import torch.nn as nn
import torchvision
import torchvision.models as models


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

        print('flatten_size:', flatten_size)

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

        # Switch to eval mode to avoid BatchNorm error
        was_training = self.features.training
        self.features.eval()
        with torch.no_grad():
            dummy_output = self.features(dummy_input)

        # Restore original mode
        if was_training: self.features.train()
    
        # If AdaptiveAvgPool2d was applied, the feature map size will be 1x1
        if dummy_output.dim() == 4 and dummy_output.shape[2:] == (1, 1):
            return dummy_output.size(1)  # Only channels
        else:
            return dummy_output.view(1, -1).size(1)  # Flatten all dims

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
    


def get_pretrained_vgg16(num_classes=1000, fine_tune='NEW_LAYERS'):
    # Load VGG16 pretrained model
    model = models.vgg16_bn(weights=torchvision.models.VGG16_BN_Weights.IMAGENET1K_V1)

    # Replace the classifier (fc layers)
    in_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(in_features, num_classes)

    # Freeze all parameters
    for p in model.parameters(): p.requires_grad = False
    
    # Unfreeze parameters based on fine_tune
    if fine_tune == 'NEW_LAYERS':
        for l in [model.classifier[6]]:
            for p in l.parameters():
                p.requires_grad = True

    elif fine_tune == 'CLASSIFIER':
        for l in [model.classifier]:
            for p in l.parameters():
                p.requires_grad = True

    else:
        for p in m.parameters():
            p.requires_grad = True

    return model



def get_pretrained_resnet50(num_classes=1000, fine_tune='NEW_LAYERS'):
    # Load Resnet50 pretrained model
    model = models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)

    # Replace the fully connected layer
    in_features = model.fc.in_features  # 2048 for ResNet50
    model.fc = nn.Linear(in_features, num_classes)

    # Freeze all parameters
    for p in model.parameters(): p.requires_grad = False
    
    # Unfreeze parameters based on fine_tune
    if fine_tune == 'NEW_LAYERS':
        for l in [model.fc]:
            for p in l.parameters():
                p.requires_grad = True

    elif fine_tune == 'CLASSIFIER':
        for l in [model.fc]:
            for p in l.parameters():
                p.requires_grad = True

    else:
        for p in m.parameters():
            p.requires_grad = True

    return model



def get_model_from_params(model_params):
    fine_tune = model_params['fine_tune']
    
    if model_params['name'] == 'pretrained_VGG16':
        return get_pretrained_vgg16(num_classes=model_params['num_classes'], fine_tune=fine_tune)
        
    elif model_params['name'] == 'pretrained_resnet50':
        return get_pretrained_resnet50(num_classes=model_params['num_classes'], fine_tune=fine_tune)
        
    else:
        return model_params['type'](
            model_params['name'],
            in_channels = model_params['in_channels'],
            num_classes = model_params['num_classes'],
            img_size    = model_params['img_size'],
            batchnorm   = model_params['batchnorm']
        )
    

    
def test():
    net = CNN('GA_VGG11')
    x = torch.randn(2,3,32,32)
    y = net(x)
    print(y.size())