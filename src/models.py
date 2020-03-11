import pretrainedmodels
import torch.nn as nn
from torch.nn import functional as F
from ghost_net import ghost_net
from efficientnet_pytorch import EfficientNet

class ResNet34(nn.Module):
    def __init__(self,pretrained):
        super(ResNet34, self).__init__()
        if pretrained == True:
            self.model = pretrainedmodels.__dict__['resnet34'](pretrained = 'imagenet')
        else:
            self.model = pretrainedmodels.__dict__['resnet34'](pretrained = None)
        
        self.l0 = nn.Linear(512, 168)
        self.l1 = nn.Linear(512, 11)
        self.l2 = nn.Linear(512, 7)
    
    def forward(self, x):
        bs, _,_,_ = x.shape
        x = self.model.features(x)
        x = F.adaptive_avg_pool2d(x, 1).reshape(bs, -1)
        l0 = self.l0(x)
        l1 = self.l1(x)
        l2 = self.l2(x)

        return l0,l1, l2



class ResNet101(nn.Module):
    def __init__(self,pretrained):
        super(ResNet101, self).__init__()
        if pretrained == True:
            self.model = pretrainedmodels.__dict__['resnet101'](pretrained = 'imagenet')
        else:
            self.model = pretrainedmodels.__dict__['resnet101'](pretrained = None)
        
        self.l0 = nn.Linear(2048, 168)
        self.l1 = nn.Linear(2048, 11)
        self.l2 = nn.Linear(2048, 7)
    
    def forward(self, x):
        bs, _,_,_ = x.shape
        x = self.model.features(x)
        x = F.adaptive_avg_pool2d(x, 1).reshape(bs, -1)
        l0 = self.l0(x)
        l1 = self.l1(x)
        l2 = self.l2(x)

        return l0,l1, l2

    
class InceptionV3(nn.Module):
    def __init__(self,pretrained):
        super(InceptionV3, self).__init__()
        if pretrained == True:
            self.model = pretrainedmodels.__dict__['inceptionv3'](pretrained = 'imagenet')
        else:
            self.model = pretrainedmodels.__dict__['inceptionv3'](pretrained = None)
        
        self.l0 = nn.Linear(2048, 168)
        self.l1 = nn.Linear(2048, 11)
        self.l2 = nn.Linear(2048, 7)
    
    def forward(self, x):
        bs, _,_,_ = x.shape
        x = self.model.features(x)
        x = F.adaptive_avg_pool2d(x, 1).reshape(bs, -1)
        l0 = self.l0(x)
        l1 = self.l1(x)
        l2 = self.l2(x)

        return l0,l1, l2


class ResNet152(nn.Module):
    def __init__(self,pretrained):
        super(ResNet152, self).__init__()
        if pretrained == True:
            self.model = pretrainedmodels.__dict__['resnet152'](pretrained = 'imagenet')
        else:
            self.model = pretrainedmodels.__dict__['resnet152'](pretrained = None)
        
        self.l0 = nn.Linear(2048, 168)
        self.l1 = nn.Linear(2048, 11)
        self.l2 = nn.Linear(2048, 7)
    
    def forward(self, x):
        bs, _,_,_ = x.shape
        x = self.model.features(x)
        x = F.adaptive_avg_pool2d(x, 1).reshape(bs, -1)
        l0 = self.l0(x)
        l1 = self.l1(x)
        l2 = self.l2(x)

        return l0,l1, l2


class ResNet50(nn.Module):
    def __init__(self,pretrained):
        super(ResNet50, self).__init__()
        if pretrained == True:
            self.model = pretrainedmodels.__dict__['resnet50'](pretrained = 'imagenet')
        else:
            self.model = pretrainedmodels.__dict__['resnet50'](pretrained = None)
        
        self.l0 = nn.Linear(512, 168)
        self.l1 = nn.Linear(512, 11)
        self.l2 = nn.Linear(512, 7)
        self.dropout = nn.Dropout(p=0.2)
        self.fc1 = nn.Linear(2048, 512)
    
    def forward(self, x):
        bs, _,_,_ = x.shape
        x = self.model.features(x)
        x = F.adaptive_avg_pool2d(x, 1).reshape(bs, -1)
        x = self.dropout(x)
        x = self.fc1(x)
        l0 = self.l0(x)
        l1 = self.l1(x)
        l2 = self.l2(x)

        return l0,l1, l2



class GhostNet(nn.Module):
    def __init__(self,pretrained):
        super(GhostNet, self).__init__()
        if pretrained == True:
            self.model = ghost_net()
        else:
            self.model = ghost_net()

        self.dropout = nn.Dropout(p=0.4)
        
        self.l0 = nn.Linear(1000, 168)
        self.l1 = nn.Linear(1000, 11)
        self.l2 = nn.Linear(1000, 7)
       
    
    def forward(self, x):
        #bs, _,_,_ = x.shape
        
        x = self.model(x)
        x = self.dropout(x)
        l0 = self.l0(x)
        l1 = self.l1(x)
        l2 = self.l2(x)

        return l0,l1,l2


class EfficientNetWrapper(nn.Module):
    def __init__(self, pretrained):
        super(EfficientNetWrapper, self).__init__()

        if pretrained:
            self.model = EfficientNet.from_pretrained('efficientnet-b4')
        else:
            self.model = EfficientNet.from_name('efficientnet-b4')
        
        # Appdend output layers based on our date
       
        self.dropout = nn.Dropout(p=0.4)

        self.l0 = nn.Linear(1000, 168)
        self.l1 = nn.Linear(1000, 11)
        self.l2 = nn.Linear(1000, 7)
        
        
    def forward(self, x):
        x = self.model(x)
        x = self.dropout(x)

        l0 = self.l0(x)
        l1 = self.l1(x)
        l2 = self.l2(x)

        return l0,l1,l2