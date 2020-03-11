import pretrainedmodels
import torch.nn as nn
from torch.nn import functional as F
from torchvision import models
from ghost_net import ghost_net
from efficientnet_pytorch import EfficientNet

class ResNet34(nn.Module):
    def __init__(self,pretrained,output_channel):
        super(ResNet34, self).__init__()
        if pretrained == True:
            self.model = pretrainedmodels.__dict__['resnet34'](pretrained = 'imagenet')
        else:
            self.model = pretrainedmodels.__dict__['resnet34'](pretrained = None)
        
        self.l0 = nn.Linear(512, output_channel)
       
    
    def forward(self, x):
        bs, _,_,_ = x.shape
        x = self.model.features(x)
        x = F.adaptive_avg_pool2d(x, 1).reshape(bs, -1)
        out = self.l0(x)
        

        return out

class ResNet50(nn.Module):
    def __init__(self,pretrained, output_channel):
        super(ResNet50, self).__init__()
        if pretrained == True:
            self.model = pretrainedmodels.__dict__['resnet50'](pretrained = 'imagenet')
        else:
            self.model = pretrainedmodels.__dict__['resnet50'](pretrained = None)
        
        self.l0 = nn.Linear(512, output_channel)
      
        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(2048, 512)
    
    def forward(self, x):
        bs, _,_,_ = x.shape
        x = self.model.features(x)
        x = F.adaptive_avg_pool2d(x, 1).reshape(bs, -1)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.dropout(x)
        out = self.l0(x)

        return out

class ResNet101(nn.Module):
    def __init__(self,pretrained, output_channel):
        super(ResNet101, self).__init__()
        if pretrained == True:
            self.model = pretrainedmodels.__dict__['resnet101'](pretrained = 'imagenet')
        else:
            self.model = pretrainedmodels.__dict__['resnet101'](pretrained = None)
        
        self.l0 = nn.Linear(512, output_channel)
      
        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(2048, 512)
    
    def forward(self, x):
        bs, _,_,_ = x.shape
        x = self.model.features(x)
        x = F.adaptive_avg_pool2d(x, 1).reshape(bs, -1)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.dropout(x)
        out = self.l0(x)

        return out


    
class InceptionV3(nn.Module):
    def __init__(self,pretrained, output_channel):
        super(InceptionV3, self).__init__()
        if pretrained == True:
            self.model = pretrainedmodels.__dict__['inceptionv3'](pretrained = 'imagenet')
        else:
            self.model = pretrainedmodels.__dict__['inceptionv3'](pretrained = None)
        
        self.l0 = nn.Linear(512, output_channel)
      
        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(2048, 512)
    
    def forward(self, x):
        bs, _,_,_ = x.shape
        x = self.model.features(x)
        x = F.adaptive_avg_pool2d(x, 1).reshape(bs, -1)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.dropout(x)
        out = self.l0(x)

        return out


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


class Squeezenet1_1(nn.Module):
    def __init__(self,pretrained, output_channel):
        super(Squeezenet1_1, self).__init__()
        if pretrained == True:
            self.model = pretrainedmodels.__dict__['squeezenet1_1'](pretrained = 'imagenet')
        else:
            self.model = pretrainedmodels.__dict__['squeezenet1_1'](pretrained = None)
        
        self.l0 = nn.Linear(512, output_channel)
      
        self.dropout = nn.Dropout(p=0.5)
        #self.fc1 = nn.Linear(2048, 512)
    
    def forward(self, x):
        bs, _,_,_ = x.shape
        x = self.model.features(x)
        x = F.adaptive_avg_pool2d(x, 1).reshape(bs, -1)
        x = self.dropout(x)
        # x = self.fc1(x)
        # x = self.dropout(x)
        out = self.l0(x)

        return out


class Densenet201(nn.Module):
    def __init__(self,pretrained, output_channel):
        super(Densenet201, self).__init__()
        if pretrained == True:
            self.model = pretrainedmodels.__dict__['densenet201'](pretrained = 'imagenet')
        else:
            self.model = pretrainedmodels.__dict__['densenet201'](pretrained = None)
        
        self.l0 = nn.Linear(512, output_channel)
      
        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(1920, 512)
    
    def forward(self, x):
        bs, _,_,_ = x.shape
        x = self.model.features(x)
        x = F.adaptive_avg_pool2d(x, 1).reshape(bs, -1)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.dropout(x)
        out = self.l0(x)

        return out


class ResNeXt_50(nn.Module):
    def __init__(self,pretrained, output_channel):
        super(ResNeXt_50, self).__init__()
        if pretrained == True:
            self.model = models.resnext50_32x4d(pretrained=True, progress=True)
        else:
            self.model = models.resnext50_32x4d(pretrained=False, progress=True)
        
        self.l0 = nn.Linear(1000, output_channel)
      
        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(2048, 1000)
    
    def forward(self, x):
        bs, _,_,_ = x.shape
        x = self.model(x)
        #x = F.adaptive_avg_pool2d(x, 1).reshape(bs, -1)
        x = self.dropout(x)
        #x = self.fc1(x)
        #x = self.dropout(x)
        out = self.l0(x)

        return out

class ResNeXt_101(nn.Module):
    def __init__(self,pretrained, output_channel):
        super(ResNeXt_101, self).__init__()
        if pretrained == True:
            self.model = models.resnext101_32x8d(pretrained=True, progress=True)
        else:
            self.model = models.resnext101_32x8d(pretrained=False, progress=True)
        
        self.l0 = nn.Linear(1000, output_channel)
      
        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(2048, 1000)
    
    def forward(self, x):
        bs, _,_,_ = x.shape
        x = self.model(x)
        #x = F.adaptive_avg_pool2d(x, 1).reshape(bs, -1)
        x = self.dropout(x)
        #x = self.fc1(x)
        #x = self.dropout(x)
        out = self.l0(x)

        return out


class GhostNet(nn.Module):
    def __init__(self,pretrained,output_channel):
        super(GhostNet, self).__init__()
        if pretrained == True:
            self.model = ghost_net(output_channel)
        else:
            self.model = ghost_net(output_channel)
        
        self.l0 = nn.Linear(512, output_channel)
       
    
    def forward(self, x):
        #bs, _,_,_ = x.shape
        
        out = self.model(x)
        

        return out


class EfficientNetWrapper(nn.Module):
    def __init__(self, pretrained, output_channel):
        super(EfficientNetWrapper, self).__init__()

        if pretrained:
            self.model = EfficientNet.from_pretrained('efficientnet-b7')
        else:
            self.model = EfficientNet.from_name('efficientnet-b7')
        
        # Appdend output layers based on our date
        self.fc = nn.Linear(in_features=1000, out_features=output_channel)
        self.dropout = nn.Dropout(p=0.4)
        
        
    def forward(self, x):
        x = self.model(x)
        x = self.dropout(x)

        out = self.fc(x)
        return out