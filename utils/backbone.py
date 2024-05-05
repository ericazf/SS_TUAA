import math 
import torch 
import torch.nn as nn 
from torchvision import models 
 
#这里nn.AdaptiveAvgPool2d用与不用有什么区别
class AlexNet(nn.Module): 
    def __init__(self, bit):
        super(AlexNet, self).__init__()
        self.model = models.alexnet(pretrained = True)
        self.model.classifier[6] = nn.Linear(4096, bit)

        self.tanh = nn.Tanh()
        self.mean = torch.tensor([[[0.485]], [[0.456]], [[0.406]]]).cuda()
        self.std = torch.tensor([[[0.229]], [[0.224]], [[0.225]]]).cuda()

    def forward(self, x, alpha = 1):
        x = (x - self.mean)/self.std 
        H = self.model(x)
        output = self.tanh(alpha * H)
        return H, output 

vgg_dict = {"VGG11": models.vgg11, "VGG13": models.vgg13, "VGG16": models.vgg16, "VGG19": models.vgg19,
            "VGG11BN": models.vgg11_bn, "VGG13BN": models.vgg13_bn, "VGG16BN": models.vgg16_bn, "VGG19BN": models.vgg19_bn}
class VGG(nn.Module):
    def __init__(self, model_name, bit):
        super(VGG, self).__init__()
        self.model = vgg_dict[model_name](pretrained = True)
        self.model.classifier[6] = nn.Linear(4096, bit)
        
        self.tanh = nn.Tanh()
        self.mean = torch.tensor([[[0.485]], [[0.456]], [[0.406]]]).cuda()
        self.std = torch.tensor([[[0.229]], [[0.224]], [[0.225]]]).cuda()

    def forward(self, x, alpha = 1):
        x = (x - self.mean)/self.std 
        H = self.model(x)
        output = self.tanh(alpha * H)
        return H, output


resnet_dict = {"ResNet18": models.resnet18, "ResNet34": models.resnet34,
               "ResNet50": models.resnet50, "ResNet101": models.resnet101, "ResNet152": models.resnet152}
class ResNet(nn.Module):
    def __init__(self, model_name, bit):
        super(ResNet, self).__init__()
        self.model = resnet_dict[model_name](pretrained = True)
        self.in_feature = self.model.fc.in_features
        self.model.fc = nn.Linear(self.in_feature, bit)

        self.tanh = nn.Tanh()
        self.mean = torch.tensor([[[0.485]], [[0.456]], [[0.406]]]).cuda()
        self.std = torch.tensor([[[0.229]], [[0.224]], [[0.225]]]).cuda()
    
    def forward(self, x, alpha = 1):
        x = (x - self.mean)/self.std 
        H = self.model(x)
        output = self.tanh(alpha * H)
        return H, output 
  
densenet_dict = {"DenseNet161":models.densenet161}
class DenseNet(nn.Module):
    def __init__(self, model_name, bit):
        super(DenseNet, self).__init__()
        self.model = densenet_dict[model_name](pretrained = True)
        self.in_feature = self.model.classifier.in_features
        self.model.classifier = nn.Linear(self.in_feature, bit)
        self.tanh = nn.Tanh()
        self.mean = torch.tensor([[[0.485]], [[0.456]], [[0.406]]]).cuda()
        self.std = torch.tensor([[[0.229]], [[0.224]], [[0.225]]]).cuda()
    
    def forward(self, x, alpha = 1):
        x = (x - self.mean)/self.std 
        H = self.model(x)
        output = self.tanh(alpha * H)
        return H, output 
 
inceptionv3 = {"Incv3": models.inception_v3}
class InceptionV3(nn.Module):
    def __init__(self, model_name, bit):
        super(InceptionV3, self).__init__()
        self.model = inceptionv3[model_name](pretrained = True, aux_logits = False)
        self.in_feature = self.model.fc.in_features 
        self.model.fc = nn.Linear(self.in_feature, bit)
        self.tanh = nn.Tanh()
        self.mean = torch.tensor([[[0.485]], [[0.456]], [[0.406]]]).cuda()
        self.std = torch.tensor([[[0.229]], [[0.224]], [[0.225]]]).cuda()
     
    def forward(self, x, alpha = 1):
        x = (x - self.mean)/self.std 
        H = self.model(x)
        output = self.tanh(alpha * H)
        return H, output 

   