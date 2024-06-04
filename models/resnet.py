import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class ResNet(nn.Module):
    def __init__(self, num_classes=7, pretrained=False, input_size=48, is_classifier=True, model_name='resnet50'):
        super(ResNet, self).__init__()
        self.model_name = model_name
        final_size = self.calculate_final_size(input_size)

        if model_name == 'resnet50':
            self.resnet = models.resnet50(pretrained=pretrained)
            self.classifier = nn.Sequential(nn.Linear(2048 * final_size * final_size, num_classes))
        else:
            self.resnet = models.resnet18(pretrained=pretrained)
            self.classifier = nn.Sequential(nn.Linear(512 * final_size * final_size, num_classes))
        
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.is_classifier = is_classifier  
        



    def forward(self, x):
       
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
    
        x = self.resnet.avgpool(x)
        if not self.is_classifier:
            return x

        x = x.view(x.size(0), -1)
        print("X SHAPE: ", x.shape)

        x = self.classifier(x)
        return x
        
        

    def calculate_final_size(self, size):
        size = size // 2  # First maxpool
        size = size // 2  # layer1
        size = size // 2  # layer2
        size = size // 2  # layer3
        size = size // 2  # layer4
        return size