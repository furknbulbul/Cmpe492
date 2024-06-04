import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class ResNet50(nn.Module):
    def __init__(self, num_classes=7, pretrained=False, input_size=48, is_classifier=True,dropout=0.2):
        super(ResNet50, self).__init__()

        self.resnet50 = models.resnet50(pretrained=pretrained)
        
        self.resnet50.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        final_size = self.calculate_final_size(input_size)

        self.is_classifier = is_classifier
        self.classifier = nn.Sequential(
        nn.Linear(2048 * final_size * final_size, num_classes),
        )



    def forward(self, x):
        x = self.resnet50.conv1(x)
        x = self.resnet50.bn1(x)
        x = self.resnet50.relu(x)
        x = self.resnet50.maxpool(x)

        x = self.resnet50.layer1(x)
        x = self.resnet50.layer2(x)
        x = self.resnet50.layer3(x)
        

        x = self.resnet50.layer4(x)
    
        x = self.resnet50.avgpool(x)
        if not self.is_classifier:
            return x

        
        x = x.view(x.size(0), -1)

        x = self.classifier(x)
        return x
        

    def calculate_final_size(self, size):
        size = size // 2  # First maxpool
        size = size // 2  # layer1
        size = size // 2  # layer2
        size = size // 2  # layer3
        size = size // 2  # layer4
        return size