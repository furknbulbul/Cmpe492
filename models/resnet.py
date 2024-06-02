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
        nn.Linear(512 * final_size * final_size, 4096),
        nn.Dropout(dropout),
        nn.ReLU(True),
        nn.Linear(4096, 4096),
        nn.Dropout(dropout),
        nn.ReLU(True),
        nn.Linear(4096, num_classes)
         )



    def forward(self, x):
        x = self.resnet50.conv1(x)
        x = self.resnet50.bn1(x)
        x = self.resnet50.relu(x)
        x = self.resnet50.maxpool(x)
        print('After maxpool:', x.shape)

        x = self.resnet50.layer1(x)
        print('After layer1:', x.shape)

        x = self.resnet50.layer2(x)
        print('After layer2:', x.shape)

        x = self.resnet50.layer3(x)
        print('After layer3:', x.shape)

        x = self.resnet50.layer4(x)
        print('After layer4:', x.shape)

        if not self.is_classifier:
            print('Returning x')
            return x

        x = self.resnet50.avgpool(x)
        print('After avgpool:', x.shape)

        x = torch.flatten(x, 1)  # Flatten the tensor
        print('After flatten:', x.shape)

        x = self.classifier(x)
        return x
        

    def calculate_final_size(self, size):
        size = size // 2  # First maxpool
        size = size // 2  # layer1
        size = size // 2  # layer2
        size = size // 2  # layer3
        size = size // 2  # layer4
        return size