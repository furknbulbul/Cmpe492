from torch import nn
import math
# Khaireddin, Yousif, and Zhuofa Chen.
# "Facial Emotion Recognition: State of the Art Performance on FER2013." arXiv preprint arXiv:2105.03588 (2021).

configs = {
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 'M']

}


class VGGNet(nn.Module):
    def __init__(self, num_classes=7, config='vgg11', dropout=0.2, is_classifier=True, input_size=48):
        super(VGGNet, self).__init__()
        self.config = configs[config]
        self.features = self.make_layers()
        self.is_classifier = is_classifier

        
        final_size = self.calculate_final_size(input_size)

        self.classifier = nn.Sequential(
            nn.Linear(512 * final_size * final_size, 4096),
            nn.Dropout(dropout),
            nn.ReLU(True),
            nn.Linear(4096, 4096),
            nn.Dropout(dropout),
            nn.ReLU(True),
            nn.Linear(4096, num_classes)
        )

    def make_layers(self):
        layers = []
        in_channel = 1
        for x in self.config:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channel, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channel = x
        return nn.Sequential(*layers)

    def calculate_final_size(self, size):
        for x in self.config:
            if x == 'M':
                size = size // 2
            else:
            
                pass
        return size

    def forward(self, x):
        x = self.features(x)
        if not self.is_classifier:
            return x

        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.classifier(x)
        return x


class SiamaseNetVGG(nn.Module):
    def __init__(self, vgg_net, use_classifier=False, freeze_cnn=True):
        super(SiamaseNetVGG, self).__init__()
        self.embedding_net = vgg_net.features
        self.classifier_net = vgg_net.classifier
        self.use_classifier = use_classifier
        self.freeze_cnn = freeze_cnn
    
    def forward(self, x1, x2) :
        if not self.use_classifier:
            output1 = self.embedding_net(x1)
            output2 = self.embedding_net(x2)
            output1 = output1.view(-1, 512 * 1 * 1)
            output2 = output2.view(-1, 512 * 1 * 1)
            return output1, output2
        else:
            if self.freeze_cnn:
                for param in self.embedding_net.parameters():
                    param.requires_grad = False
            
            features = self.embedding_net(x1)
            features = features.view(-1, 512 * 1 * 1)
            logits = self.classifier_net(features)
            return logits
        
    
    def reset_classifier(self):
        for layer in self.classifier_net:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, a=math.sqrt(5))
                if layer.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(layer.weight)
                    bound = 1 / math.sqrt(fan_in)
                    nn.init.uniform_(layer.bias, -bound, bound)


    
        
