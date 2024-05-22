from torch import nn

# Khaireddin, Yousif, and Zhuofa Chen.
# "Facial Emotion Recognition: State of the Art Performance on FER2013." arXiv preprint arXiv:2105.03588 (2021).

configs = {
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg11': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M']

}


class VGGNet(nn.Module):
    def __init__(self, num_classes=7, config='vgg11', dropout=0.2, isClassifier=True):
        super(VGGNet, self).__init__()
        self.config = configs[config]
        self.features = self.make_layers()
        self.classifier = None
        self.isClassifier = isClassifier
        if isClassifier:
            self.classifier =  nn.Sequential(nn.Linear(512 * 3 * 3, 4096), nn.Dropout(dropout), nn.ReLU(True),
                                        nn.Linear(4096, 4096), nn.Dropout(dropout), nn.ReLU(True),
                                        nn.Linear(4096, num_classes))

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
        # layers += [nn.AvgPool2d(kernel_size=1, stride=1)]

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        if not self.isClassifier:
            return x

        x = x.view(-1, 512 * 3 * 3)
        x = self.classifier(x)
        return x
