import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class ModifiedVit(nn.Module):
    def __init__(self, num_classes = 7, is_classifier = True):
        super(ModifiedVit, self).__init__()
        self.vit = models.vit_b_16(num_classes=num_classes)
        self.is_classifier = is_classifier
    
        # Modify the first layer to accept 1 input channel
        self.vit.conv_proj = nn.Conv2d(1, out_channels=self.vit.hidden_dim, kernel_size = self.vit.patch_size, stride=self.vit.patch_size)

        # If the model is a classifier, replace the last layer with a new one with the correct number of outputs
        self.features = nn.Sequential(*list(self.vit.children())[:-1])[1]
        self.classifier = nn.Linear(self.vit.hidden_dim, num_classes)

    def forward(self, x):
        if not self.is_classifier:
            x = self.features(x)[:,0]
            return x
        x = self.vit(x)
        return x