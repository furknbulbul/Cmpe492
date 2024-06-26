from torch import nn
from torch.nn import functional as F
import torch    
from .word_embedding import WordEmbedding
from .projection_mlp import ProjectionMLP
from ..VGGNet import VGGNet
from ..resnet import ResNet
import math

class Multimodal(nn.Module):
    def __init__(self, hidden_dim, output_dim, image_embedding = "vgg11", image_embedding_dim = 512 * 1 * 1, text_embedding_dim =  100, num_classes = 7, use_classifier = False, freeze_cnn = False, dropout = 0.2):
        super(Multimodal, self).__init__()
        
        assert not (use_classifier and not freeze_cnn), "freeze_cnn can only be True when use_classifier is True"
        self.use_classifier = use_classifier
        self.freeze_cnn = freeze_cnn
        self.image_embedding_dim = image_embedding_dim
        print("IMAGE EMBEDDING: ", image_embedding)
        if image_embedding == "vgg11":
            self.image_embedding = VGGNet(num_classes=num_classes, config='vgg11', dropout=dropout, is_classifier=False)

        elif image_embedding == "vgg16":
            self.image_embedding = VGGNet(num_classes=num_classes, config='vgg16', dropout=dropout, is_classifier=False)
        
        elif image_embedding == "resnet50":
            self.image_embedding = ResNet(num_classes=num_classes, pretrained=False, is_classifier=False, model_name = 'resnet50')
        elif image_embedding == "resnet18":
            self.image_embedding = ResNet(num_classes=num_classes, pretrained=False, is_classifier=False, model_name = 'resnet18')



        if freeze_cnn:
            for param in self.image_embedding.parameters():
                param.requires_grad = False
        self.text_embedding = WordEmbedding(embedding_dim=text_embedding_dim)
        print("TEXT EMBEDDING: ", self.text_embedding.embedding_dim)

        print("IMAGE EMBEDDING DIM: ", self.image_embedding_dim)
        self.image_projector = ProjectionMLP(image_embedding_dim, hidden_dim, output_dim)
        self.text_projector = ProjectionMLP(text_embedding_dim * 7, hidden_dim, output_dim, is_text=True)
        
            
        self.classifier =  self.image_embedding.classifier
 

    def forward(self, image, texts):

        if not self.use_classifier:
            image_projected = self.image_projector(self.image_embedding(image))
            text_projected = self.text_projector(self.text_embedding(texts))
            return image_projected, text_projected

        
        image_embedding = self.image_embedding(image)
        flattened_image = image_embedding.view(-1, self.image_embedding_dim * 1 * 1)
            
        logits = self.classifier(flattened_image)
        return logits # only return the classification output if use_classifier is True
    
    def reset_classifier(self):
        for layer in self.classifier:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, a=math.sqrt(5))
                if layer.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(layer.weight)
                    bound = 1 / math.sqrt(fan_in)
                    nn.init.uniform_(layer.bias, -bound, bound)