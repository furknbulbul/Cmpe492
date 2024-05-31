import torch
import torch.nn as nn
import torchtext; torchtext.disable_torchtext_deprecation_warning()
from torchtext.vocab import GloVe 



class WordEmbedding(nn.Module):
    def __init__(self, embedding_dim = 300, trainable = False):
        super(WordEmbedding, self).__init__()
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding.from_pretrained(GloVe(name='6B', dim=embedding_dim).vectors, freeze = not trainable)

    def forward(self, x):
        
        return self.embedding(x)

