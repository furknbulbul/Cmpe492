import torch
import torch.nn as nn

class EuclideanDistanceLoss(nn.Module):
    def __init__(self):
        super(EuclideanDistanceLoss, self).__init__()

    def forward(self, image_embedding, text_embeddings):
        # print("Image embedding:", image_embedding.shape)
        # print("Text embeddings:", text_embeddings.shape)
        return torch.norm(image_embedding - text_embeddings, dim=1, p=2).mean()
        # image_embedding_exp = image_embedding.unsqueeze(1).expand(-1, text_embeddings.shape[1], -1)
        # return torch.norm(image_embedding_exp - text_embeddings, dim=2, p=2).mean()
