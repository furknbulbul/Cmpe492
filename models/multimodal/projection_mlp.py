import torch.nn as nn


class ProjectionMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, is_text=False):
        super(ProjectionMLP, self).__init__()
        self.is_text = is_text
        self.projector = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        #return all the embeddings 
        # if self.is_text:
        #     batch_size, num_embeddings, embedding_dim = x.size()
        #     x = x.view(batch_size * num_embeddings, -1)
        #     x = self.projector(x)
        #     return x.view(batch_size, num_embeddings, -1)
        x = x.view(x.size(0), -1)
        return self.projector(x)