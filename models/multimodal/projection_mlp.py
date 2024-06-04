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

        x = x.view(x.size(0), -1)
        return self.projector(x)