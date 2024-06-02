from torch import nn
import torch


# 1- turn an image into patches
# 2- flatten the patch feature maps into a single dimension
# 3- Convert the output into Desried output (flattened 2D patches):

class PatchEmbedding(nn.Module):
    """Turns a 2D input image into a 1D sequence learnable embedding vector.

    Args:
        in_channels (int): Number of color channels for the input images. Defaults to 3.
        patch_size (int): Size of patches to convert input image into. Defaults to 16.
        embedding_dim (int): Size of embedding to turn image into. Defaults to 768.
    """

    
    def __init__(self,
                 in_channels: int = 1,
                 patch_size: int = 6, # 8*8 patches in total
                 embedding_dim: int = 36): # patch size 6x6 x channel  =  36 patches in total
        super().__init__()

        
        self.patcher = nn.Conv2d(in_channels=in_channels,
                                 out_channels=embedding_dim,
                                 kernel_size=patch_size,
                                 stride=patch_size,
                                 padding=0)

        
        self.flatten = nn.Flatten(start_dim=2,  # flattening operation will only flatten the height and width dimensions, leaving the batch size and channel dimensions unchanged.
                                  end_dim=3)

    def forward(self, x):

        x_patched = self.patcher(x)
        x_flattened = self.flatten(x_patched)

        # 6. Make sure the output shape has the right order
        return x_flattened.permute(0, 2, 1)  # adjust so the embedding is on the final dimension [batch_size, P^2•C, N] -> [batch_size, N, P^2•C]


