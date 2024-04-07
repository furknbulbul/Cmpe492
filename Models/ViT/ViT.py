from torch import nn
from Models.ViT.patch_embedding import PatchEmbedding
from Models.ViT.TransformerEncoder import TransformerEncoderBlock
import torch


# 1. patch embedding
# 2. positional embedding
# 3. transformer encoder
# 3.1. multi-head self-attention block
# 3.1.1. layer normalization
# 3.1.2. multi-head self-attention
# 3.2. MLP
# 3.2.1. layer normalization
# 3.2.2. multi-layer perceptron





class ViT(nn.Module):
    def __init__(self,
                 img_size=48, # Training resolution from Table 3 in ViT paper
                 in_channels:int=1, # Number of channels in input image
                 patch_size:int=6, # Patch size, default is 16 for ViT-Base
                 num_transformer_layers=3, # Layers from Table 1 for ViT-Base, 12 in default
                 embedding_dim:int= 36, # Hidden size D from Table 1 for ViT-Base
                 mlp_size=144, # MLP size from Table 1 for ViT-Base, 36 * 4 = 144 in my case
                 num_heads=3, # Heads from Table 1 for ViT-Base,  default 12
                 attn_dropout=0, # Dropout for attention projection
                 mlp_dropout=0.1, # Dropout for dense/MLP layers 
                 embedding_dropout=0.1, # Dropout for patch and position embeddings
                 num_classes=7): # Default for ImageNet but can customize this
        super().__init__() # don't forget the super().__init__()!
        
        # 3. Make the image size is divisble by the patch size 
        assert img_size % patch_size == 0, f"Image size must be divisible by patch size, image size: {img_size}, patch size: {patch_size}."
        
        # 4. Calculate number of patches (height * width/patch^2)
        self.num_patches = (img_size * img_size) // patch_size**2
                 
        # 5. Create learnable class embedding (needs to go at front of sequence of patch embeddings)
        self.class_embedding = nn.Parameter(data=torch.randn(1, 1, embedding_dim),
                                            requires_grad=True)
        
        # 6. Create learnable position embedding
        self.position_embedding = nn.Parameter(data=torch.randn(1, self.num_patches+1, embedding_dim),
                                               requires_grad=True)
                
        # 7. Create embedding dropout value
        self.embedding_dropout = nn.Dropout(p=embedding_dropout)
        
        # 8. Create patch embedding layer
        self.patch_embedding = PatchEmbedding(in_channels=in_channels,
                                              patch_size=patch_size,
                                              embedding_dim=embedding_dim)
        
        # 9. Create Transformer Encoder blocks (we can stack Transformer Encoder blocks using nn.Sequential()) 
        self.transformer_encoder = nn.Sequential(*[TransformerEncoderBlock(embedding_dim=embedding_dim,
                                                                            num_heads=num_heads,
                                                                            mlp_size=mlp_size,
                                                                            mlp_dropout=mlp_dropout) for _ in range(num_transformer_layers)])
       
        # 10. Create classifier head
        self.classifier = nn.Sequential(
            nn.LayerNorm(normalized_shape=embedding_dim),
            nn.Linear(in_features=embedding_dim, 
                      out_features=num_classes)
        )
    
    
    def forward(self, x):
        
        
        batch_size = x.shape[0]
                
        class_token = self.class_embedding.expand(batch_size, -1, -1) # "-1" means to infer the dimension (try this line on its own)

        x = self.patch_embedding(x)

        # 15. Concat class embedding and patch embedding (equation 1)
        x = torch.cat((class_token, x), dim=1)

        # 16. Add position embedding to patch embedding (equation 1) 
        x = self.position_embedding + x

        # 17. Run embedding dropout (Appendix B.1)
        x = self.embedding_dropout(x)

        # 18. Pass patch, position and class embedding through transformer encoder layers (equations 2 & 3)
        x = self.transformer_encoder(x)

        # 19. Put 0 index logit through classifier (equation 4)
        x = self.classifier(x[:, 0]) # run on each sample in a batch at 0 index

        return x       
