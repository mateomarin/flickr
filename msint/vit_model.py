import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, 
                 image_size=56,
                 patch_size=7,
                 in_channels=1,
                 embed_dim=256):
        super().__init__()
        
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        
        patch_dim = in_channels * patch_size * patch_size
        self.projection = nn.Linear(patch_dim, embed_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, embed_dim))
        
    def forward(self, x):
        # Input: (batch_size, channels=1, height=56, width=56)
        batch_size = x.shape[0]
        
        # Extract patches using unfold
        # Shape: (batch_size, 1, 8, 8, 7, 7)
        patches = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        
        # Reshape patches
        # Shape: (batch_size, num_patches=64, patch_pixels=49)
        patches = patches.contiguous().view(batch_size, -1, self.patch_size * self.patch_size)
        
        # Linear projection
        # Shape: (batch_size, num_patches=64, embed_dim=256)
        x = self.projection(patches)
        
        # Add positional embedding (broadcasting from (1, 64, 256))
        # Shape: (batch_size, num_patches=64, embed_dim=256)
        x = x + self.pos_embedding
        
        return x

# Example usage:
if __name__ == "__main__":
    batch_size = 4
    sample_images = torch.randn(batch_size, 1, 56, 56)
    
    model = PatchEmbedding()
    output = model(sample_images)
    print(f"Input shape: {sample_images.shape}")
    print(f"Output shape: {output.shape}") 