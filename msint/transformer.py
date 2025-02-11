import torch
import torch.nn as nn
from vit_model import PatchEmbedding

class EncoderBlock(nn.Module):
    def __init__(self, 
                 embed_dim=256,
                 num_heads=8,
                 ff_dim=512):          # Smaller dimension
        super().__init__()
        
        # Multi-head Self Attention
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        
        # Layer Normalization
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # Simpler Feed Forward Network
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )
        
    def forward(self, x):
        # Self Attention with residual connection
        attention_output, _ = self.attention(x, x, x)
        x = x + attention_output
        x = self.norm1(x)
        
        # Feed Forward with residual connection
        ff_output = self.ff(x)
        x = x + ff_output
        x = self.norm2(x)
        
        return x

class DecoderBlock(nn.Module):
    def __init__(self, 
                 embed_dim=256,
                 num_heads=8,
                 ff_dim=512):
        super().__init__()
        
        # Self Attention
        self.self_attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        
        # Cross Attention
        self.cross_attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        
        # Layer Normalization
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        
        # Feed Forward
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )
        
    def forward(self, x, encoder_output):
        # Self Attention
        self_attn_output, _ = self.self_attention(x, x, x)
        x = x + self_attn_output
        x = self.norm1(x)
        
        # Cross Attention
        cross_attn_output, _ = self.cross_attention(x, encoder_output, encoder_output)
        x = x + cross_attn_output
        x = self.norm2(x)
        
        # Feed Forward
        ff_output = self.ff(x)
        x = x + ff_output
        x = self.norm3(x)
        
        return x

class VisionTransformer(nn.Module):
    def __init__(self,
                 image_size=56,
                 patch_size=7,
                 in_channels=1,
                 embed_dim=256,
                 num_heads=8,
                 num_layers=6,
                 ff_dim=512):
        super().__init__()
        
        # Patch Embedding
        self.patch_embed = PatchEmbedding(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim
        )
        
        # Transformer blocks (encoder + decoder)
        self.blocks = nn.ModuleList([
            # Encoder blocks
            *[EncoderBlock(embed_dim, num_heads, ff_dim) for _ in range(num_layers)],
            # Decoder blocks
            *[DecoderBlock(embed_dim, num_heads, ff_dim) for _ in range(num_layers)]
        ])
        
        # Layer Norm
        self.norm = nn.LayerNorm(embed_dim)
        
        # Initialize positional embeddings with smaller values
        self.digit_positions = nn.Parameter(torch.randn(4, embed_dim) * 0.02)
        
        # Initialize classifier with bias
        self.classifier = nn.Linear(embed_dim, 10)
        torch.nn.init.zeros_(self.classifier.bias)  # Start with zero bias
        
    def forward(self, x):
        # Encode
        encoded = self.patch_embed(x)
        
        # Encoder blocks (first half of self.blocks)
        for block in self.blocks[:len(self.blocks)//2]:
            encoded = block(encoded)
        encoded = self.norm(encoded)
        
        # Decode
        batch_size = x.shape[0]
        decoded = self.digit_positions.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Decoder blocks (second half of self.blocks)
        for block in self.blocks[len(self.blocks)//2:]:
            decoded = block(decoded, encoded)
            
        # Predict all digits using the same classifier
        digits = []
        for i in range(4):
            digit = self.classifier(decoded[:, i])
            digits.append(digit)
        
        # Add positional information about patch location
        B, C, H, W = x.shape
        positions = torch.arange(H*W, device=x.device).float() / (H*W)
        positions = positions.view(1, -1, 1)  # [1, H*W, 1]
        
        return digits  # Returns list of 4 digit predictions

if __name__ == "__main__":
    batch_size = 4
    sample_images = torch.randn(batch_size, 1, 56, 56)
    
    model = VisionTransformer()
    digits = model(sample_images)
    
    print(f"Input shape: {sample_images.shape}")
    for i, digit in enumerate(digits):
        print(f"Digit {i+1} output shape: {digit.shape}")  # Should be (batch_size, 10) 