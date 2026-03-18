import torch
import torch.nn as nn
import torch.nn.functional as F

class NeighborFeaturePooling(nn.Module):
    """
    Neighbor Feature Attention-based Pooling (NFP) Layer.
    Replaces standard Global Average Pooling.
    Based on Li et al. [29] as described in the manuscript.
    """
    def __init__(self, in_channels):
        super(NeighborFeaturePooling, self).__init__()
        
        # Encoder E: Depthwise Conv + BatchNorm
        # Captures spatial features and dependencies
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, groups=in_channels),
            nn.BatchNorm2d(in_channels)
        )
        
        # Attention Module: Generates attention weights via 1x1 Conv + Softmax
        self.attention_conv = nn.Conv2d(in_channels, 1, kernel_size=1)
        
    def forward(self, x):
        # x shape: (Batch, Channels, Height, Width)
        
        # 1. Encode spatial dependencies
        sd = self.encoder(x) # Spatial dependence maps
        
        # 2. Generate Attention Weights
        # Squeezing channel dimension to get (Batch, 1, H, W)
        aw = self.attention_conv(sd) 
        aw = F.softmax(aw.view(aw.size(0), -1), dim=1).view(aw.size()) # Softmax over spatial dimensions
        
        # 3. Apply Attention Pooling
        # Multiply input features by attention weights and sum
        out = x * aw
        out = torch.sum(out, dim=[2, 3], keepdim=True) # Sum over H and W -> (Batch, C, 1, 1)
        
        return out