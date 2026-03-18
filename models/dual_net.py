import torch
import torch.nn as nn
import timm
from models.nfp import NeighborFeaturePooling

class DualTransferNet(nn.Module):
    def __init__(self, num_classes=5):
        super(DualTransferNet, self).__init__()
        
        # 1. Load Backbones (Pre-trained on ImageNet)
        # Backbone A: DenseNet-169
        self.dense_net = timm.create_model('densenet169', pretrained=True, features_only=True, out_indices=(4,))
        dense_feat_channels = 1664 # DenseNet-169 final feature channels
        
        # Backbone B: InceptionResNet-V2 (Using Inception_ResNet_V2 via timm)
        self.inception_net = timm.create_model('inception_resnet_v2', pretrained=True, features_only=True, out_indices=(3,))
        inception_feat_channels = 1088 # InceptionResNet-V2 final feature channels
        
        # 2. Replace Pooling Layers with NFP
        self.nfp_dense = NeighborFeaturePooling(dense_feat_channels)
        self.nfp_inception = NeighborFeaturePooling(inception_feat_channels)
        
        # 3. Classifier
        # Fused feature size = DenseNet features + InceptionResNet features
        fused_features = dense_feat_channels + inception_feat_channels
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(fused_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        # Pass through DenseNet
        x_dense = self.dense_net(x)[0] # Output: (B, 1664, H, W)
        x_dense = self.nfp_dense(x_dense) # Output: (B, 1664, 1, 1)
        
        # Pass through InceptionResNet
        x_incep = self.inception_net(x)[0] # Output: (B, 1088, H, W)
        x_incep = self.nfp_inception(x_incep) # Output: (B, 1088, 1, 1)
        
        # Concatenate (Fusion)
        x_fused = torch.cat([x_dense, x_incep], dim=1) # Concatenate along channel dimension
        
        # Classification
        out = self.classifier(x_fused)
        return out