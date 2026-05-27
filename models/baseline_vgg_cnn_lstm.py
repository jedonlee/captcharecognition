# -*- coding: utf-8 -*-
"""
Baseline Model: VGG CNN + BiLSTM + CTC

Architecture:
1. VGG-style CNN feature extraction (4 conv blocks: 3->32->64->128->256)
2. AdaptiveAvgPool2d((1, 16)) for height pooling
3. Reshape to (batch, seq=16, features=256)
4. BiLSTM sequence modeling
5. FC classification layer
6. Dual output: (encoder_out, decoder_out) matching core model interface
"""
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class VGGConvBlock(nn.Module):
    """VGG-style convolutional module"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        
        self.pool = nn.MaxPool2d(2, 2)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool(x)
        return x


class BiLSTMDecoder(nn.Module):
    """
    Bidirectional LSTM decoder
    
    Args:
        input_dim: input feature dimension
        hidden_size: LSTM hidden dimension
        num_layers: number of LSTM layers
        dropout: dropout probability
    """
    
    def __init__(self, input_dim=256, hidden_size=256, num_layers=2, dropout=0.3):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        self.layer_norm = nn.LayerNorm(hidden_size * 2)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, features)
            
        Returns:
            output: (batch, seq_len, hidden_size * 2)
        """
        output, _ = self.lstm(x)
        output = self.layer_norm(output)
        output = self.dropout(output)
        return output


class BaselineVGGCNNBiLSTM(nn.Module):
    """Baseline model: VGG CNN + BiLSTM + CTC
    
    Architecture:
    - 4 VGGConvBlocks for CNN feature extraction (3->32->64->128->256)
    - AdaptiveAvgPool2d((1, 16)) for spatial pooling
    - BiLSTM for sequence modeling
    - FC layer for classification
    """
    
    def __init__(self, num_classes=63):
        super().__init__()
        self.num_classes = num_classes
        
        self.conv_blocks = nn.Sequential(
            VGGConvBlock(3, 32),
            VGGConvBlock(32, 64),
            VGGConvBlock(64, 128),
            VGGConvBlock(128, 256)
        )
        
        self.feature_pool = nn.AdaptiveAvgPool2d((1, 16))
        
        self.decoder = BiLSTMDecoder(
            input_dim=256,
            hidden_size=256,
            num_layers=2,
            dropout=0.3
        )
        
        self.fc = nn.Linear(512, num_classes)
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc.weight)
        if self.fc.bias is not None:
            nn.init.zeros_(self.fc.bias)
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, 3, 64, 256) input image
            
        Returns:
            encoder_out: (seq_len, batch, num_classes) with log_softmax
            decoder_out: (batch, seq_len, num_classes) with log_softmax
        """
        batch_size = x.size(0)
        
        features = self.conv_blocks(x)
        
        pooled = self.feature_pool(features)
        
        pooled = pooled.reshape(batch_size, 16, 256)
        
        output = self.decoder(pooled)
        
        logits = self.fc(output)
        
        encoder_out = logits.permute(0, 2, 1)
        encoder_out = encoder_out.permute(2, 0, 1)
        
        encoder_out = torch.clamp(encoder_out, min=-100.0, max=100.0)
        
        encoder_out = F.log_softmax(encoder_out, dim=2)
        
        decoder_out = encoder_out.permute(1, 0, 2)
        
        return encoder_out, decoder_out


if __name__ == '__main__':
    print("=" * 60)
    print("VGG CNN + BiLSTM - Baseline Model")
    print("=" * 60)
    
    num_classes = 63
    model = BaselineVGGCNNBiLSTM(num_classes=num_classes)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {num_params:,}")
    
    batch_size = 2
    x = torch.randn(batch_size, 3, 64, 256)
    print(f"\nInput tensor shape: {x.shape}")
    
    model.eval()
    with torch.no_grad():
        encoder_out, decoder_out = model(x)
    
    print(f"\nEncoder output shape (CTC): {encoder_out.shape}")
    print(f"  - Time steps: {encoder_out.shape[0]}")
    print(f"  - Batch size: {encoder_out.shape[1]}")
    print(f"  - Num classes: {encoder_out.shape[2]}")
    
    print(f"\nDecoder output shape: {decoder_out.shape}")
    print(f"  - Batch size: {decoder_out.shape[0]}")
    print(f"  - Sequence length: {decoder_out.shape[1]}")
    print(f"  - Num classes: {decoder_out.shape[2]}")
    
    print(f"\n" + "=" * 60)
    print("Model test passed!")
    print("=" * 60)
