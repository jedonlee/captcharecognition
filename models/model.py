# -*- coding: utf-8 -*-
"""
CAPTCHA Recognition Model - ConvNeXt V2-Tiny + Transformer/BiLSTM + CTC/CE Hybrid Loss

Tech Stack:
  - Backbone: ConvNeXt V2-Tiny (64×256) + CBAM
  - Decoder: TransformerEncoder (default) or BiLSTM (switched via config.model.core.decoder_type)
  - Loss: CTC + CE hybrid loss
"""

import os
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.config_loader import get_config

logger = logging.getLogger(__name__)


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module (CBAM)
    Channel Attention + Spatial Attention
    """
    def __init__(self, channels):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.channel_mlp = nn.Sequential(
            nn.Linear(channels, channels // 16, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // 16, channels, bias=False)
        )
        self.channel_sigmoid = nn.Sigmoid()
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.spatial_sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.shape
        avg_out = self.avg_pool(x).view(b, c)
        max_out = self.max_pool(x).view(b, c)
        channel_att = self.channel_sigmoid(self.channel_mlp(avg_out) + self.channel_mlp(max_out)).view(b, c, 1, 1)
        x = x * channel_att
        avg_spatial = torch.mean(x, dim=1, keepdim=True)
        max_spatial, _ = torch.max(x, dim=1, keepdim=True)
        spatial_input = torch.cat([avg_spatial, max_spatial], dim=1)
        spatial_att = self.spatial_sigmoid(self.spatial_conv(spatial_input))
        x = x * spatial_att
        return x


class PositionalEncoding(nn.Module):
    """
    Positional Encoding (Sine-Cosine)
    """
    def __init__(self, d_model, dropout=0.1, max_len=None):
        super().__init__()
        if max_len is None:
            max_len = get_config().get("model.core.time_steps", 16)
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerSeqDecoder(nn.Module):
    """
    Transformer sequence decoder (replaces BiLSTM)
    Uses bidirectional self-attention, suitable for CTC alignment tasks

    Args:
        input_dim: input feature dimension (backbone output)
        d_model: Transformer hidden dimension
        nhead: number of attention heads
        num_layers: number of Transformer layers
        dim_feedforward: FFN hidden dimension
        dropout: dropout probability
    """

    def __init__(self, input_dim=768, d_model=512, nhead=8, num_layers=2,
                 dim_feedforward=2048, dropout=0.3):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation='gelu',
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        x = self.layer_norm(x)
        x = self.dropout(x)
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

    def __init__(self, input_dim=768, hidden_size=256, num_layers=2, dropout=0.3):
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


class CaptchaModel(nn.Module):
    """
    CAPTCHA Recognition Model: ConvNeXt V2-Tiny + BiLSTM/Transformer + CTC/CE

    Architecture:
    1. ConvNeXt V2-Tiny backbone for feature extraction
    2. CBAM channel attention
    3. Adaptive pooling to pool_size
    4. Reshape to (batch, seq=time_steps, features=backbone_channels)
    5. BiLSTM/Transformer for sequence modeling
    6. Linear classification layer -> log_softmax
    7. CTC Loss for sequence alignment

    Input/Output shapes:
        forward(x):
            x:          (B, 3, H, W)           — input image
            features:   (B, C, H/32, W/32)     — backbone feature map
            pooled:     (B, C, pool_h, pool_w) — after pooling
            seq:        (B, time_steps, C)     — flattened sequence
            output:     (B, time_steps, d_model) — post sequence decoding
            logits:     (B, time_steps, num_classes) — classification output
            encoder_out:(T, B, C')              — CTC input (log_softmax applied)
            decoder_out:(B, T, C')              — CTC-aligned copy
    """

    def __init__(self, num_chars=None, max_length=None, pretrained=None):
        super().__init__()

        config = get_config()
        if num_chars is None:
            num_chars = config.get_total_classes() - 1
        self.num_classes = num_chars + 1

        self.backbone = self._create_backbone()

        with torch.no_grad():
            dummy = torch.zeros(1, 3, 64, 256)
            dummy_out = self.backbone.forward_features(dummy)
            self.backbone_channels = dummy_out.shape[1]

        self.cbam = CBAM(self.backbone_channels)
        pool_h, pool_w = config.get("model.core.pool_size", [2, 8])
        self.feature_pool = nn.AdaptiveAvgPool2d((pool_h, pool_w))
        self.time_steps = config.get("model.core.time_steps", 16)

        decoder_config = config.get("model.core.decoder", {})
        transformer_config = config.get("model.core.transformer_encoder", {})
        decoder_type = config.get("model.core.decoder_type", "bilstm")

        if decoder_type == "transformer":
            self.decoder = TransformerSeqDecoder(
                input_dim=self.backbone_channels,
                d_model=transformer_config.get("d_model", 512),
                nhead=transformer_config.get("nhead", 8),
                num_layers=transformer_config.get("num_layers", 2),
                dim_feedforward=transformer_config.get("dim_feedforward", 2048),
                dropout=transformer_config.get("dropout", 0.3),
            )
            decoder_out_dim = transformer_config.get("d_model", 512)
        else:
            lstm_hidden_size = decoder_config.get("hidden_size", 256)
            lstm_num_layers = decoder_config.get("num_layers", 2)
            lstm_dropout = decoder_config.get("dropout", 0.3)

            self.decoder = BiLSTMDecoder(
                input_dim=self.backbone_channels,
                hidden_size=lstm_hidden_size,
                num_layers=lstm_num_layers,
                dropout=lstm_dropout
            )
            decoder_out_dim = lstm_hidden_size * 2

        self.extra_dropout = nn.Dropout(0.2)

        self.fc = nn.Linear(decoder_out_dim, self.num_classes)

        self._init_weights()

    def _create_backbone(self):
        try:
            import timm
        except ImportError:
            raise ImportError("timm library is required: pip install timm")

        config = get_config()
        local_weights = str(config.get_project_root() / config.get('checkpoint.checkpoint_dir', 'checkpoints') / 'model.safetensors')

        try:
            backbone = timm.create_model(
                "convnextv2_tiny",
                pretrained=True,
                in_chans=3,
                num_classes=0,
            )
            logger.info("Using backbone: ConvNeXt V2-Tiny (pretrained weights)")
        except Exception as e:
            logger.warning(f"Pretrained loading failed, trying local file: {type(e).__name__}")
            if os.path.exists(local_weights):
                try:
                    import torch
                    state_dict = torch.load(local_weights, map_location='cpu', weights_only=True)
                    if isinstance(state_dict, dict):
                        if 'model' in state_dict:
                            state_dict = state_dict['model']
                        elif 'state_dict' in state_dict:
                            state_dict = state_dict['state_dict']
                    backbone = timm.create_model(
                        "convnextv2_tiny",
                        pretrained=False,
                        in_chans=3,
                        num_classes=0,
                    )
                    result = backbone.load_state_dict(state_dict, strict=False)
                    logger.info("Using backbone: ConvNeXt V2-Tiny (local pretrained weights)")
                    logger.info(f"  missing={len(result.missing_keys)}, unexpected={len(result.unexpected_keys)}")
                except Exception as e2:
                    logger.warning(f"Local loading also failed: {type(e2).__name__}, using random initialization")
                    backbone = timm.create_model(
                        "convnextv2_tiny",
                        pretrained=False,
                        in_chans=3,
                        num_classes=0,
                    )
                    logger.info("Using backbone: ConvNeXt V2-Tiny (random initialization)")
            else:
                logger.warning(f"Local weights file not found: {local_weights}, using random initialization")
                backbone = timm.create_model(
                    "convnextv2_tiny",
                    pretrained=False,
                    in_chans=3,
                    num_classes=0,
                )
                logger.info("Using backbone: ConvNeXt V2-Tiny (random initialization)")

        # Modify stem stride from 4 to 2, reduce total downsampling factor
        try:
            # ConvNeXt stem is a Sequential, first element is Conv2d
            if hasattr(backbone, 'stem') and hasattr(backbone.stem, '0'):
                stem_conv = backbone.stem[0]
                if hasattr(stem_conv, 'stride'):
                    old_stride = stem_conv.stride
                    stem_conv.stride = (2, 2)
                    logger.info(f"Stem stride modified: {old_stride} -> (2, 2)")
        except Exception as e:
            logger.warning(f"Failed to modify stem stride: {e}")

        return backbone

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc.weight)
        if self.fc.bias is not None:
            nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        batch_size = x.shape[0]

        features = self.backbone.forward_features(x)

        features = self.cbam(features)

        pooled = self.feature_pool(features)
        pooled = pooled.reshape(batch_size, self.time_steps, self.backbone_channels)

        output = self.decoder(pooled)

        output = self.extra_dropout(output)

        logits = self.fc(output)

        encoder_out = logits.permute(0, 2, 1)
        encoder_out = encoder_out.permute(2, 0, 1)

        encoder_out = torch.clamp(encoder_out, min=-100.0, max=100.0)

        encoder_out = F.log_softmax(encoder_out, dim=2)

        decoder_out = encoder_out.permute(1, 0, 2)

        return encoder_out, decoder_out


def create_model_from_config():
    config = get_config()
    num_chars = config.get_total_classes() - 1
    return CaptchaModel(num_chars=num_chars)


if __name__ == '__main__':
    print("=" * 60)
    print("ConvNeXt V2-Tiny + BiLSTM - CAPTCHA Recognition Model")
    print("=" * 60)

    config = get_config()
    num_chars = config.get_total_classes() - 1
    max_length = config.get_max_length()
    img_height, img_width = config.get_preprocessed_image_size()

    print(f"\nImage input size: {img_height} x {img_width}")

    model = CaptchaModel(num_chars=num_chars, max_length=max_length)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {num_params:,}")

    batch_size = 2
    x = torch.randn(batch_size, 3, img_height, img_width)

    print(f"\nInput tensor shape: {x.shape}")

    model.eval()
    with torch.no_grad():
        encoder_out, decoder_out = model(x)

    print(f"\nEncoder output shape (CTC): {encoder_out.shape}")
    print(f"  - Time steps: {encoder_out.shape[0]}")
    print(f"  - Batch size: {encoder_out.shape[1]}")
    print(f"  - Num classes: {encoder_out.shape[2]}")

    print(f"\nDecoder output shape: {decoder_out.shape}")

    print(f"\nTesting hybrid loss function...")
    from models.hybrid_loss_fixed import HybridCTCELoss

    criterion = HybridCTCELoss(
        ctc_weight=config.get('training.ctc_weight', 0.6),
        ce_weight=config.get('training.ce_weight', 0.4),
        label_smoothing=config.get('training.label_smoothing', 0.1)
    )

    targets_list = []
    target_lengths = []
    for i in range(batch_size):
        length = torch.randint(4, 7, (1,)).item()
        target = torch.randint(0, num_chars, (length,))
        targets_list.append(target)
        target_lengths.append(length)

    targets = torch.cat(targets_list)
    target_lengths = torch.tensor(target_lengths, dtype=torch.long)
    input_lengths = torch.tensor([self.time_steps] * batch_size, dtype=torch.long)

    loss, ctc_loss, ce_loss = criterion(encoder_out, decoder_out, targets, input_lengths, target_lengths)

    print(f"Total loss: {loss.item():.4f}")
    print(f"CTC loss: {ctc_loss.item():.4f}")
    print(f"CE loss: {ce_loss.item():.4f}")

    print(f"\n" + "=" * 60)
    print("Model created successfully!")
    print("=" * 60)