# -*- coding: utf-8 -*-
"""
CAPTCHA Recognition Model - Ablation Experiment Framework

5 ablation experiments:
1. full:          Complete core model (ConvNeXt V2-Tiny + CBAM + BiLSTM + CTC/CE)
2. no_cbam:       Remove CBAM attention module
3. no_ctc:        Remove CTC loss (CE only, ctc_weight=0, ce_weight=1)
4. resnet34:      Replace backbone with ResNet-34
5. no_bilstm:     Remove BiLSTM (CNN only + CTC/CE, use_bilstm=False)

All experiments share the same streaming training protocol:
- Streaming dataset (100K samples/epoch)
- Unified hyperparameters (epochs=15, batch_size=64, lr=2e-4)
- Evaluate on fixed val set every 2 epochs
- Best model saved to checkpoints/ablation_<exp>_best.pth

CLI usage:
  python run_ablation.py --experiments all
  python run_ablation.py --experiments full,no_cbam
  python run_ablation.py --experiments all --skip_training
"""

import os
import sys
import json
import time
import gc
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from contextlib import nullcontext
from tqdm import tqdm
import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from utils.config_loader import get_config
from utils.device_manager import DeviceManager
from utils.directory_manager import DirectoryManager
from utils.training_utils import (
    MemoryMonitor,
    setup_logging,
    set_seed,
    save_checkpoint,
)
from utils.decoder import beam_search_decode, postprocess_text_list, normalize_equiv
from utils.chars import CharMapper, get_all_chars
from models.hybrid_loss_fixed import HybridCTCELoss
from models.dataset import StreamCaptchaDataset, CaptchaDataset, collate_fn
from models.transforms import get_train_transforms, get_val_transforms


EXPERIMENT_CONFIGS = {
    "full": {
        "description": "Complete model (ConvNeXt V2-Tiny + CBAM + BiLSTM + CTC/CE)",
        "backbone": "convnextv2_tiny",
        "use_cbam": True,
        "use_bilstm": True,
        "ctc_weight": 0.6,
        "ce_weight": 0.4,
        "pretrained": True,
    },
    "no_cbam": {
        "description": "Remove CBAM attention module",
        "backbone": "convnextv2_tiny",
        "use_cbam": False,
        "use_bilstm": True,
        "ctc_weight": 0.6,
        "ce_weight": 0.4,
        "pretrained": True,
    },
    "no_ctc": {
        "description": "Remove CTC loss (CE only)",
        "backbone": "convnextv2_tiny",
        "use_cbam": True,
        "use_bilstm": True,
        "ctc_weight": 0.0,
        "ce_weight": 1.0,
        "pretrained": True,
    },
    "resnet34": {
        "description": "Replace backbone with ResNet-34",
        "backbone": "resnet34",
        "use_cbam": True,
        "use_bilstm": True,
        "ctc_weight": 0.6,
        "ce_weight": 0.4,
        "pretrained": True,
    },
    "no_bilstm": {
        "description": "Remove BiLSTM (CNN only + CTC/CE)",
        "backbone": "convnextv2_tiny",
        "use_cbam": True,
        "use_bilstm": False,
        "ctc_weight": 0.6,
        "ce_weight": 0.4,
        "pretrained": True,
    },
}


class CBAM(nn.Module):
    """Convolutional Block Attention Module (CBAM)"""

    def __init__(self, channels: int):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.channel_mlp = nn.Sequential(
            nn.Linear(channels, channels // 16, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // 16, channels, bias=False),
        )
        self.channel_sigmoid = nn.Sigmoid()
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.spatial_sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        avg_out = self.avg_pool(x).view(b, c)
        max_out = self.max_pool(x).view(b, c)
        channel_att = self.channel_sigmoid(
            self.channel_mlp(avg_out) + self.channel_mlp(max_out)
        ).view(b, c, 1, 1)
        x = x * channel_att
        avg_spatial = torch.mean(x, dim=1, keepdim=True)
        max_spatial, _ = torch.max(x, dim=1, keepdim=True)
        spatial_input = torch.cat([avg_spatial, max_spatial], dim=1)
        spatial_att = self.spatial_sigmoid(self.spatial_conv(spatial_input))
        x = x * spatial_att
        return x


class BiLSTMDecoder(nn.Module):
    """Bidirectional LSTM decoder"""

    def __init__(
        self,
        input_dim: int = 768,
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
        )
        self.layer_norm = nn.LayerNorm(hidden_size * 2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(x)
        output = self.layer_norm(output)
        output = self.dropout(output)
        return output


class AblationModel(nn.Module):
    """
    Parameterized model for ablation experiments.

    Supports:
    - Backbone: convnextv2_tiny or resnet34
    - CBAM: optional attention module
    - BiLSTM: optional sequence modeling
    """

    def __init__(
        self,
        num_classes: int,
        backbone: str = "convnextv2_tiny",
        use_cbam: bool = True,
        use_bilstm: bool = True,
        pretrained: bool = True,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.backbone_name = backbone
        self.use_cbam = use_cbam
        self.use_bilstm = use_bilstm

        self.backbone = self._create_backbone(backbone, pretrained)

        if backbone == "resnet34":
            backbone_channels = 512
        else:
            backbone_channels = 768

        self.cbam = CBAM(backbone_channels) if use_cbam else None
        self.feature_pool = nn.AdaptiveAvgPool2d((1, 16))

        config = get_config()
        decoder_config = config.get("model.core.decoder", {})
        lstm_hidden_size = decoder_config.get("hidden_size", 256)
        lstm_num_layers = decoder_config.get("num_layers", 2)
        lstm_dropout = decoder_config.get("dropout", 0.3)

        if use_bilstm:
            self.decoder = BiLSTMDecoder(
                input_dim=backbone_channels,
                hidden_size=lstm_hidden_size,
                num_layers=lstm_num_layers,
                dropout=lstm_dropout,
            )
            decoder_out_dim = lstm_hidden_size * 2
        else:
            self.decoder = None
            decoder_out_dim = backbone_channels

        self.extra_dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(decoder_out_dim, num_classes)
        self._init_weights()

    def _create_backbone(
        self, backbone_name: str, pretrained: bool
    ) -> nn.Module:
        try:
            import timm
        except ImportError:
            raise ImportError("timm is required: pip install timm")

        if backbone_name == "resnet34":
            backbone = timm.create_model(
                "resnet34",
                pretrained=pretrained,
                in_chans=3,
                num_classes=0,
            )
        else:
            try:
                backbone = timm.create_model(
                    "convnextv2_tiny",
                    pretrained=pretrained,
                    in_chans=3,
                    num_classes=0,
                )
                if hasattr(backbone, "stem") and hasattr(backbone.stem, "0"):
                    stem_conv = backbone.stem[0]
                    if hasattr(stem_conv, "stride"):
                        stem_conv.stride = (2, 2)
            except Exception:
                backbone = timm.create_model(
                    "convnextv2_tiny",
                    pretrained=False,
                    in_chans=3,
                    num_classes=0,
                )

        return backbone

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc.weight)
        if self.fc.bias is not None:
            nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = x.shape[0]

        if self.backbone_name == "resnet34":
            features = self.backbone.forward_features(x)
            if features.dim() == 2:
                b, c = features.shape
                features = features.view(b, c, 1, 1)
        else:
            features = self.backbone.forward_features(x)

        if self.use_cbam and self.cbam is not None:
            features = self.cbam(features)

        pooled = self.feature_pool(features)

        if self.backbone_name == "resnet34":
            pooled = pooled.reshape(batch_size, 16, 512)
            if self.use_bilstm and self.decoder is not None:
                output = self.decoder(pooled)
            else:
                output = pooled
        else:
            pooled = pooled.reshape(batch_size, 16, 768)
            if self.use_bilstm and self.decoder is not None:
                output = self.decoder(pooled)
            else:
                output = pooled

        output = self.extra_dropout(output)
        logits = self.fc(output)

        encoder_out = logits.permute(0, 2, 1)
        encoder_out = encoder_out.permute(2, 0, 1)
        encoder_out = torch.clamp(encoder_out, min=-100.0, max=100.0)
        encoder_out = F.log_softmax(encoder_out, dim=2)
        decoder_out = encoder_out.permute(1, 0, 2)

        return encoder_out, decoder_out


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    logger: logging.Logger,
    scaler: Optional[GradScaler] = None,
    scheduler: Optional[Any] = None,
    memory_monitor: Optional[MemoryMonitor] = None,
) -> Tuple[float, float, float]:
    """Train for one epoch."""
    model.train()

    total_loss = 0.0
    total_ctc_loss = 0.0
    total_ce_loss = 0.0
    total_batches = 0
    moving_avg_loss = 0.0
    alpha = 0.1

    device_manager = DeviceManager(verbose=False)
    use_amp = device_manager.supports_amp() and scaler is not None

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]", leave=False)

    for batch in progress_bar:
        try:
            if not batch or "images" not in batch:
                continue
            if batch["images"].numel() == 0:
                continue

            non_blocking = device.type == "cuda"
            images = batch["images"].to(device, non_blocking=non_blocking)
            label_indices = batch["label_indices"].to(device, non_blocking=non_blocking)
            label_lengths = batch["label_lengths"].to(device, non_blocking=non_blocking)

            targets_list = []
            for i in range(len(label_lengths)):
                targets_list.append(label_indices[i][: label_lengths[i]])
            targets = torch.cat(targets_list)
            targets_lengths = label_lengths

            if use_amp:
                amp_ctx = autocast("cuda")
            else:
                amp_ctx = nullcontext()

            with amp_ctx:
                encoder_out, decoder_out = model(images)
                batch_size = images.size(0)
                input_lengths = torch.full(
                    (batch_size,), encoder_out.size(0), dtype=torch.long, device=device
                )

                if torch.isnan(encoder_out).any() or torch.isinf(encoder_out).any():
                    continue
                if torch.isnan(decoder_out).any() or torch.isinf(decoder_out).any():
                    continue

                loss, ctc_loss, ce_loss = criterion(
                    encoder_out, decoder_out, targets, input_lengths, targets_lengths
                )

            if torch.isnan(loss) or torch.isinf(loss):
                continue

            optimizer.zero_grad()
            if use_amp and scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            if scheduler is not None:
                scheduler.step()

            total_loss += loss.item()
            total_ctc_loss += ctc_loss.item()
            total_ce_loss += ce_loss.item()
            total_batches += 1

            moving_avg_loss = alpha * loss.item() + (1 - alpha) * moving_avg_loss
            progress_bar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "avg": f"{moving_avg_loss:.4f}",
            })

            del loss, ctc_loss, ce_loss, encoder_out, decoder_out
            del images, label_indices, label_lengths, targets, targets_lengths, input_lengths

        except RuntimeError as e:
            logger.exception(f"RuntimeError in batch: {e}")
            continue
        except Exception as e:
            logger.exception(f"Error in batch: {e}")
            continue

    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    if total_batches > 0:
        avg_loss = total_loss / total_batches
        avg_ctc_loss = total_ctc_loss / total_batches
        avg_ce_loss = total_ce_loss / total_batches
    else:
        avg_loss, avg_ctc_loss, avg_ce_loss = 0.0, 0.0, 0.0

    return avg_loss, avg_ctc_loss, avg_ce_loss


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    logger: logging.Logger,
) -> Tuple[float, float, float, float, float]:
    """Validate model on fixed val set."""
    model.eval()

    total_loss = 0.0
    total_ctc_loss = 0.0
    total_ce_loss = 0.0
    total_batches = 0
    total_images = 0
    correct_images = 0
    total_chars = 0
    correct_chars = 0

    chars = get_all_chars()
    mapper = CharMapper.get_instance()

    try:
        with torch.no_grad():
            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch} [Val]", leave=False)
            for batch in progress_bar:
                try:
                    images = batch["images"].to(device)
                    label_indices = batch["label_indices"].to(device)
                    label_lengths = batch["label_lengths"].to(device)

                    encoder_out, decoder_out = model(images)

                    batch_size = images.size(0)
                    input_lengths = torch.full(
                        (batch_size,), encoder_out.size(0), dtype=torch.long, device=device
                    )

                    targets_list = []
                    for i in range(len(label_lengths)):
                        targets_list.append(label_indices[i][: label_lengths[i]])
                    targets = torch.cat(targets_list)
                    targets_lengths = label_lengths

                    loss, ctc_loss, ce_loss = criterion(
                        encoder_out, decoder_out, targets, input_lengths, targets_lengths
                    )

                    total_loss += loss.item()
                    total_ctc_loss += ctc_loss.item()
                    total_ce_loss += ce_loss.item()
                    total_batches += 1

                    try:
                        pred_strings = beam_search_decode(encoder_out, mapper, beam_width=10, enable_corrector=False)
                        pred_strings = postprocess_text_list(pred_strings)

                        target_strings = []
                        for i in range(batch_size):
                            target_len = label_lengths[i].item()
                            target_indices = label_indices[i, :target_len].tolist()
                            target_str = "".join([chars[idx] for idx in target_indices])
                            target_strings.append(target_str)

                        for pred_str, target_str in zip(pred_strings, target_strings):
                            pred_norm = normalize_equiv(pred_str)
                            target_norm = normalize_equiv(target_str)
                            total_images += 1
                            if pred_norm == target_norm:
                                correct_images += 1
                            for p_char, t_char in zip(pred_norm, target_norm):
                                total_chars += 1
                                if p_char == t_char:
                                    correct_chars += 1

                    except Exception as decode_error:
                        logger.warning(f"Decode error: {decode_error}")

                    del loss, ctc_loss, ce_loss, encoder_out, decoder_out
                    del images, label_indices, label_lengths, targets, targets_lengths, input_lengths

                except Exception as e:
                    logger.error(f"Validation error at batch: {e}")
                    continue

    except Exception as e:
        logger.error(f"Validation error: {e}")

    gc.collect()

    if total_batches > 0:
        avg_loss = total_loss / total_batches
        avg_ctc_loss = total_ctc_loss / total_batches
        avg_ce_loss = total_ce_loss / total_batches
    else:
        avg_loss, avg_ctc_loss, avg_ce_loss = 0.0, 0.0, 0.0

    image_accuracy = correct_images / total_images if total_images > 0 else 0.0
    char_accuracy = correct_chars / total_chars if total_chars > 0 else 0.0

    return avg_loss, avg_ctc_loss, avg_ce_loss, image_accuracy, char_accuracy


def run_single_experiment(
    experiment_name: str,
    experiment_config: Dict[str, Any],
    ablation_config: Dict[str, Any],
    device: torch.device,
    train_loader: DataLoader,
    val_loader: DataLoader,
    logger: logging.Logger,
    project_root: Path,
) -> Dict[str, Any]:
    """Run a single ablation experiment."""
    name = experiment_name
    logger.info("=" * 80)
    logger.info(f"Starting ablation experiment: {name}")
    logger.info(f"  Description: {experiment_config['description']}")
    logger.info(
        f"  Backbone={experiment_config['backbone']}, "
        f"CBAM={experiment_config['use_cbam']}, "
        f"BiLSTM={experiment_config['use_bilstm']}, "
        f"CTC/CE={experiment_config['ctc_weight']}/{experiment_config['ce_weight']}"
    )
    logger.info("=" * 80)

    config = get_config()
    num_classes = config.get_total_classes()

    model = AblationModel(
        num_classes=num_classes,
        backbone=experiment_config["backbone"],
        use_cbam=experiment_config["use_cbam"],
        use_bilstm=experiment_config["use_bilstm"],
        pretrained=experiment_config["pretrained"],
    )
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"  Parameters: {total_params:,} (trainable: {trainable_params:,})")

    epochs = ablation_config.get("epochs", 15)
    batch_size = ablation_config.get("batch_size", 64)
    lr = float(ablation_config.get("learning_rate", 2e-4))
    max_lr = float(ablation_config.get("max_lr", 1e-3))
    weight_decay = float(ablation_config.get("weight_decay", 5e-5))
    warmup_epochs = ablation_config.get("warmup_epochs", 2)

    label_smoothing = float(config.get('training.label_smoothing', 0.1))
    criterion = HybridCTCELoss(
        ctc_weight=experiment_config["ctc_weight"],
        ce_weight=experiment_config["ce_weight"],
        label_smoothing=label_smoothing,
    )
    criterion = criterion.to(device)
    logger.info(f"  Loss weights: CTC={criterion.ctc_weight}, CE={criterion.ce_weight}")

    trainable_params_list = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params_list, lr=lr, weight_decay=weight_decay)

    steps_per_epoch = len(train_loader)
    total_steps = epochs * steps_per_epoch
    pct_start = warmup_epochs / epochs if epochs > 0 else 0.2

    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=max_lr,
        total_steps=total_steps,
        pct_start=pct_start,
        anneal_strategy="cos",
        cycle_momentum=False,
        div_factor=10.0,
        final_div_factor=1000.0,
    )

    device_manager = DeviceManager(verbose=False)
    scaler = GradScaler() if device_manager.supports_amp() and device.type == "cuda" else None
    use_amp_str = "enabled" if scaler is not None else "disabled"
    logger.info(f"  AMP: {use_amp_str}, Scheduler: OneCycleLR (max_lr={max_lr:.6f})")

    val_interval = 2

    best_val_accuracy = 0.0
    best_epoch = 0
    epochs_no_improve = 0
    patience = max(5, epochs // 3)
    train_history = []
    val_history = []

    checkpoint_dir = project_root / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    training_start_time = time.time()

    for epoch in range(epochs):
        epoch_start_time = time.time()
        logger.info(f"\n--- Epoch {epoch + 1}/{epochs} ---")

        train_loss, train_ctc, train_ce = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch + 1,
            logger=logger,
            scaler=scaler,
            scheduler=scheduler,
        )

        epoch_time = time.time() - epoch_start_time

        if (epoch + 1) % val_interval == 0:
            val_loss, val_ctc, val_ce, val_img_acc, val_char_acc = validate(
                model=model,
                dataloader=val_loader,
                criterion=criterion,
                device=device,
                epoch=epoch + 1,
                logger=logger,
            )

            logger.info(
                f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                f"Val Accuracy: {val_img_acc * 100:.2f}% ({val_char_acc * 100:.2f}% char) | "
                f"Time: {epoch_time:.1f}s"
            )

            train_history.append({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_ctc": train_ctc,
                "train_ce": train_ce,
            })
            val_history.append({
                "epoch": epoch + 1,
                "val_loss": val_loss,
                "val_ctc": val_ctc,
                "val_ce": val_ce,
                "val_img_acc": val_img_acc,
                "val_char_acc": val_char_acc,
            })

            if val_img_acc > best_val_accuracy:
                best_val_accuracy = val_img_acc
                best_char_acc = val_char_acc
                best_epoch = epoch + 1
                epochs_no_improve = 0

                best_ckpt_path = checkpoint_dir / f"ablation_{name}_best.pth"
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "scaler_state_dict": scaler.state_dict() if scaler else None,
                        "best_val_accuracy": best_val_accuracy,
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "val_img_acc": val_img_acc,
                        "val_char_acc": val_char_acc,
                        "experiment_config": experiment_config,
                        "ablation_config": ablation_config,
                    },
                    best_ckpt_path,
                )
                logger.info(
                    f"  ** New best! Accuracy: {best_val_accuracy * 100:.2f}% "
                    f"(saved to ablation_{name}_best.pth)"
                )
            else:
                epochs_no_improve += 1
                logger.info(
                    f"  Best accuracy: {best_val_accuracy * 100:.2f}% "
                    f"(no improvement for {epochs_no_improve}/{patience} epochs)"
                )

            if epochs_no_improve >= patience:
                logger.info(f"  Early stopping triggered after {epoch + 1} epochs")
                break

        else:
            logger.info(
                f"  Train Loss: {train_loss:.4f} | Val: skipped (odd epoch) | "
                f"Time: {epoch_time:.1f}s"
            )
            train_history.append({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_ctc": train_ctc,
                "train_ce": train_ce,
            })

    total_training_time = time.time() - training_start_time

    logger.info(f"\nExperiment '{name}' completed!")
    logger.info(f"  Best validation accuracy: {best_val_accuracy * 100:.2f}% (epoch {best_epoch})")
    logger.info(f"  Total training time: {total_training_time / 60:.2f} min")

    result = {
        "experiment_name": name,
        "description": experiment_config["description"],
        "backbone": experiment_config["backbone"],
        "use_cbam": experiment_config["use_cbam"],
        "use_bilstm": experiment_config["use_bilstm"],
        "ctc_weight": experiment_config["ctc_weight"],
        "ce_weight": experiment_config["ce_weight"],
        "best_val_accuracy": best_val_accuracy,
        "best_val_char_accuracy": locals().get("best_char_acc", 0.0),
        "best_epoch": best_epoch,
        "total_training_time_sec": total_training_time,
        "total_epochs_trained": len(train_history),
        "train_history": train_history,
        "val_history": val_history,
        "checkpoint_path": str(checkpoint_dir / f"ablation_{name}_best.pth"),
    }

    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return result


def generate_results_table(
    results: List[Dict[str, Any]],
    project_root: Path,
    logger: logging.Logger,
):
    """Generate JSON, Markdown, and chart for ablation results."""
    results_dir = project_root / "results"
    figures_dir = results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    json_path = results_dir / "ablation_results.json"
    md_path = results_dir / "ablation_report.md"
    chart_path = figures_dir / "ablation_comparison.png"

    summary_data = []
    for r in results:
        summary_data.append({
            "experiment_name": r["experiment_name"],
            "description": r["description"],
            "backbone": r["backbone"],
            "use_cbam": r["use_cbam"],
            "use_bilstm": r["use_bilstm"],
            "ctc_weight": r["ctc_weight"],
            "ce_weight": r["ce_weight"],
            "best_val_accuracy": r["best_val_accuracy"],
            "best_val_char_accuracy": r.get("best_val_char_accuracy", 0.0),
            "best_epoch": r["best_epoch"],
            "total_epochs_trained": r["total_epochs_trained"],
            "total_training_time_sec": r["total_training_time_sec"],
        })

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "experiments": summary_data,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    logger.info(f"JSON results saved: {json_path}")

    md_lines = [
        "# Ablation Experiment Results",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Experiment Summary",
        "",
        "| Experiment | Backbone | CBAM | BiLSTM | CTC Weight | CE Weight | Val Accuracy (%) | Char Accuracy (%) | Epochs | Time (min) |",
        "|------------|----------|------|--------|------------|-----------|------------------|-------------------|--------|------------|",
    ]

    for s in summary_data:
        md_lines.append(
            f"| {s['experiment_name']} | {s['backbone']} | {s['use_cbam']} | "
            f"{s['use_bilstm']} | {s['ctc_weight']} | {s['ce_weight']} | "
            f"{s['best_val_accuracy'] * 100:.2f} | "
            f"{s['best_val_char_accuracy'] * 100:.2f} | "
            f"{s['total_epochs_trained']} | "
            f"{s['total_training_time_sec'] / 60:.1f} |"
        )

    md_lines.append("")
    md_lines.append("## Training Curves")
    md_lines.append("")
    md_lines.append("![Training Curves](figures/ablation_comparison.png)")
    md_lines.append("")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    logger.info(f"Markdown report saved: {md_path}")

    if HAS_MATPLOTLIB and summary_data:
        try:
            names = [s["experiment_name"] for s in summary_data]
            accuracies = [s["best_val_accuracy"] * 100 for s in summary_data]
            char_accuracies = [s["best_val_char_accuracy"] * 100 for s in summary_data]

            fig, ax = plt.subplots(figsize=(12, 6))

            x = np.arange(len(names))
            width = 0.35

            bars1 = ax.bar(x - width / 2, accuracies, width, label="Image Accuracy (%)", color="#4C72B0")
            bars2 = ax.bar(x + width / 2, char_accuracies, width, label="Char Accuracy (%)", color="#55A868")

            ax.set_xlabel("Experiment", fontsize=12)
            ax.set_ylabel("Accuracy (%)", fontsize=12)
            ax.set_title("Ablation Experiment Results", fontsize=14, fontweight="bold")
            ax.set_xticks(x)
            ax.set_xticklabels(names, rotation=15, ha="right", fontsize=10)
            ax.legend(fontsize=11)
            ax.set_ylim(0, 105)

            for bar in bars1:
                height = bar.get_height()
                ax.annotate(
                    f"{height:.1f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )
            for bar in bars2:
                height = bar.get_height()
                ax.annotate(
                    f"{height:.1f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

            plt.tight_layout()
            plt.savefig(chart_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            logger.info(f"Bar chart saved: {chart_path}")
        except Exception as e:
            logger.exception(f"Failed to generate chart: {e}")


def main():
    parser = argparse.ArgumentParser(description="Captcha Recognition Ablation Experiments")
    parser.add_argument(
        "--experiments",
        type=str,
        default="all",
        help="Comma-separated experiment names, or 'all' for all 5 experiments",
    )
    parser.add_argument(
        "--skip_training",
        action="store_true",
        help="Skip training and only generate results from existing JSON",
    )
    args = parser.parse_args()

    config = get_config()
    project_root = config.get_project_root()

    dir_manager = DirectoryManager(verbose=False)
    dir_manager.ensure_directory(project_root / "checkpoints")
    dir_manager.ensure_directory(project_root / "results")
    dir_manager.ensure_directory(project_root / "results" / "figures")

    log_dir = project_root / "logs"
    dir_manager.ensure_directory(log_dir)
    logger = setup_logging(str(log_dir), log_filename="ablation.log")

    logger.info("=" * 80)
    logger.info("Captcha Recognition - Ablation Experiment Framework")
    logger.info("=" * 80)

    set_seed(config.get_seed())

    device_manager = DeviceManager(verbose=False)
    device = device_manager.device
    logger.info(f"Device: {device}")

    ablation_config = {
        "epochs": config.get("ablation.num_epochs", 15),
        "batch_size": config.get("ablation.batch_size", 64),
        "num_workers": config.get("ablation.num_workers", 8),
        "learning_rate": float(config.get("ablation.learning_rate", 2e-4)),
        "max_lr": float(config.get("ablation.max_lr", 1e-3)),
        "weight_decay": float(config.get("ablation.weight_decay", 5e-5)),
        "warmup_epochs": config.get("ablation.warmup_epochs", 2),
    }
    logger.info(
        f"Ablation hyperparameters: epochs={ablation_config['epochs']}, "
        f"batch_size={ablation_config['batch_size']}, lr={ablation_config['learning_rate']}"
    )

    if args.experiments.lower() == "all":
        selected_experiments = list(EXPERIMENT_CONFIGS.keys())
    else:
        selected_experiments = [e.strip() for e in args.experiments.split(",")]

    for exp_name in selected_experiments:
        if exp_name not in EXPERIMENT_CONFIGS:
            logger.error(f"Unknown experiment: {exp_name}")
            logger.error(f"Available experiments: {', '.join(EXPERIMENT_CONFIGS.keys())}")
            sys.exit(1)

    logger.info(f"Selected experiments: {', '.join(selected_experiments)}")

    if args.skip_training:
        json_path = project_root / "results" / "ablation_results.json"
        if json_path.exists():
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            results = data.get("experiments", [])
            logger.info(f"Loaded existing results from {json_path} ({len(results)} experiments)")
            generate_results_table(results, project_root, logger)
            return
        else:
            logger.error(f"No existing results found at {json_path}")
            return

    val_dir = str(project_root / "data" / "preprocessed" / "val")

    logger.info(f"Loading validation dataset from: {val_dir}")
    val_dataset = CaptchaDataset(
        data_dir=val_dir,
        transform=get_val_transforms(),
        max_length=config.get_max_length(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=ablation_config["batch_size"],
        shuffle=False,
        num_workers=ablation_config["num_workers"],
        pin_memory=device.type == "cuda",
        collate_fn=collate_fn,
        persistent_workers=ablation_config["num_workers"] > 0,
        prefetch_factor=2,
    )
    logger.info(f"Validation set: {len(val_dataset)} samples")

    results = []

    for exp_idx, exp_name in enumerate(selected_experiments):
        exp_config = EXPERIMENT_CONFIGS[exp_name]

        logger.info(f"\n{'=' * 80}")
        logger.info(f"Preparing dataset for experiment {exp_idx + 1}/{len(selected_experiments)}: {exp_name}")
        logger.info(f"{'=' * 80}")

        train_dataset = StreamCaptchaDataset(
            transform=get_train_transforms(),
            max_length=config.get_max_length(),
            num_samples_per_epoch=100000,
            seed=config.get_seed() + exp_idx,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=ablation_config["batch_size"],
            shuffle=True,
            num_workers=ablation_config["num_workers"],
            pin_memory=device.type == "cuda",
            drop_last=True,
            collate_fn=collate_fn,
            persistent_workers=ablation_config["num_workers"] > 0,
            prefetch_factor=2,
        )
        logger.info(f"Streaming dataset: 100K samples/epoch for '{exp_name}'")

        try:
            result = run_single_experiment(
                experiment_name=exp_name,
                experiment_config=exp_config,
                ablation_config=ablation_config,
                device=device,
                train_loader=train_loader,
                val_loader=val_loader,
                logger=logger,
                project_root=project_root,
            )
            results.append(result)

        except Exception as e:
            logger.exception(f"Experiment '{exp_name}' failed: {e}")
            results.append({
                "experiment_name": exp_name,
                "description": exp_config["description"],
                "error": str(e),
                "best_val_accuracy": 0.0,
                "best_val_char_accuracy": 0.0,
                "best_epoch": 0,
                "total_epochs_trained": 0,
                "total_training_time_sec": 0.0,
                "train_history": [],
                "val_history": [],
                "backbone": exp_config["backbone"],
                "use_cbam": exp_config["use_cbam"],
                "use_bilstm": exp_config["use_bilstm"],
                "ctc_weight": exp_config["ctc_weight"],
                "ce_weight": exp_config["ce_weight"],
                "checkpoint_path": "",
            })

        del train_dataset
        del train_loader
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    generate_results_table(results, project_root, logger)

    logger.info("\n" + "=" * 80)
    logger.info("Ablation Experiment Summary")
    logger.info("=" * 80)

    for r in results:
        acc = r.get("best_val_accuracy", 0.0) * 100
        logger.info(f"  {r['experiment_name']:20s} -> Val Accuracy: {acc:6.2f}%")

    logger.info("=" * 80)
    logger.info("All ablation experiments completed successfully!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
