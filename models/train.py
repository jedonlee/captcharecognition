# -*- coding: utf-8 -*-
"""
CAPTCHA Recognition Model Training Script - ConvNeXt-Tiny + BiLSTM + CTC

Training Flow:
+------------------------------------------------------------------+
| 1. Initialization                                                 |
|    - Load config (config.yaml)                                    |
|    - Set random seed (reproducibility)                            |
|    - Initialize device (GPU/CPU)                                  |
|    - Create model (ConvNeXt-Tiny + BiLSTM + CTC)                  |
|    - Load dataset (train, validation)                             |
|    - Create data loader (multi-process)                           |
|    - Define loss function (Hybrid CTC/Attention Loss)             |
|    - Define optimizer (AdamW)                                     |
|    - Define learning rate scheduler (OneCycleLR)                  |
+------------------------------------------------------------------+
                            |
+------------------------------------------------------------------+
| 2. Training Phase (per epoch)                                     |
|    - Set model to train mode                                      |
|    - Iterate over training data loader                            |
|    - Forward pass: compute model output                           |
|    - Compute hybrid loss (CTC Loss + CE Loss)                     |
|    - Backward pass: compute gradients                             |
|    - Gradient clipping: prevent gradient explosion                |
|    - Update parameters: optimizer step                            |
|    - Update learning rate: scheduler step                         |
|    - Record training metrics (loss, accuracy, etc.)               |
+------------------------------------------------------------------+
                            |
+------------------------------------------------------------------+
| 3. Validation Phase (after each epoch)                            |
|    - Set model to eval mode                                       |
|    - Iterate over validation data loader                          |
|    - Forward pass: compute model output                           |
|    - Compute validation metrics (loss, accuracy, etc.)            |
|    - Record validation metrics to TensorBoard                     |
|    - Save best model weights                                      |
+------------------------------------------------------------------+
"""

import os
import time
import gc
import logging
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from tqdm import tqdm
import numpy as np

try:
    from tensorboardX import SummaryWriter
except ImportError:
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        SummaryWriter = None

from utils.config_loader import ConfigLoader, config_loader
from utils.device_manager import DeviceManager
from utils.directory_manager import DirectoryManager
from utils.chars import get_all_chars, char_to_index, index_to_char, NUM_CHARS, get_mapper
from utils.decoder import ctc_decode, decode_predictions, beam_search_decode, normalize_equiv
from utils.language_model import CharNGramLM
from utils.metrics import calculate_accuracy
from models.model import CaptchaModel
from models.dataset import CaptchaDataset, StreamCaptchaDataset
from models.hybrid_loss_fixed import HybridCTCELoss
from models.transforms import get_train_transforms, get_val_transforms

logger = logging.getLogger(__name__)

from utils.training_utils import (
    MemoryMonitor,
    setup_logging,
    verify_tensorboard_logs,
    set_seed,
    save_checkpoint,
    load_checkpoint,
    auto_adjust_batch_size,
    calculate_onecycle_params,
    calculate_dynamic_patience,
    ModelEMA,
)
from utils.training_recorder import TrainingRecorder


def train_one_epoch(model, dataloader, criterion, optimizer, device, writer, epoch,
                    scaler=None, scheduler=None, grad_clip_enabled=True,
                    grad_clip_max_norm=5.0, memory_monitor=None, logger=None,
                    gradient_accumulation_steps=1, ema_model=None, mixup_alpha=0.0):
    """
    Train for one epoch (memory-optimized version)

    Returns:
        tuple: (average loss, average CTC loss, average CE loss)
    """
    model.train()

    total_loss = 0.0
    total_ctc_loss = 0.0
    total_ce_loss = 0.0
    total_batches = 0

    # Moving average loss for progress bar
    moving_avg_loss = 0.0
    alpha = 0.1  # Smoothing factor

    use_mixup = mixup_alpha > 0.0

    device_manager = DeviceManager()

    use_amp = device_manager.supports_amp() and scaler is not None
    if use_amp:
        from torch.amp import autocast
        amp_context = autocast('cuda')
    else:
        from contextlib import nullcontext
        amp_context = nullcontext()

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")

    try:
        for batch_idx, batch in enumerate(progress_bar):
            try:
                if not batch or 'images' not in batch:
                    if logger:
                        logger.warning(f"Invalid batch at {batch_idx}")
                    continue

                if batch['images'].numel() == 0:
                    if logger:
                        logger.warning(f"Empty images in batch {batch_idx}")
                    continue

                # Use non_blocking only on CUDA for performance
                non_blocking = device.type == 'cuda'
                images = batch['images'].to(device, non_blocking=non_blocking)
                label_indices = batch['label_indices'].to(device, non_blocking=non_blocking)
                label_lengths = batch['label_lengths'].to(device, non_blocking=non_blocking)

                # Prepare CTC targets: concatenate all labels into 1D sequence
                targets_list = []
                for i in range(len(label_lengths)):
                    targets_list.append(label_indices[i][:label_lengths[i]])
                targets = torch.cat(targets_list)
                targets_lengths = label_lengths  # (B,)
                input_lengths = None  # Will be set after forward

                # MixUp augmentation
                mixup_lam = 1.0
                if use_mixup:
                    mixup_lam = np.random.beta(mixup_alpha, mixup_alpha)
                    perm_index = torch.randperm(images.size(0)).to(device)
                    images = mixup_lam * images + (1 - mixup_lam) * images[perm_index]
                    # Prepare permuted targets for mixup labels
                    targets_perm_list = []
                    for i in range(len(label_lengths)):
                        targets_perm_list.append(label_indices[perm_index[i]][:label_lengths[perm_index[i]]])
                    targets_perm = torch.cat(targets_perm_list)

                # Use fixed loss weights from config.yaml, disable dynamic weighting
                if batch_idx == 0 and logger:
                    logger.info(f"Fixed loss weights: ctc={criterion.ctc_weight:.1f} ce={criterion.ce_weight:.1f} [config specified]")

                # Forward pass (mixed precision)
                with amp_context:
                    encoder_out, decoder_out = model(images)

                    batch_size = images.size(0)
                    input_lengths = torch.full((batch_size,), encoder_out.size(0),
                                              dtype=torch.long).to(device)

                    # Check if output is valid
                    if torch.isnan(encoder_out).any() or torch.isinf(encoder_out).any():
                        if logger:
                            logger.warning(f"Epoch {epoch}, Batch {batch_idx}: encoder_out contains NaN/Inf, skipping")
                        del encoder_out, decoder_out, images, label_indices, label_lengths
                        del targets, targets_lengths
                        continue

                    if torch.isnan(decoder_out).any() or torch.isinf(decoder_out).any():
                        if logger:
                            logger.warning(f"Epoch {epoch}, Batch {batch_idx}: decoder_out contains NaN/Inf, skipping")
                        del encoder_out, decoder_out, images, label_indices, label_lengths
                        continue

                    # Compute hybrid loss
                    loss, ctc_loss, ce_loss = criterion(
                        encoder_out, decoder_out, targets, input_lengths, targets_lengths
                    )

                    # MixUp: compute second loss for permuted labels and combine
                    if use_mixup:
                        loss_perm, ctc_loss_perm, ce_loss_perm = criterion(
                            encoder_out, decoder_out, targets_perm, input_lengths, label_lengths[perm_index]
                        )
                        loss = mixup_lam * loss + (1 - mixup_lam) * loss_perm
                        ctc_loss = mixup_lam * ctc_loss + (1 - mixup_lam) * ctc_loss_perm
                        ce_loss = mixup_lam * ce_loss + (1 - mixup_lam) * ce_loss_perm

                # Check if loss is NaN/Inf
                if torch.isnan(loss) or torch.isinf(loss):
                    if logger:
                        logger.warning(f"Loss is NaN/Inf at batch {batch_idx}")
                    del encoder_out, decoder_out, images, label_indices, label_lengths
                    del targets, targets_lengths, loss, ctc_loss, ce_loss
                    continue

                # Backward pass
                if use_amp and scaler is not None:
                    # Mixed precision training
                    scaler.scale(loss).backward()

                    # Gradient clipping (must come after unscale_)
                    scaler.unscale_(optimizer)
                    if grad_clip_enabled:
                        torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                     max_norm=float(grad_clip_max_norm))

                    # Update parameters
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                else:
                    # Standard training (CPU mode or GPU without AMP)
                    loss.backward()

                    # Gradient clipping
                    if grad_clip_enabled:
                        torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                     max_norm=float(grad_clip_max_norm))

                    # Update parameters
                    optimizer.step()
                    optimizer.zero_grad()

                # EMA weight update (every step)
                if ema_model is not None:
                    ema_model.update(model)

                # Learning rate scheduler (called every batch)
                if scheduler is not None:
                    scheduler.step()

                total_loss += loss.item()
                total_ctc_loss += ctc_loss.item()
                total_ce_loss += ce_loss.item()
                total_batches += 1

                moving_avg_loss = alpha * loss.item() + (1 - alpha) * moving_avg_loss \
                    if total_batches > 0 else loss.item()

                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'avg_loss': f'{moving_avg_loss:.4f}'
                })

                global_step = epoch * len(dataloader) + batch_idx

                # TensorBoard logging (reduce frequency to lower I/O overhead)
                if batch_idx % 5 == 0 and writer:
                    writer.add_scalar('Loss/train_batch', loss.item(), global_step)
                    writer.add_scalar('Loss/train_ctc_batch', ctc_loss.item(), global_step)
                    writer.add_scalar('Loss/train_ce_batch', ce_loss.item(), global_step)

                if memory_monitor is not None and batch_idx % 50 == 0:
                    mem_info = memory_monitor.update()
                    if writer:
                        writer.add_scalar('Memory/GPU_Allocated_GB',
                                        mem_info['gpu']['allocated_gb'], global_step)
                        writer.add_scalar('Memory/GPU_Reserved_GB',
                                        mem_info['gpu']['reserved_gb'], global_step)
                        writer.add_scalar('Memory/CPU_Used_GB',
                                        mem_info['cpu']['used_gb'], global_step)

                # Clean up intermediate tensors
                del loss, ctc_loss, ce_loss, encoder_out, decoder_out
                del images, label_indices, label_lengths, targets, targets_lengths, input_lengths

            except RuntimeError as e:
                error_msg = f"RuntimeError in batch {batch_idx}: {str(e)}"
                if logger:
                    logger.exception(error_msg)
                continue

            except Exception as e:
                error_msg = f"Error in batch {batch_idx}: {str(e)}"
                if logger:
                    logger.exception(error_msg)
                continue

    except Exception as e:
        error_msg = f"Error in epoch {epoch}: {str(e)}"
        if logger:
            logger.exception(error_msg)
        # Do not raise exception, return current loss values

    finally:
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()

    if total_batches > 0:
        avg_loss = total_loss / total_batches
        avg_ctc_loss = total_ctc_loss / total_batches
        avg_ce_loss = total_ce_loss / total_batches
    else:
        if logger:
            logger.warning(f"No valid batches processed in epoch {epoch}")
        avg_loss, avg_ctc_loss, avg_ce_loss = 0.0, 0.0, 0.0

    if writer:
        writer.add_scalar('Loss/train_epoch', avg_loss, epoch)
        writer.add_scalar('Loss/train_ctc_epoch', avg_ctc_loss, epoch)
        writer.add_scalar('Loss/train_ce_epoch', avg_ce_loss, epoch)

    return avg_loss, avg_ctc_loss, avg_ce_loss


def validate(model, dataloader, criterion, device, writer, epoch, chars=None,
             blank_index=None, memory_monitor=None, logger=None,
             lm_model=None, lm_weight=0.0):
    """
    Validation function (memory-optimized version)

    Returns:
        tuple: (average loss, average CTC loss, average CE loss, image accuracy, char accuracy)
    """
    if blank_index is None:
        config_loader = ConfigLoader()
        blank_index = config_loader.get_blank_idx()

    if chars is None:
        chars = get_all_chars()

    model.eval()

    total_loss = 0.0
    total_ctc_loss = 0.0
    total_ce_loss = 0.0
    total_batches = 0
    total_images = 0
    correct_images = 0
    total_chars = 0
    correct_chars = 0

    # Moving average loss
    moving_avg_loss = 0.0
    alpha = 0.1

    # Collect predictions for statistics
    all_predictions = []
    all_targets = []

    try:
        with torch.no_grad():
            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch} [Val]")

            for batch_idx, batch in enumerate(progress_bar):
                try:
                    images = batch['images'].to(device)
                    label_indices = batch['label_indices'].to(device)
                    label_lengths = batch['label_lengths'].to(device)

                    encoder_out, decoder_out = model(images)

                    batch_size = images.size(0)
                    input_lengths = torch.full((batch_size,), encoder_out.size(0),
                                              dtype=torch.long).to(device)

                    targets_list = []
                    for i in range(len(label_lengths)):
                        targets_list.append(label_indices[i][:label_lengths[i]])
                    targets = torch.cat(targets_list)
                    targets_lengths = label_lengths

                    loss, ctc_loss, ce_loss = criterion(
                        encoder_out, decoder_out, targets, input_lengths, targets_lengths
                    )

                    total_loss += loss.item()
                    total_ctc_loss += ctc_loss.item()
                    total_ce_loss += ce_loss.item()
                    total_batches += 1

                    moving_avg_loss = alpha * loss.item() + (1 - alpha) * moving_avg_loss \
                        if total_batches > 0 else loss.item()

                    progress_bar.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        'avg_loss': f'{moving_avg_loss:.4f}'
                    })

                    # Decode predictions
                    try:
                        # Decode using Beam Search (beam width=10)
                        mapper = get_mapper()
                        pred_strings = beam_search_decode(encoder_out, mapper, beam_width=15, enable_corrector=False,
                                                                   lm_model=lm_model, lm_weight=lm_weight)
                        
                        # Apply post-processing (zero-training-cost boost of 5%-10%)
                        from utils.decoder import postprocess_text_list
                        pred_strings = postprocess_text_list(pred_strings)

                        # Decode ground truth labels
                        target_strings = []
                        for i in range(batch_size):
                            target_len = label_lengths[i].item()
                            target_indices = label_indices[i, :target_len].tolist()
                            target_str = ''.join([chars[idx] for idx in target_indices])
                            target_strings.append(target_str)

                        # Compute accuracy with confusion-tolerant equivalence
                        for pred_str, target_str in zip(pred_strings, target_strings):
                            pred_norm = normalize_equiv(pred_str)
                            target_norm = normalize_equiv(target_str)
                            all_predictions.append(pred_norm)
                            all_targets.append(target_norm)
                            total_images += 1

                            if pred_norm == target_norm:
                                correct_images += 1

                            for p_char, t_char in zip(pred_norm, target_norm):
                                total_chars += 1
                                if p_char == t_char:
                                    correct_chars += 1

                    except Exception as decode_error:
                        if logger:
                            logger.warning(f"Decode error at batch {batch_idx}: {decode_error}")

                    # Memory monitoring
                    if memory_monitor is not None and batch_idx % 50 == 0:
                        mem_info = memory_monitor.update()

                    # Clean up variables
                    del loss, ctc_loss, ce_loss, encoder_out, decoder_out
                    del images, label_indices, label_lengths, targets, targets_lengths, input_lengths

                except Exception as e:
                    error_msg = f"Validation error at batch {batch_idx}: {str(e)}"
                    if logger:
                        logger.error(error_msg)
                    continue

    except Exception as e:
        error_msg = f"Validation error: {str(e)}"
        if logger:
            logger.error(error_msg)

    finally:
        del all_predictions, all_targets
        gc.collect()

    if total_batches > 0:
        avg_loss = total_loss / total_batches
        avg_ctc_loss = total_ctc_loss / total_batches
        avg_ce_loss = total_ce_loss / total_batches
    else:
        if logger:
            logger.warning("No valid predictions in validation")
        avg_loss, avg_ctc_loss, avg_ce_loss = 0.0, 0.0, 0.0

    image_accuracy = correct_images / total_images if total_images > 0 else 0.0
    char_accuracy = correct_chars / total_chars if total_chars > 0 else 0.0

    # Log to TensorBoard
    if writer:
        writer.add_scalar('Loss/val_epoch', avg_loss, epoch)
        writer.add_scalar('Loss/val_ctc_epoch', avg_ctc_loss, epoch)
        writer.add_scalar('Loss/val_ce_epoch', avg_ce_loss, epoch)
        writer.add_scalar('Accuracy/val_image', image_accuracy, epoch)
        writer.add_scalar('Accuracy/val_char', char_accuracy, epoch)

    return avg_loss, avg_ctc_loss, avg_ce_loss, image_accuracy, char_accuracy


_DEPRECATED_CONFIG = object()


def train(config=_DEPRECATED_CONFIG, resume_from=None):
    """
    Main training function

    Args:
        config: deprecated (backward compat), config is read from config_loader singleton
        resume_from: checkpoint path to resume from (optional)
    """
    import os

    dir_manager = DirectoryManager(verbose=True)

    # Set up logging
    log_dir = config_loader.get('training.log_dir', 'logs')
    dir_manager.ensure_directory(log_dir)
    logger = setup_logging(log_dir, level=logging.INFO)

    logger.info("=" * 80)
    logger.info("Starting training process")
    logger.info("=" * 80)

    seed = config_loader.get('training.seed', 42)
    set_seed(seed)
    logger.info(f"Random seed set: {seed}")

    # Initialize device
    device_manager = DeviceManager()
    device = device_manager.device
    logger.info(f"Using device: {device}")

    # Memory monitoring
    enable_memory_monitor = config_loader.get('training', {}).get('enable_memory_monitor', True)
    memory_monitor = None
    if enable_memory_monitor:
        memory_monitor = MemoryMonitor(device, logger=logger)
        logger.info("Memory monitoring enabled")

    # Create model
    logger.info("Creating model...")
    model_config = config_loader.get('model', {})
    arch_cfg = model_config.get('architecture', {})
    model = CaptchaModel(
        pretrained=arch_cfg.get('pretrained', False)
    )
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {total_params:,} (trainable: {trainable_params:,})")

    # Load dataset
    logger.info("Loading dataset...")
    
    # Check if streaming dataset is enabled
    use_streaming = config_loader.get('streaming_dataset', {}).get('enabled', False)
    
    # Safely get path config with reasonable defaults
    train_dir = str(config_loader.get_project_root() / config_loader.get('data.preprocessed_dir', 'preprocessed') / 'train')
    val_dir = str(config_loader.get_project_root() / config_loader.get('data.preprocessed_dir', 'preprocessed') / 'val')
    max_length = 6  # Fixed max length
    
    # Log path config for debugging
    logger.info(f"Dataset mode: {'streaming generation' if use_streaming else 'fixed files'}")
    logger.info(f"Training set path: {train_dir}")
    logger.info(f"Validation set path: {val_dir}")

    if use_streaming:
        streaming_config = config_loader.get('streaming_dataset', {})
        num_samples_per_epoch = streaming_config.get('num_samples_per_epoch', 100000)
        
        logger.info(f"✅ Using streaming generation dataset, {num_samples_per_epoch} samples per epoch")
        
        train_dataset = StreamCaptchaDataset(
            transform=get_train_transforms(),
            max_length=max_length,
            num_samples_per_epoch=num_samples_per_epoch,
            seed=config_loader.get('streaming_dataset', {}).get('seed', 42)
        )
    else:
        train_dataset = CaptchaDataset(
            data_dir=train_dir,
            transform=get_train_transforms(),
            max_length=max_length
        )

    val_dataset = CaptchaDataset(
        data_dir=val_dir,
        transform=get_val_transforms(),
        max_length=max_length
    )

    # Create data loaders
    batch_size = config_loader.get('batch_size', 256)  # Backward compat: prefer training.batch_size
    batch_size = config_loader.get('training.batch_size', batch_size)
    num_workers = config_loader.get('training.num_workers', 4)

    logger.info(f"Using batch_size: {batch_size}")

    from models.dataset import collate_fn
    
    pin_memory = config_loader.get('system', {}).get('pin_memory', True)
    prefetch_factor = config_loader.get('system', {}).get('prefetch_factor', 2)
    persistent_workers = config_loader.get('system', {}).get('persistent_workers', True)
    gradient_accumulation_steps = config_loader.get('gradient_accumulation_steps', 1)
    gradient_accumulation_steps = config_loader.get('training.gradient_accumulation_steps', gradient_accumulation_steps)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        collate_fn=collate_fn,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers
    )
    
    logger.info(f"DataLoader optimization: pin_memory={pin_memory}, prefetch_factor={prefetch_factor}, persistent_workers={persistent_workers}")
    logger.info(f"Gradient accumulation steps: {gradient_accumulation_steps}")

    logger.info(f"Training set size: {len(train_dataset)} batches: {len(train_loader)}")
    logger.info(f"Validation set size: {len(val_dataset)} batches: {len(val_loader)}")

    # Build n-gram language model (for decoding error correction)
    lm_model = None
    lm_weight = config_loader.get('training.lm_weight', 0.0)
    if lm_weight > 0.0 and use_streaming:
        try:
            mapper = get_mapper()
            chars = mapper.characters
            num_lm_samples = min(100000, num_samples_per_epoch)
            synthetic_texts = []
            lm_rng = np.random.RandomState(42)
            for _ in range(num_lm_samples):
                length = lm_rng.randint(1, max_length + 1)
                text = ''.join(lm_rng.choice(list(chars), size=length))
                synthetic_texts.append(text)
            lm_model = CharNGramLM(order=3, smoothing=0.01)
            lm_model.build(synthetic_texts)
            logger.info(f"✅ n-gram language model built: {num_lm_samples} samples, "
                        f"vocab_size={len(chars)}, lm_weight={lm_weight}")
        except Exception as e:
            logger.warning(f"n-gram language model construction failed: {e}")
            lm_model = None
    elif lm_weight > 0.0:
        logger.info("n-gram language model requires streaming dataset, skipped")

    # Define loss function
    ctc_weight = config_loader.get('training.ctc_weight', 0.6)
    ce_weight = config_loader.get('training.ce_weight', 0.4)
    label_smoothing = config_loader.get('training.label_smoothing', 0.1)
    criterion = HybridCTCELoss(
        num_chars=NUM_CHARS,
        ctc_weight=ctc_weight,
        ce_weight=ce_weight,
        label_smoothing=label_smoothing
    )
    criterion = criterion.to(device)
    logger.info(f"Loss function: CTC weight={criterion.ctc_weight}, CE weight={criterion.ce_weight}, label_smoothing={label_smoothing}")

    # Define optimizer - force type conversion to ensure success
    lr = config_loader.get('training.learning_rate', 0.00008)
    weight_decay = config_loader.get('training.weight_decay', 0.0001)
    try:
        lr = float(lr)
    except (ValueError, TypeError):
        lr = 0.00008  # Fallback to reasonable default

    try:
        weight_decay = float(weight_decay)
    except (ValueError, TypeError):
        weight_decay = 0.0001  # Fallback to reasonable default

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    logger.info(f"Optimizer: AdamW (lr={lr}, weight_decay={weight_decay})")
    logger.info(f"  - lr type: {type(lr)}, value: {lr}")
    logger.info(f"  - weight_decay type: {type(weight_decay)}, value: {weight_decay}")

    # Define learning rate scheduler - Linear warmup + Cosine annealing warm restarts
    num_epochs = config_loader.get('training.num_epochs', 80)
    max_lr = config_loader.get('training.max_lr', 0.0005)
    warmup_pct = config_loader.get('training.warmup_pct', 0.15)
    final_div_factor = config_loader.get('training.final_div_factor', 250)

    batch_size = config_loader.get('training.batch_size', 256)
    steps_per_epoch = (len(train_dataset) + batch_size - 1) // batch_size

    warmup_steps = int(warmup_pct * num_epochs * steps_per_epoch)
    T_0 = 30 * steps_per_epoch
    T_mult = 2
    eta_min = max_lr / 100

    warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1e-8, end_factor=1.0, total_iters=warmup_steps
    )
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=T_0, T_mult=T_mult, eta_min=eta_min
    )
    if warmup_steps > 0:
        scheduler = optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps]
        )
        logger.info(f"LR scheduler: LinearWarmup({warmup_steps} steps) → CosineAnnealingWarmRestarts (T_0={30} epochs, T_mult=2, eta_min={eta_min:.6f})")
    else:
        scheduler = cosine_scheduler
        logger.info(f"LR scheduler: CosineAnnealingWarmRestarts (T_0={30} epochs, T_mult=2, eta_min={eta_min:.6f})")

    # Mixed precision training - force enable
    use_amp = True  # Force enable AMP
    scaler = GradScaler('cuda') if use_amp else None
    logger.info(f"Mixed precision training: {'enabled' if use_amp else 'disabled'}")
    
    # Freeze Backbone training strategy
    freeze_backbone_epochs = config_loader.get('training.freeze_backbone_epochs', 0)
    if freeze_backbone_epochs > 0:
        for param in model.backbone.parameters():
            param.requires_grad = False
        logger.info(f"Two-stage training: freeze backbone for first {freeze_backbone_epochs} epochs, only train Head layer")
    else:
        logger.info("Two-stage training: freeze strategy not enabled")

    # EMA (Exponential Moving Average)
    ema_model = ModelEMA(model, decay=0.999)
    logger.info(f"EMA enabled: decay=0.999")

    mixup_alpha = config_loader.get('training.mixup_alpha', 0.0)
    if mixup_alpha > 0.0:
        logger.info(f"MixUp enabled: alpha={mixup_alpha}")
    else:
        logger.info("MixUp: not enabled")

    # TensorBoard
    tensorboard_dir = config_loader.get('logging', {}).get('tensorboard_dir', 'runs')
    dir_manager.ensure_directory(tensorboard_dir)
    writer = SummaryWriter(tensorboard_dir) if SummaryWriter else None
    if writer:
        logger.info(f"TensorBoard log directory: {tensorboard_dir}")

    # Early stopping params
    patience = calculate_dynamic_patience(num_epochs)
    best_val_accuracy = 0.0
    epochs_no_improve = 0
    logger.info(f"Early stopping patience: {patience}")

    # Gradient clipping
    grad_clip_config = config_loader.get('gradient_clipping', {})
    grad_clip_enabled = grad_clip_config.get('enabled', True)
    grad_clip_max_norm = grad_clip_config.get('max_norm', 5.0)

    checkpoint_dir = config_loader.get('training', {}).get('checkpoint_dir', 'checkpoints')
    dir_manager.ensure_directory(checkpoint_dir)

    # Resume from checkpoint (if specified)
    start_epoch = 0
    if resume_from and os.path.exists(resume_from):
        resume_info = load_checkpoint(resume_from, model, optimizer, scheduler, scaler, device)
        start_epoch = resume_info['epoch'] + 1
        best_val_accuracy = resume_info['best_val_accuracy']
        logger.info(f"Resumed from checkpoint: epoch {start_epoch}, best accuracy: {best_val_accuracy:.4f}")

        # Handle unfreeze point crossing during resume
        backbone_frozen = not any(p.requires_grad for p in model.backbone.parameters())
        if start_epoch >= freeze_backbone_epochs and backbone_frozen:
            logger.info(f"Backbone still frozen at resume (checkpoint epoch={resume_info['epoch']}), manually unfreezing")
            for param in model.backbone.parameters():
                param.requires_grad = True
            logger.info(f"Epoch {start_epoch}: 🔓 Backbone unfrozen (manual handling during resume)")

            backbone_lr_ratio = config_loader.get('training.backbone_lr_ratio', 1.0)
            backbone_lr = lr * backbone_lr_ratio
            head_lr = lr

            head_params = [p for p in model.parameters() if id(p) not in set(id(p) for p in model.backbone.parameters())]
            optimizer = optim.AdamW([
                {'params': model.backbone.parameters(), 'lr': backbone_lr},
                {'params': head_params, 'lr': head_lr},
            ], weight_decay=config_loader.get('training.weight_decay', 0.0001))

            T_0 = 30 * steps_per_epoch
            T_mult = 2
            eta_min = max_lr / 100

            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=T_0, T_mult=T_mult, eta_min=eta_min
            )
            logger.info(f"Differential learning rate (manual setting during resume): Backbone={backbone_lr:.6f}, Head={head_lr:.6f}")
            logger.info(f"LR scheduler rebuilt (manual setting during resume): CosineAnnealingWarmRestarts")
    else:
        # Fully load the previously trained best_model.pth (includes all params)
        best_model_path = str(config_loader.get_project_root() / config_loader.get('checkpoint.checkpoint_dir', 'checkpoints') / 'best_model.pth')
        if os.path.exists(best_model_path):
            try:
                checkpoint = torch.load(best_model_path, map_location=device, weights_only=True)
                model.load_state_dict(checkpoint['model_state_dict'], strict=True)
                if 'optimizer_state_dict' in checkpoint:
                    try:
                        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    except:
                        pass
                if 'scheduler_state_dict' in checkpoint:
                    try:
                        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                    except:
                        pass
                if 'best_val_accuracy' in checkpoint:
                    best_val_accuracy = checkpoint['best_val_accuracy']
                if 'epoch' in checkpoint:
                    start_epoch = checkpoint['epoch'] + 1
                logger.info(f"✅ Successfully loaded previously trained model weights! Continuing from epoch {start_epoch}, previous best accuracy: {best_val_accuracy:.4f}")
            except Exception as e:
                logger.warning(f"Error loading complete model: {e}")
                logger.info("Will only load backbone part, note FC layer may have been partially loaded")
                try:
                    # Fallback: load backbone only
                    checkpoint = torch.load(best_model_path, map_location=device, weights_only=True)
                    backbone_state_dict = {k.replace('backbone.', ''): v for k, v in checkpoint['model_state_dict'].items() if 'backbone.' in k}
                    model.backbone.load_state_dict(backbone_state_dict, strict=False)
                    logger.info("✅ Fallback: Successfully loaded previously trained ConvNeXt-Tiny backbone weights!")
                except Exception as e2:
                    logger.warning(f"Fallback also failed: {e2}")
                    logger.info("Will train from scratch")

    # Character set
    chars = get_all_chars()

    # Initialize training recorder
    recorder = TrainingRecorder(
        experiment_name=f"captcha_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        output_dir='experiments/records'
    )
    recorder.set_config(config_loader.get_config())
    logger.info("✅ Training recorder initialized")

    # Start training loop
    logger.info("=" * 80)
    logger.info(f"Starting training, total {num_epochs} epochs")
    logger.info(f"✅ Using Beam Search decoding, beam width=10")
    logger.info("=" * 80)

    training_start_time = time.time()

    for epoch in range(start_epoch, num_epochs):
        # Freeze/unfreeze backbone logic
        if freeze_backbone_epochs > 0:
            if epoch < freeze_backbone_epochs:
                # Freeze phase: check if already frozen
                if any(p.requires_grad for p in model.backbone.parameters()):
                    for param in model.backbone.parameters():
                        param.requires_grad = False
                if epoch == 0:
                    logger.info(f"Epoch {epoch+1}: Backbone frozen (only training Head layer)")
            elif epoch == freeze_backbone_epochs:
                # Unfreeze phase: unfreeze backbone for full fine-tuning
                for param in model.backbone.parameters():
                    param.requires_grad = True
                logger.info(f"Epoch {epoch+1}: 🔓 Unfreezing Backbone, starting full fine-tuning!")
                
                # Differential learning rate: Backbone (pretrained) uses lower LR, Head (new) uses base LR
                backbone_lr_ratio = config_loader.get('training.backbone_lr_ratio', 1.0)
                backbone_lr = lr * backbone_lr_ratio
                head_lr = lr
                
                # Separate Backbone and Head parameter groups
                backbone_ids = set(id(p) for p in model.backbone.parameters())
                head_params = [p for p in model.parameters() if id(p) not in backbone_ids]
                
                optimizer = optim.AdamW([
                    {'params': model.backbone.parameters(), 'lr': backbone_lr},
                    {'params': head_params, 'lr': head_lr},
                ], weight_decay=weight_decay)
                
                # Rebuild CosineAnnealingWarmRestarts scheduler for remaining epochs, starting from step 0
                T_0_remaining = 30 * steps_per_epoch
                T_mult_remaining = 2
                eta_min_unified = max_lr / 100

                scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    optimizer,
                    T_0=T_0_remaining,
                    T_mult=T_mult_remaining,
                    eta_min=eta_min_unified
                )
                logger.info(f"Differential learning rate: Backbone={backbone_lr:.6f}, Head={head_lr:.6f}")
                logger.info(f"LR scheduler rebuilt: CosineAnnealingWarmRestarts "
                           f"(T_0={30} epochs, T_mult=2, eta_min={eta_min_unified:.6f})")
        
        epoch_start_time = time.time()
        logger.info(f"\n{'='*40} Epoch {epoch+1}/{num_epochs} {'='*40}")

        # Train
        train_loss, train_ctc_loss, train_ce_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            writer=writer,
            epoch=epoch,
            scaler=scaler,
            scheduler=scheduler,
            grad_clip_enabled=grad_clip_enabled,
            grad_clip_max_norm=grad_clip_max_norm,
            memory_monitor=memory_monitor,
            logger=logger,
            ema_model=ema_model,
            mixup_alpha=mixup_alpha,
        )

        # Validation (every epoch)
        if True:
            # Switch to EMA weights before validation
            ema_model.swap(model)
            val_loss, val_ctc_loss, val_ce_loss, val_image_acc, val_char_acc = validate(
                model=model,
                dataloader=val_loader,
                criterion=criterion,
                device=device,
                writer=writer,
                epoch=epoch,
                chars=chars,
                memory_monitor=memory_monitor,
                logger=logger,
                lm_model=lm_model,
                lm_weight=lm_weight
            )
            # Restore original weights after validation
            ema_model.restore(model)
        else:
            # Skip validation (retained but unused)
            val_loss, val_ctc_loss, val_ce_loss, val_image_acc, val_char_acc = 0.0, 0.0, 0.0, 0.0, 0.0

        epoch_time = time.time() - epoch_start_time

        logger.info(f"\nEpoch {epoch+1} Summary:")
        logger.info(f"  Training Loss: {train_loss:.4f} (CTC: {train_ctc_loss:.4f}, CE: {train_ce_loss:.4f})")
        if (epoch + 1) % 2 == 0:
            logger.info(f"  Validation Loss: {val_loss:.4f} (CTC: {val_ctc_loss:.4f}, CE: {val_ce_loss:.4f})")
            logger.info(f"  Image Accuracy: {val_image_acc:.4f} ({val_image_acc*100:.2f}%)")
        else:
            logger.info(f"  Validation: skipped (odd epoch)")
        logger.info(f"  Char Accuracy: {val_char_acc:.4f} ({val_char_acc*100:.2f}%)")
        logger.info(f"  Time: {epoch_time:.2f}s")
        
        # Record training data
        current_lr = optimizer.param_groups[0]['lr']
        recorder.record_epoch(
            epoch=epoch,
            train_loss=train_loss,
            train_ctc_loss=train_ctc_loss,
            train_ce_loss=train_ce_loss,
            val_loss=val_loss,
            val_ctc_loss=val_ctc_loss,
            val_ce_loss=val_ce_loss,
            val_image_acc=val_image_acc,
            val_char_acc=val_char_acc,
            learning_rate=current_lr
        )

        if memory_monitor and (epoch + 1) % 10 == 0:
            memory_monitor.log_memory_summary(epoch=epoch+1)

        # Save best model
        if val_image_acc > best_val_accuracy:
            best_val_accuracy = val_image_acc
            epochs_no_improve = 0

            # Swap EMA weights into model for persistence
            ema_model.swap(model)
            save_path = save_checkpoint(
                save_dir=checkpoint_dir,
                filename='best_model.pth',
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                best_val_accuracy=best_val_accuracy,
                train_loss=train_loss,
                val_loss=val_loss,
                val_image_accuracy=val_image_acc,
                val_char_accuracy=val_char_acc,
                config=config_loader.get_config(),
                dir_manager=dir_manager,
                logger=logger
            )
            ema_model.restore(model)
            logger.info(f"✅ New best model (EMA weights)! Image accuracy: {best_val_accuracy:.4f}")
        else:
            epochs_no_improve += 1
            logger.info(f"Current best accuracy: {best_val_accuracy:.4f} ({epochs_no_improve}/{patience} epochs without improvement)")

        # Periodically save checkpoint (snapshot of EMA weights)
        if (epoch + 1) % 5 == 0:
            ema_model.swap(model)
            save_checkpoint(
                save_dir=checkpoint_dir,
                filename=f'checkpoint_epoch_{epoch+1}.pth',
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                best_val_accuracy=best_val_accuracy,
                train_loss=train_loss,
                val_loss=val_loss,
                val_image_accuracy=val_image_acc,
                val_char_accuracy=val_char_acc,
                config=config_loader.get_config(),
                dir_manager=dir_manager,
                logger=logger
            )
            ema_model.restore(model)

        # Early stopping check
        if epochs_no_improve >= patience:
            logger.info(f"\nEarly stopping triggered! {patience} consecutive epochs without improvement")
            logger.info(f"Best validation accuracy: {best_val_accuracy:.4f}")
            break

    # Training complete
    total_training_time = time.time() - training_start_time
    logger.info("\n" + "=" * 80)
    logger.info("Training completed!")
    logger.info("=" * 80)
    logger.info(f"Total training time: {total_training_time/3600:.2f} hours")
    logger.info(f"Best validation image accuracy: {best_val_accuracy:.4f} ({best_val_accuracy*100:.2f}%)")
    
    # Print training summary
    recorder.print_summary()

    # Final memory summary
    if memory_monitor:
        memory_monitor.log_memory_summary()

    # Verify TensorBoard logs
    if writer:
        writer.close()
        verify_tensorboard_logs(tensorboard_dir, logger=logger)


def main():
    """Main function"""
    import argparse
    import sys
    import subprocess
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='CAPTCHA Recognition Model Training Program')
    parser.add_argument('--resume', type=str, default=None, help='Resume training from specified checkpoint')
    parser.add_argument('--streaming', action='store_true', help='Use streaming generation dataset')
    parser.add_argument('--num_epochs', type=int, default=None, help='Number of training epochs (override config)')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size (override config)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (for ensemble training different models)')
    parser.add_argument('--force', action='store_true', help='Force start, ignore duplicate training check')
    parser.add_argument('--save_as', type=str, default=None, help='Custom save filename (for ensemble model differentiation)')
    args = parser.parse_args()
    
    # Safety lock 3: prevent multiple training processes
    if not args.force:
        import os
        current_pid = os.getpid()
        result = subprocess.run(
            ["bash", "-c", f"ps aux | grep 'models.train' | grep -v grep | grep -v 'train.py' | awk '{{print $2}}'"],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            pids = [p.strip() for p in result.stdout.strip().split('\n') if p.strip()]
            other_processes = [p for p in pids if p != str(current_pid)]
            if len(other_processes) > 0:
                logger.error(f"\n❌ Detected existing training processes running! (PID: {', '.join(other_processes)})")
                logger.error("   To force start, use --force parameter")
                logger.error("   Or clean up residual processes first: pkill -9 -f models/train")
                sys.exit(1)
    
    logger.info("=" * 80)
    logger.info("CAPTCHA Recognition Model Training Program")
    logger.info("ConvNeXt-Tiny + Bidirectional LSTM + CTC")
    logger.info("=" * 80)

    config_loader = ConfigLoader()
    config_loader.reload()
    config = config_loader.get_config()

    logger.info(f"\nConfiguration:")
    logger.info(f"  Model type: {config.get('model', {}).get('type', 'captcha_model')}")
    logger.info(f"  Training epochs: {config.get('training.num_epochs', 80)}")
    logger.info(f"  Batch Size: {config.get('training.batch_size', 256)}")
    logger.info(f"  Learning rate: {config.get('training.learning_rate', 0.00008)}")
    logger.info(f"  CTC/CE weights: {config.get('training.ctc_weight', 0.6)}/{config.get('training.ce_weight', 0.4)}")
    
    if args.resume:
        logger.info(f"  Resume from checkpoint: {args.resume}")
    if args.seed != 42:
        logger.info(f"  Random seed: {args.seed} (command line)")
    if args.save_as:
        logger.info(f"  Save as: {args.save_as} (command line)")

    # Command-line arguments override config
    if args.seed != 42:
        config_loader.set('seed', args.seed)
        logger.info(f"  Random seed: {args.seed} (command line)")

    if args.streaming:
        config_loader.set('streaming_dataset.enabled', True)
        logger.info("  Streaming dataset: enabled (command line)")
    if args.num_epochs is not None:
        config_loader.set('epochs', args.num_epochs)
        config_loader.set('training.num_epochs', args.num_epochs)
        logger.info(f"  Training epochs: {args.num_epochs} (command line)")
    if args.batch_size is not None:
        config_loader.set('batch_size', args.batch_size)
        config_loader.set('training.batch_size', args.batch_size)
        logger.info(f"  Batch Size: {args.batch_size} (command line)")

    # Start training
    try:
        train(config, resume_from=args.resume)
        logger.info("\n✅ Training completed successfully!")
    except KeyboardInterrupt:
        logger.warning("\n⚠️  Training interrupted by user")
    except Exception as e:
        logger.error(f"\n❌ Training failed: {e}")
        raise


if __name__ == '__main__':
    main()
