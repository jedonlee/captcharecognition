# -*- coding: utf-8 -*-
"""
Baseline VGG CNN + BiLSTM Training Script

Training pipeline:
- MemoryMonitor, setup_logging, set_seed from utils.training_utils
- StreamCaptchaDataset for streaming data generation
- HybridCTCELoss (CTC 40% + CE 60%)
- AMP mixed precision training
- OneCycleLR scheduler
- Gradient clipping (max_norm=1.0)
- Early stopping based on fixed val set
- Validation every 2 epochs
- Best model auto-saved to checkpoints/best_model_baseline.pth
- TensorBoard logging
"""

import os
import sys
import time
import gc
import logging
import argparse
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast as amp_autocast
from tqdm import tqdm
import numpy as np

try:
    from tensorboardX import SummaryWriter
except ImportError:
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        SummaryWriter = None


from utils.config_loader import ConfigLoader
from utils.device_manager import DeviceManager
from utils.directory_manager import DirectoryManager
from utils.chars import get_all_chars, NUM_CHARS, get_mapper
from utils.decoder import beam_search_decode, postprocess_text_list, normalize_equiv
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
)
from utils.training_recorder import TrainingRecorder
from models.dataset import StreamCaptchaDataset, CaptchaDataset, collate_fn
from models.hybrid_loss_fixed import HybridCTCELoss
from models.transforms import get_train_transform, get_val_transform
from models.baseline_vgg_cnn_lstm import BaselineVGGCNNBiLSTM


def train_one_epoch(model, dataloader, criterion, optimizer, device, writer, epoch,
                    scaler=None, scheduler=None, grad_clip_enabled=True,
                    grad_clip_max_norm=1.0, memory_monitor=None, logger=None):
    model.train()

    total_loss = 0.0
    total_ctc_loss = 0.0
    total_ce_loss = 0.0
    total_batches = 0

    moving_avg_loss = 0.0
    alpha = 0.1

    device_manager = DeviceManager(verbose=False)
    use_amp = device_manager.supports_amp() and scaler is not None
    amp_context = amp_autocast('cuda') if use_amp else torch.contextlib.nullcontext()

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")

    for batch_idx, batch in enumerate(progress_bar):
        try:
            if not batch or 'images' not in batch:
                continue

            if batch['images'].numel() == 0:
                continue

            non_blocking = device.type == 'cuda'
            images = batch['images'].to(device, non_blocking=non_blocking)
            label_indices = batch['label_indices'].to(device, non_blocking=non_blocking)
            label_lengths = batch['label_lengths'].to(device, non_blocking=non_blocking)

            targets_list = []
            for i in range(len(label_lengths)):
                targets_list.append(label_indices[i][:label_lengths[i]])
            targets = torch.cat(targets_list)
            targets_lengths = label_lengths

            with amp_context:
                encoder_out, decoder_out = model(images)

                batch_size = images.size(0)
                input_lengths = torch.full((batch_size,), encoder_out.size(0),
                                          dtype=torch.long).to(device)

                if torch.isnan(encoder_out).any() or torch.isinf(encoder_out).any():
                    del encoder_out, decoder_out, images, label_indices, label_lengths
                    del targets, targets_lengths
                    continue

                if torch.isnan(decoder_out).any() or torch.isinf(decoder_out).any():
                    del encoder_out, decoder_out, images, label_indices, label_lengths
                    continue

                loss, ctc_loss, ce_loss = criterion(
                    encoder_out, decoder_out, targets, input_lengths, targets_lengths
                )

            if torch.isnan(loss) or torch.isinf(loss):
                del encoder_out, decoder_out, images, label_indices, label_lengths
                del targets, targets_lengths, loss, ctc_loss, ce_loss
                continue

            if use_amp and scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                if grad_clip_enabled:
                    torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                 max_norm=float(grad_clip_max_norm))
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            else:
                loss.backward()
                if grad_clip_enabled:
                    torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                 max_norm=float(grad_clip_max_norm))
                optimizer.step()
                optimizer.zero_grad()

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

            del loss, ctc_loss, ce_loss, encoder_out, decoder_out
            del images, label_indices, label_lengths, targets, targets_lengths, input_lengths

        except RuntimeError as e:
            if logger:
                logger.exception(f"RuntimeError in batch {batch_idx}")
            continue
        except Exception as e:
            if logger:
                logger.exception(f"Error in batch {batch_idx}")
            continue

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
             memory_monitor=None, logger=None):
    mapper = get_mapper()
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

    try:
        with torch.no_grad():
            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch} [Val]")

            for batch_idx, batch in enumerate(progress_bar):
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

                try:
                    pred_strings = beam_search_decode(encoder_out, mapper, beam_width=10, enable_corrector=False)
                    pred_strings = postprocess_text_list(pred_strings)

                    target_strings = []
                    for i in range(batch_size):
                        target_len = label_lengths[i].item()
                        target_indices = label_indices[i, :target_len].tolist()
                        target_str = ''.join([chars[idx] for idx in target_indices])
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
                    if logger:
                        logger.warning(f"Decode error at batch {batch_idx}: {decode_error}")

                if memory_monitor is not None and batch_idx % 50 == 0:
                    memory_monitor.update()

                del loss, ctc_loss, ce_loss, encoder_out, decoder_out
                del images, label_indices, label_lengths, targets, targets_lengths, input_lengths

    except Exception as e:
        if logger:
            logger.exception(f"Validation error: {str(e)}")
    finally:
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

    if writer:
        writer.add_scalar('Loss/val_epoch', avg_loss, epoch)
        writer.add_scalar('Loss/val_ctc_epoch', avg_ctc_loss, epoch)
        writer.add_scalar('Loss/val_ce_epoch', avg_ce_loss, epoch)
        writer.add_scalar('Accuracy/val_image', image_accuracy, epoch)
        writer.add_scalar('Accuracy/val_char', char_accuracy, epoch)

    return avg_loss, avg_ctc_loss, avg_ce_loss, image_accuracy, char_accuracy


def parse_args():
    parser = argparse.ArgumentParser(description='Baseline VGG CNN + BiLSTM Training')
    parser.add_argument('--model_type', type=str, default='baseline',
                       help='Model type (default: baseline)')
    parser.add_argument('--num_epochs', type=int, default=None,
                       help='Number of training epochs (overrides config)')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Batch size (overrides config)')
    parser.add_argument('--streaming', action='store_true', default=None,
                       help='Use streaming dataset')
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume from checkpoint path')
    return parser.parse_args()


def train(args):
    config_loader = ConfigLoader()
    config_loader.reload()
    config = config_loader.get_config()

    dir_manager = DirectoryManager(verbose=True)

    log_dir = config.get('training.log_dir', 'logs')
    dir_manager.ensure_directory(log_dir)
    logger = setup_logging(log_dir, level=logging.INFO)

    logger.info("=" * 80)
    logger.info("Baseline VGG CNN + BiLSTM Training Started")
    logger.info("=" * 80)

    seed = config.get('training.seed', 42)
    set_seed(seed)
    logger.info(f"Random seed set: {seed}")

    device_manager = DeviceManager()
    device = device_manager.device
    logger.info(f"Device: {device}")

    enable_memory_monitor = config.get('training', {}).get('enable_memory_monitor', True)
    memory_monitor = None
    if enable_memory_monitor:
        memory_monitor = MemoryMonitor(device, logger=logger)
        logger.info("Memory monitor enabled")

    logger.info("Creating model...")
    num_classes = NUM_CHARS + 1
    model = BaselineVGGCNNBiLSTM(num_classes=num_classes)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model params: {total_params:,} (trainable: {trainable_params:,})")

    logger.info("Loading dataset...")

    use_streaming = args.streaming
    if use_streaming is None:
        use_streaming = config.get('streaming_dataset', {}).get('enabled', False)

    num_samples_per_epoch = config.get('streaming_dataset', {}).get('num_samples_per_epoch', 100000)

    if use_streaming:
        logger.info(f"Using streaming dataset: {num_samples_per_epoch} samples/epoch")
        train_dataset = StreamCaptchaDataset(
            transform=get_train_transform(),
            max_length=6,
            num_samples_per_epoch=num_samples_per_epoch,
            seed=config.get('streaming_dataset', {}).get('seed', 42)
        )
    else:
        train_dir = config_loader.get_data_dir('train')
        logger.info(f"Using fixed dataset, train_dir: {train_dir}")
        train_dataset = CaptchaDataset(
            data_dir=train_dir,
            transform=get_train_transform(),
            max_length=6
        )

    val_dir = config_loader.get_data_dir('val')
    val_dataset = CaptchaDataset(
        data_dir=val_dir,
        transform=get_val_transform(),
        max_length=6
    )

    batch_size = args.batch_size if args.batch_size is not None else config.get('batch_size', 128)
    batch_size = config.get('training.batch_size', batch_size)
    num_workers = config.get('training.num_workers', 4)

    logger.info(f"Batch size: {batch_size}")

    pin_memory = config.get('system', {}).get('pin_memory', True)
    prefetch_factor = config.get('system', {}).get('prefetch_factor', 2)
    persistent_workers = config.get('system', {}).get('persistent_workers', True)

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

    logger.info(f"Train set: {len(train_dataset)}, batches: {len(train_loader)}")
    logger.info(f"Val set: {len(val_dataset)}, batches: {len(val_loader)}")

    ctc_weight = config.get('training.ctc_weight', 0.4)
    ce_weight = config.get('training.ce_weight', 0.6)
    label_smoothing = config.get('training.label_smoothing', 0.05)
    criterion = HybridCTCELoss(
        num_chars=NUM_CHARS,
        ctc_weight=ctc_weight,
        ce_weight=ce_weight,
        label_smoothing=label_smoothing
    )
    criterion = criterion.to(device)
    logger.info(f"Loss: CTC={criterion.ctc_weight:.1f}, CE={criterion.ce_weight:.1f}, smoothing={label_smoothing}")

    lr = config.get('training.learning_rate', 0.00005)
    weight_decay = config.get('training.weight_decay', 0.00005)

    try:
        lr = float(lr)
    except (ValueError, TypeError):
        lr = 0.00005

    try:
        weight_decay = float(weight_decay)
    except (ValueError, TypeError):
        weight_decay = 0.00005

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    logger.info(f"Optimizer: AdamW (lr={lr:.6f}, weight_decay={weight_decay})")

    num_epochs = args.num_epochs if args.num_epochs is not None else config.get('training.num_epochs', 80)
    max_lr = config.get('training.max_lr', 0.0005)
    warmup_pct = config.get('training.warmup_pct', 0.15)

    steps_per_epoch = len(train_loader)
    total_steps = num_epochs * steps_per_epoch

    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=max_lr,
        total_steps=total_steps,
        pct_start=warmup_pct,
        anneal_strategy='cos',
        final_div_factor=100
    )
    logger.info(f"Scheduler: OneCycleLR (max_lr={max_lr:.6f}, total_steps={total_steps}, warmup={warmup_pct})")

    use_amp = config.get('amp', True)
    scaler = GradScaler() if use_amp else None
    logger.info(f"AMP: {'enabled' if use_amp else 'disabled'}")

    tensorboard_dir = config.get('logging', {}).get('tensorboard', {}).get('tensorboard_dir', 'runs')
    dir_manager.ensure_directory(tensorboard_dir)
    writer = SummaryWriter(tensorboard_dir) if SummaryWriter else None
    if writer:
        logger.info(f"TensorBoard log dir: {tensorboard_dir}")

    patience = config.get('early_stopping', {}).get('patience', 15)
    best_val_accuracy = 0.0
    epochs_no_improve = 0
    logger.info(f"Early stopping patience: {patience}")

    grad_clip_config = config.get('training', {}).get('gradient_clipping', {})
    grad_clip_enabled = grad_clip_config.get('enabled', True)
    grad_clip_max_norm = grad_clip_config.get('max_norm', 1.0)
    logger.info(f"Gradient clipping: enabled={grad_clip_enabled}, max_norm={grad_clip_max_norm}")

    checkpoint_dir = config.get('training', {}).get('checkpoint_dir', 'checkpoints')
    dir_manager.ensure_directory(checkpoint_dir)

    start_epoch = 0
    resume_path = args.resume
    if resume_path and os.path.exists(resume_path):
        resume_info = load_checkpoint(resume_path, model, optimizer, scheduler, scaler, device)
        start_epoch = resume_info['epoch'] + 1
        best_val_accuracy = resume_info['best_val_accuracy']
        logger.info(f"Resumed from checkpoint: epoch {start_epoch}, best_acc={best_val_accuracy:.4f}")
    else:
        baseline_ckpt = str(config_loader.get_project_root() / config.get('checkpoint', {}).get('checkpoint_dir', 'checkpoints') /
                            config.get('model.baseline', {}).get('checkpoint_name', 'baseline_vgg_best.pth'))
        if os.path.exists(baseline_ckpt):
            try:
                checkpoint = torch.load(baseline_ckpt, map_location=device, weights_only=True)
                model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                if 'optimizer_state_dict' in checkpoint:
                    try:
                        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    except Exception:
                        pass
                if 'scheduler_state_dict' in checkpoint:
                    try:
                        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                    except Exception:
                        pass
                if 'best_val_accuracy' in checkpoint:
                    best_val_accuracy = checkpoint['best_val_accuracy']
                if 'epoch' in checkpoint:
                    start_epoch = checkpoint['epoch'] + 1
                logger.info(f"Loaded previous baseline model from epoch {start_epoch}, best_acc={best_val_accuracy:.4f}")
            except Exception as e:
                logger.warning(f"Failed to load baseline checkpoint: {e}")
                logger.info("Starting from scratch")

    chars = get_all_chars()

    recorder = TrainingRecorder(
        experiment_name=f"baseline_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        output_dir='experiments/records'
    )
    recorder.set_config(config)
    logger.info("Training recorder initialized")

    logger.info("=" * 80)
    logger.info(f"Starting training: {num_epochs} epochs")
    logger.info("=" * 80)

    training_start_time = time.time()

    for epoch in range(start_epoch, num_epochs):
        epoch_start_time = time.time()
        logger.info(f"\n{'='*40} Epoch {epoch+1}/{num_epochs} {'='*40}")

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
            logger=logger
        )

        if (epoch + 1) % 2 == 0:
            val_loss, val_ctc_loss, val_ce_loss, val_image_acc, val_char_acc = validate(
                model=model,
                dataloader=val_loader,
                criterion=criterion,
                device=device,
                writer=writer,
                epoch=epoch,
                chars=chars,
                memory_monitor=memory_monitor,
                logger=logger
            )
        else:
            val_loss, val_ctc_loss, val_ce_loss, val_image_acc, val_char_acc = 0.0, 0.0, 0.0, 0.0, 0.0

        epoch_time = time.time() - epoch_start_time

        logger.info(f"\nEpoch {epoch+1} Summary:")
        logger.info(f"  Train Loss: {train_loss:.4f} (CTC: {train_ctc_loss:.4f}, CE: {train_ce_loss:.4f})")
        if (epoch + 1) % 2 == 0:
            logger.info(f"  Val Loss: {val_loss:.4f} (CTC: {val_ctc_loss:.4f}, CE: {val_ce_loss:.4f})")
            logger.info(f"  Image Accuracy: {val_image_acc:.4f} ({val_image_acc*100:.2f}%)")
        else:
            logger.info(f"  Val: skipped (odd epoch)")
        logger.info(f"  Char Accuracy: {val_char_acc:.4f} ({val_char_acc*100:.2f}%)")
        logger.info(f"  Time: {epoch_time:.2f}s")

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

        if val_image_acc > best_val_accuracy:
            best_val_accuracy = val_image_acc
            epochs_no_improve = 0

            save_checkpoint(
                save_dir=checkpoint_dir,
                filename='baseline_vgg_best.pth',
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
                config=config,
                dir_manager=dir_manager,
                logger=logger
            )
            logger.info(f"New best model! Image accuracy: {best_val_accuracy:.4f}")
        else:
            epochs_no_improve += 1
            logger.info(f"Current best accuracy: {best_val_accuracy:.4f} ({epochs_no_improve}/{patience} epochs no improvement)")

        if (epoch + 1) % 5 == 0:
            save_checkpoint(
                save_dir=checkpoint_dir,
                filename=f'baseline_checkpoint_epoch_{epoch+1}.pth',
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
                config=config,
                dir_manager=dir_manager,
                logger=logger
            )

        if epochs_no_improve >= patience:
            logger.info(f"\nEarly stopping triggered! {patience} epochs without improvement")
            logger.info(f"Best validation accuracy: {best_val_accuracy:.4f}")
            break

    total_training_time = time.time() - training_start_time
    logger.info("\n" + "=" * 80)
    logger.info("Training completed!")
    logger.info("=" * 80)
    logger.info(f"Total training time: {total_training_time/3600:.2f} hours")
    logger.info(f"Best validation image accuracy: {best_val_accuracy:.4f} ({best_val_accuracy*100:.2f}%)")

    recorder.print_summary()

    if memory_monitor:
        memory_monitor.log_memory_summary()

    if writer:
        writer.close()
        verify_tensorboard_logs(tensorboard_dir, logger=logger)


def main():
    args = parse_args()

    # Initialize logger in main()
    log_dir = str(Path(__file__).parent / 'logs')
    os.makedirs(log_dir, exist_ok=True)
    logger = setup_logging(log_dir, level=logging.INFO)

    logger.info("=" * 80)
    logger.info("Baseline VGG CNN + BiLSTM Training")
    logger.info("=" * 80)

    config_loader = ConfigLoader()
    config_loader.reload()
    config = config_loader.get_config()

    logger.info(f"\nConfiguration:")
    logger.info(f"  Model type: {args.model_type}")
    logger.info(f"  Epochs: {args.num_epochs if args.num_epochs is not None else config.get('training.num_epochs', 80)}")
    logger.info(f"  Batch size: {args.batch_size if args.batch_size is not None else config.get('training.batch_size', 128)}")
    logger.info(f"  Streaming: {args.streaming if args.streaming is not None else config.get('streaming_dataset', {}).get('enabled', False)}")
    logger.info(f"  Learning rate: {config.get('training.learning_rate', 0.00005)}")
    logger.info(f"  CTC/CE weights: {config.get('training.ctc_weight', 0.6)}/{config.get('training.ce_weight', 0.4)}")

    try:
        train(args)
        logger.info("\nTraining completed successfully!")
    except KeyboardInterrupt:
        logger.info("\nTraining interrupted by user")
    except Exception as e:
        logger.exception(f"\nTraining failed: {e}")
        raise


if __name__ == '__main__':
    main()
