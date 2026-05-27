# -*- coding: utf-8 -*-
"""
Training utility functions (extracted from train.py as standalone tools)
Functions: memory monitoring, logging config, random seeds, checkpoint save/load, batch_size auto-adjust, LR params calculation, EMA

Extraction source: models/train.py → 9 independent functions/classes, no business logic coupling
Usage: from utils.training_utils import MemoryMonitor, setup_logging, ...
"""

import os
import sys
import random
import math
import logging
import gc
import threading
import copy

import torch
import numpy as np

logger = logging.getLogger(__name__)

try:
    import psutil
except ImportError:
    psutil = None

from pathlib import Path
from utils.directory_manager import DirectoryManager


class MemoryMonitor:
    """
    Memory monitor class (GPU and CPU memory monitoring, thread-safe)

    Features:
    - Monitor GPU memory usage
    - Monitor CPU memory usage
    - Record memory peaks
    - Provide memory usage reports
    - Auto-clean cache
    - Thread-safe design, supports multi-process data loading
    """

    def __init__(self, device, logger=None, monitor_interval=10, max_history=1000,
                 high_memory_threshold=0.85, critical_memory_threshold=0.95):
        self.device = device
        self.logger = logger
        self.base_monitor_interval = monitor_interval
        self.monitor_interval = monitor_interval
        self.max_history = max_history
        self.batch_count = 0
        self.high_memory_threshold = high_memory_threshold
        self.critical_memory_threshold = critical_memory_threshold

        self._lock = threading.Lock()

        self.gpu_memory_stats = {
            'allocated': [],
            'reserved': [],
            'max_allocated': 0,
            'max_reserved': 0
        }

        self.cpu_memory_stats = {
            'used': [],
            'available': [],
            'percent': [],
            'max_used': 0,
            'max_percent': 0
        }

        self._record_initial_memory()

    def _record_initial_memory(self):
        if self.device.type == 'cuda':
            self.initial_gpu_allocated = torch.cuda.memory_allocated(self.device) / 1024**3
            self.initial_gpu_reserved = torch.cuda.memory_reserved(self.device) / 1024**3
        else:
            self.initial_gpu_allocated = 0
            self.initial_gpu_reserved = 0

        if psutil is not None:
            self.initial_cpu_used = psutil.virtual_memory().used / 1024**3
            self.initial_cpu_available = psutil.virtual_memory().available / 1024**3
            self.initial_cpu_percent = psutil.virtual_memory().percent
        else:
            self.initial_cpu_used = 0
            self.initial_cpu_available = 0
            self.initial_cpu_percent = 0

    def _trim_history(self):
        if len(self.gpu_memory_stats['allocated']) > self.max_history:
            self.gpu_memory_stats['allocated'] = self.gpu_memory_stats['allocated'][-self.max_history:]
            self.gpu_memory_stats['reserved'] = self.gpu_memory_stats['reserved'][-self.max_history:]

        if len(self.cpu_memory_stats['used']) > self.max_history:
            self.cpu_memory_stats['used'] = self.cpu_memory_stats['used'][-self.max_history:]
            self.cpu_memory_stats['available'] = self.cpu_memory_stats['available'][-self.max_history:]
            self.cpu_memory_stats['percent'] = self.cpu_memory_stats['percent'][-self.max_history:]

    def _adjust_cleanup_interval(self, gpu_percent=None, cpu_percent=None):
        max_percent = 0.0
        if gpu_percent is not None:
            max_percent = max(max_percent, gpu_percent)
        if cpu_percent is not None:
            max_percent = max(max_percent, cpu_percent / 100.0)

        if max_percent >= self.critical_memory_threshold:
            self.monitor_interval = 1
        elif max_percent >= self.high_memory_threshold:
            self.monitor_interval = max(2, self.base_monitor_interval // 2)
        else:
            self.monitor_interval = self.base_monitor_interval

    def update(self):
        with self._lock:
            self.batch_count += 1
            gpu_percent = None

            if self.device.type == 'cuda':
                gpu_allocated = torch.cuda.memory_allocated(self.device) / 1024**3
                gpu_reserved = torch.cuda.memory_reserved(self.device) / 1024**3

                total_memory = torch.cuda.get_device_properties(self.device).total_memory / 1024**3
                gpu_percent = gpu_reserved / total_memory if total_memory > 0 else 0.0

                self.gpu_memory_stats['allocated'].append(gpu_allocated)
                self.gpu_memory_stats['reserved'].append(gpu_reserved)
                self.gpu_memory_stats['max_allocated'] = max(self.gpu_memory_stats['max_allocated'], gpu_allocated)
                self.gpu_memory_stats['max_reserved'] = max(self.gpu_memory_stats['max_reserved'], gpu_reserved)

                gpu_info = {
                    'allocated_gb': gpu_allocated,
                    'reserved_gb': gpu_reserved,
                    'percent': gpu_percent,
                    'max_allocated_gb': self.gpu_memory_stats['max_allocated'],
                    'max_reserved_gb': self.gpu_memory_stats['max_reserved']
                }
            else:
                gpu_info = {
                    'allocated_gb': 0,
                    'reserved_gb': 0,
                    'percent': 0.0,
                    'max_allocated_gb': 0,
                    'max_reserved_gb': 0
                }

            if psutil is not None:
                cpu_memory = psutil.virtual_memory()
                cpu_used = cpu_memory.used / 1024**3
                cpu_available = cpu_memory.available / 1024**3
                cpu_percent = cpu_memory.percent
            else:
                cpu_used = 0
                cpu_available = 0
                cpu_percent = 0

            self.cpu_memory_stats['used'].append(cpu_used)
            self.cpu_memory_stats['available'].append(cpu_available)
            self.cpu_memory_stats['percent'].append(cpu_percent)
            self.cpu_memory_stats['max_used'] = max(self.cpu_memory_stats['max_used'], cpu_used)
            self.cpu_memory_stats['max_percent'] = max(self.cpu_memory_stats['max_percent'], cpu_percent)

            cpu_info = {
                'used_gb': cpu_used,
                'available_gb': cpu_available,
                'percent': cpu_percent,
                'max_used_gb': self.cpu_memory_stats['max_used'],
                'max_percent': self.cpu_memory_stats['max_percent']
            }

            self._adjust_cleanup_interval(gpu_percent=gpu_percent, cpu_percent=cpu_percent)

            if self.batch_count % (self.monitor_interval * 10) == 0:
                self._trim_history()

            if self.batch_count % self.monitor_interval == 0:
                self._cleanup_cache()

            return {'gpu': gpu_info, 'cpu': cpu_info, 'batch_count': self.batch_count, 'monitor_interval': self.monitor_interval}

    def _cleanup_cache(self):
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()
        if self.logger and self.batch_count % (self.monitor_interval * 5) == 0:
            self.logger.info(f"Memory cache cleared (batch {self.batch_count})")

    def get_memory_summary(self):
        with self._lock:
            return {
                'batch_count': self.batch_count,
                'gpu': {
                    'initial_allocated_gb': self.initial_gpu_allocated,
                    'initial_reserved_gb': self.initial_gpu_reserved,
                    'current_allocated_gb': self.gpu_memory_stats['allocated'][-1] if self.gpu_memory_stats['allocated'] else 0,
                    'current_reserved_gb': self.gpu_memory_stats['reserved'][-1] if self.gpu_memory_stats['reserved'] else 0,
                    'max_allocated_gb': self.gpu_memory_stats['max_allocated'],
                    'max_reserved_gb': self.gpu_memory_stats['max_reserved'],
                    'avg_allocated_gb': np.mean(self.gpu_memory_stats['allocated']) if self.gpu_memory_stats['allocated'] else 0,
                    'avg_reserved_gb': np.mean(self.gpu_memory_stats['reserved']) if self.gpu_memory_stats['reserved'] else 0
                },
                'cpu': {
                    'initial_used_gb': self.initial_cpu_used,
                    'initial_available_gb': self.initial_cpu_available,
                    'initial_percent': self.initial_cpu_percent,
                    'current_used_gb': self.cpu_memory_stats['used'][-1] if self.cpu_memory_stats['used'] else 0,
                    'current_available_gb': self.cpu_memory_stats['available'][-1] if self.cpu_memory_stats['available'] else 0,
                    'current_percent': self.cpu_memory_stats['percent'][-1] if self.cpu_memory_stats['percent'] else 0,
                    'max_used_gb': self.cpu_memory_stats['max_used'],
                    'max_percent': self.cpu_memory_stats['max_percent'],
                    'avg_used_gb': np.mean(self.cpu_memory_stats['used']) if self.cpu_memory_stats['used'] else 0,
                    'avg_percent': np.mean(self.cpu_memory_stats['percent']) if self.cpu_memory_stats['percent'] else 0
                }
            }

    def log_memory_summary(self, epoch=None):
        summary = self.get_memory_summary()
        prefix = f"Epoch {epoch} - " if epoch is not None else ""
        if self.logger:
            self.logger.info("=" * 60)
            self.logger.info(f"{prefix}Memory Usage Summary")
            self.logger.info("=" * 60)
            self.logger.info(f"Total training batches: {summary['batch_count']}")
            if self.device.type == 'cuda':
                self.logger.info("\nGPU Memory Usage:")
                for line in [
                    f"  Initial Allocated: {summary['gpu']['initial_allocated_gb']:.3f} GB",
                    f"  Current Allocated: {summary['gpu']['current_allocated_gb']:.3f} GB",
                    f"  Peak Allocated: {summary['gpu']['max_allocated_gb']:.3f} GB",
                    f"  Avg Allocated: {summary['gpu']['avg_allocated_gb']:.3f} GB",
                    f"  Current Reserved: {summary['gpu']['current_reserved_gb']:.3f} GB",
                    f"  Peak Reserved: {summary['gpu']['max_reserved_gb']:.3f} GB",
                    f"  Avg Reserved: {summary['gpu']['avg_reserved_gb']:.3f} GB",
                ]:
                    self.logger.info(line)
            self.logger.info("\nCPU Memory Usage:")
            for line in [
                f"  Initial Used: {summary['cpu']['initial_used_gb']:.3f} GB ({summary['cpu']['initial_percent']:.1f}%)",
                f"  Current Used: {summary['cpu']['current_used_gb']:.3f} GB ({summary['cpu']['current_percent']:.1f}%)",
                f"  Current Available: {summary['cpu']['current_available_gb']:.3f} GB",
                f"  Peak Used: {summary['cpu']['max_used_gb']:.3f} GB ({summary['cpu']['max_percent']:.1f}%)",
                f"  Avg Used: {summary['cpu']['avg_used_gb']:.3f} GB ({summary['cpu']['avg_percent']:.1f}%)",
            ]:
                self.logger.info(line)
            self.logger.info("=" * 60)

    def get_current_memory(self):
        with self._lock:
            current = {}
            if self.device.type == 'cuda':
                current['gpu_allocated_gb'] = torch.cuda.memory_allocated(self.device) / 1024**3
                current['gpu_reserved_gb'] = torch.cuda.memory_reserved(self.device) / 1024**3
            else:
                current['gpu_allocated_gb'] = 0
                current['gpu_reserved_gb'] = 0
            if psutil is not None:
                cpu_memory = psutil.virtual_memory()
                current['cpu_used_gb'] = cpu_memory.used / 1024**3
                current['cpu_available_gb'] = cpu_memory.available / 1024**3
                current['cpu_percent'] = cpu_memory.percent
            else:
                current['cpu_used_gb'] = 0
                current['cpu_available_gb'] = 0
                current['cpu_percent'] = 0
            return current

    def clear_history(self):
        with self._lock:
            self.gpu_memory_stats['allocated'].clear()
            self.gpu_memory_stats['reserved'].clear()
            self.cpu_memory_stats['used'].clear()
            self.cpu_memory_stats['available'].clear()
            self.cpu_memory_stats['percent'].clear()


def setup_logging(log_dir, log_filename='training.log', level=logging.INFO,
                  console_level=logging.INFO, max_bytes=10485760, backup_count=5):
    """
    Configure logging system (file + console)

    Args:
        log_dir: log directory path
        log_filename: log file name
        level: file log level
        console_level: console log level
        max_bytes: max bytes per log file
        backup_count: number of backup log files to keep

    Returns:
        logger: configured logger object
    """
    dir_manager = DirectoryManager(verbose=True)
    log_path = Path(log_dir)
    dir_manager.ensure_directory(log_path)

    logger = logging.getLogger('CaptchaTraining')
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    log_file = log_path / log_filename
    try:
        file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except OSError as e:
        logger.warning(f"Failed to create file log handler: {e}")
        logger.warning("Will use console logging only")

    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    logger.info("=" * 60)
    logger.info("Logging system initialized")
    logger.info(f"Log directory: {log_path.absolute()}")
    logger.info(f"Log file: {log_file.absolute()}")
    logger.info(f"File log level: {logging.getLevelName(level)}")
    logger.info(f"Console log level: {logging.getLevelName(console_level)}")
    logger.info("=" * 60)

    return logger


def verify_tensorboard_logs(tensorboard_dir, logger=None):
    """
    Verify TensorBoard logs are generated correctly

    Args:
        tensorboard_dir: TensorBoard log directory path
        logger: logger object (optional)

    Returns:
        bool: whether TensorBoard logs are valid
    """
    tensorboard_path = Path(tensorboard_dir)

    if not tensorboard_path.exists():
        if logger:
            logger.warning(f"TensorBoard log directory does not exist: {tensorboard_path.absolute()}")
        return False

    log_files = list(tensorboard_path.glob('events.out.tfevents.*'))

    if len(log_files) > 0:
        if logger:
            logger.info(f"TensorBoard log verification successful, found {len(log_files)} log files")
            for log_file in log_files:
                logger.info(f"   - {log_file.name}")
        return True
    else:
        if logger:
            logger.warning(f"TensorBoard log directory exists but no log files found: {tensorboard_path.absolute()}")
        return False


def set_seed(seed):
    """Set random seed (ensure experiment reproducibility)"""
    if seed is not None:
        if isinstance(seed, (tuple, list)):
            seed = seed[0]
        elif isinstance(seed, str):
            seed = int(seed)
        elif isinstance(seed, (int, float)):
            seed = int(seed)

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def save_checkpoint(save_dir, filename, model, optimizer, scheduler, scaler, epoch,
                   best_val_accuracy, train_loss, val_loss, val_image_accuracy,
                   val_char_accuracy, config, dir_manager=None, logger=None):
    """
    Save training checkpoint (full training state, with disk space check and atomic save)

    Args:
        save_dir: save directory path
        filename: checkpoint filename
        model/optimizer/scheduler/scaler: training component states
        epoch/best_val_accuracy/*_loss/*_accuracy: training metrics
        config: training config dict
        dir_manager/logger: optional helper objects

    Returns:
        saved file path, None on failure
    """
    if dir_manager is not None:
        dir_manager.ensure_directory(save_dir)
    else:
        os.makedirs(save_dir, exist_ok=True)

    min_required_space = 500 * 1024 * 1024
    try:
        if os.name == 'nt':
            import ctypes
            free_bytes = ctypes.c_ulonglong(0)
            ctypes.windll.kernel32.GetDiskFreeSpaceExW(ctypes.c_wchar_p(save_dir), None, None, ctypes.pointer(free_bytes))
            free_space = free_bytes.value
        else:
            statvfs = os.statvfs(save_dir)
            free_space = statvfs.f_bavail * statvfs.f_frsize

        if free_space < min_required_space:
            error_msg = f"Insufficient disk space, need at least {min_required_space / 1024 / 1024:.0f}MB, currently available {free_space / 1024 / 1024:.0f}MB"
            if logger:
                logger.error(error_msg)
            return None
    except Exception as e:
        if logger:
            logger.warning(f"Failed to check disk space: {e}, continuing to save")

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
        'scaler_state_dict': scaler.state_dict() if scaler is not None else None,
        'best_val_accuracy': best_val_accuracy,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'val_image_accuracy': val_image_accuracy,
        'val_char_accuracy': val_char_accuracy,
        'config': config
    }

    save_path = os.path.join(save_dir, filename)
    tmp_path = save_path + '.tmp'

    try:
        torch.save(checkpoint, tmp_path)
        os.replace(tmp_path, save_path)
        success_msg = f"Checkpoint saved: {save_path}"
        logger.info(success_msg)
        return save_path
    except PermissionError as e:
        error_msg = f"Failed to save checkpoint: insufficient permissions - {e}"
        logger.error(error_msg)
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass
        return None
    except Exception as e:
        error_msg = f"Failed to save checkpoint: {e}"
        logger.error(error_msg)
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass
        return None


def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None, scaler=None, device='cpu'):
    """
    Load training state from checkpoint

    Args:
        checkpoint_path: checkpoint file path
        model: model object (required)
        optimizer/scheduler/scaler: optional restoration objects
        device: device type

    Returns:
        dict: contains epoch, best_val_accuracy, config, checkpoint info
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file does not exist: {checkpoint_path}")

    logger.info(f"Loading checkpoint: {checkpoint_path}")

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)

        missing_keys, unexpected_keys = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        if missing_keys:
            logger.warning(f"Missing keys in checkpoint (randomly initialized): {missing_keys}")
        if unexpected_keys:
            logger.warning(f"Unexpected keys in checkpoint (ignored): {unexpected_keys}")
        logger.info(f"Model parameters loaded")

        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                logger.info(f"Optimizer state loaded")
            except (ValueError, RuntimeError) as e:
                logger.warning(f"Optimizer state incompatible (skipped, starting from initial state): {e}")

        if scheduler is not None and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict'] is not None:
            try:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                logger.info(f"Learning rate scheduler state loaded")
            except (ValueError, RuntimeError, KeyError) as e:
                logger.warning(f"Scheduler state incompatible (skipped, starting from initial state): {e}")

        if scaler is not None and 'scaler_state_dict' in checkpoint and checkpoint['scaler_state_dict'] is not None:
            try:
                scaler.load_state_dict(checkpoint['scaler_state_dict'])
                logger.info(f"Gradient scaler state loaded")
            except (ValueError, RuntimeError) as e:
                logger.warning(f"Gradient scaler state incompatible (skipped, starting from initial state): {e}")

        resume_info = {
            'epoch': checkpoint.get('epoch', 0),
            'best_val_accuracy': checkpoint.get('best_val_accuracy', 0.0),
            'config': checkpoint.get('config', {}),
            'checkpoint': checkpoint
        }

        logger.info(f"Checkpoint loaded successfully")
        logger.info(f"   - Resumed to epoch {resume_info['epoch']}")
        logger.info(f"   - Best validation accuracy: {resume_info['best_val_accuracy']:.4f}")

        return resume_info

    except Exception as e:
        logger.error(f"Failed to load checkpoint: {e}")
        raise


def auto_adjust_batch_size(model, device, initial_batch_size, device_manager=None, max_attempts=5):
    """
    Auto-adjust batch_size based on GPU memory to avoid OOM errors

    Args:
        model: model object
        device: device type
        initial_batch_size: initial batch_size
        device_manager: device manager (optional)
        max_attempts: max attempts

    Returns:
        adjusted batch_size
    """
    if device.type != 'cuda':
        logger.info(f"Using CPU device, batch_size remains: {initial_batch_size}")
        return initial_batch_size

    try:
        total_memory = torch.cuda.get_device_properties(device).total_memory
        total_memory_gb = total_memory / (1024 ** 3)
        logger.info(f"Total GPU memory: {total_memory_gb:.2f} GB")
    except Exception:
        logger.warning("Unable to get GPU memory info, using default batch_size")
        return initial_batch_size

    if total_memory_gb < 4:
        recommended_batch_size = 16
    elif total_memory_gb < 6:
        recommended_batch_size = 32
    elif total_memory_gb < 8:
        recommended_batch_size = 48
    elif total_memory_gb < 12:
        recommended_batch_size = 64
    elif total_memory_gb < 16:
        recommended_batch_size = 96
    elif total_memory_gb < 20:
        recommended_batch_size = 192
    else:
        recommended_batch_size = 256

    adjusted_batch_size = min(initial_batch_size, recommended_batch_size)
    logger.info(f"Recommended batch_size: {recommended_batch_size} (based on GPU memory)")
    logger.info(f"Using batch_size: {adjusted_batch_size}")

    model.eval()
    for attempt in range(max_attempts):
        try:
            test_batch_size = adjusted_batch_size
            test_input = torch.randn(test_batch_size, 3, 32, 128).to(device)

            with torch.no_grad():
                if device_manager is not None and device_manager.supports_amp():
                    from torch.amp import autocast
                    with autocast('cuda'):
                        _ = model(test_input)
                else:
                    _ = model(test_input)

            logger.info(f"batch_size={test_batch_size} test passed")
            return test_batch_size

        except RuntimeError as e:
            if 'out of memory' in str(e):
                adjusted_batch_size = max(1, adjusted_batch_size // 2)
                logger.warning(f"batch_size={test_batch_size} caused OOM, trying {adjusted_batch_size}")
                torch.cuda.empty_cache()
            else:
                raise
        except Exception as e:
            logger.warning(f"Error testing batch_size: {e}")
            break

    logger.warning(f"Unable to determine suitable batch_size, using minimum: 1")
    return 1


def calculate_onecycle_params(dataset_size, num_epochs, initial_lr=0.001, pct_start=0.2, max_lr_factor=4.0, warmup_epochs=None):
    """
    Automatically calculate OneCycleLR parameters based on dataset size and epochs

    Args:
        dataset_size: dataset size
        num_epochs: number of epochs
        initial_lr: initial learning rate
        pct_start: warmup phase ratio
        max_lr_factor: max LR multiplier
        warmup_epochs: warmup epochs (if provided, overrides pct_start)

    Returns:
        tuple: (max_lr, total_steps, pct_start)
    """
    total_steps = num_epochs * (dataset_size // 64)

    if dataset_size < 1000:
        factor = 2.0
    elif dataset_size < 5000:
        factor = 3.0
    else:
        factor = 4.0
    factor = min(float(max_lr_factor), float(factor))
    max_lr = initial_lr * factor

    # If warmup_epochs is provided, use it to calculate pct_start
    if warmup_epochs is not None and warmup_epochs > 0:
        pct_start = warmup_epochs / num_epochs
    else:
        # Otherwise use default logic
        if num_epochs < 10:
            pct_start = 0.3
        elif num_epochs < 50:
            pct_start = 0.25
        else:
            pct_start = 0.2

    logger.info(f"OneCycleLR parameters:")
    logger.info(f"  max_lr: {max_lr:.6f} (initial lr: {initial_lr:.6f})")
    logger.info(f"  total_steps: {total_steps} (will be recalculated after DataLoader creation)")
    logger.info(f"  pct_start: {pct_start:.2f} (warmup phase ratio)")

    return max_lr, total_steps, pct_start


def calculate_dynamic_patience(num_epochs, base_patience=10):
    """
    Dynamically calculate early stopping patience based on epochs (log scaling)

    Args:
        num_epochs: number of epochs
        base_patience: base patience value

    Returns:
        int: dynamically calculated patience
    """
    if num_epochs <= 10:
        patience = base_patience
    elif num_epochs <= 50:
        patience = base_patience + int(math.log2(num_epochs / 10) * 5)
    elif num_epochs <= 100:
        patience = base_patience + int(math.log2(num_epochs / 10) * 5) + int(math.log2(num_epochs / 50) * 3)
    else:
        patience = base_patience + int(math.log2(num_epochs / 10) * 5) + int(math.log2(num_epochs / 50) * 3) + int(math.log2(num_epochs / 100) * 2)

    patience = min(patience, base_patience * 3)
    return patience


class ModelEMA:
    """
    Model Exponential Moving Average (EMA)

    Maintains a smoothed version of model parameters during training.
    Using EMA weights during inference typically yields more stable predictions,
    expected +0.5~2% accuracy improvement.

    Usage:
        ema = ModelEMA(model, decay=0.999)
        for epoch in range(num_epochs):
            train_one_epoch(...)
            ema.update(model)           # call after each step
            ema.swap(model)             # before validation: switch to EMA weights
            validate(...)
            ema.swap(model)             # after validation: restore original weights
    """

    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self._register(model)

    def _register(self, model):
        for name, param in model.state_dict().items():
            if param.dtype in (torch.float32, torch.float16, torch.bfloat16):
                self.shadow[name] = param.clone().detach()

    def update(self, model):
        with torch.no_grad():
            for name, param in model.state_dict().items():
                if name in self.shadow:
                    new_average = self.decay * self.shadow[name] + (1.0 - self.decay) * param.detach()
                    self.shadow[name].copy_(new_average)

    def swap(self, model):
        with torch.no_grad():
            self.backup.clear()
            for name, param in model.state_dict().items():
                if name in self.shadow:
                    self.backup[name] = param.clone().detach()
                    param.copy_(self.shadow[name])
            for name in self.shadow:
                if name not in dict(model.state_dict()):
                    continue

    def restore(self, model):
        with torch.no_grad():
            for name, param in model.state_dict().items():
                if name in self.backup:
                    param.copy_(self.backup[name])
        self.backup.clear()

    def state_dict(self):
        return {'decay': self.decay, 'shadow': self.shadow}

    def load_state_dict(self, state_dict):
        self.decay = state_dict['decay']
        self.shadow = state_dict['shadow']
