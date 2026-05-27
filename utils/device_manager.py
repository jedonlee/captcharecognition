# -*- coding: utf-8 -*-
"""
Device management module
Function: auto-detect GPU availability and select the best device, support CPU/GPU auto-switching

Main features:
1. Auto-detect CUDA availability
2. Auto-select the best device (GPU preferred, CPU fallback)
3. Display device information
4. Check mixed precision training support
5. Provide device-related utility functions
"""

import torch
import logging
import warnings

logger = logging.getLogger(__name__)


class DeviceManager:
    """Device manager, responsible for auto-selecting and managing compute devices"""

    def __init__(self, prefer_gpu=True, verbose=True):
        """
        Initialize device manager

        Args:
            prefer_gpu (bool): whether to prefer GPU, default True
            verbose (bool): whether to print device info, default True
        """
        self.prefer_gpu = prefer_gpu
        self.verbose = verbose
        self.device = self._select_device()
        self._print_device_info()
    
    @staticmethod
    def get_device_simple(prefer_gpu=True, verbose=False):
        """
        Static method: get device simply (no instantiation needed)

        Args:
            prefer_gpu (bool): whether to prefer GPU
            verbose (bool): whether to print info

        Returns:
            torch.device: device object
        """
        if prefer_gpu and torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        if verbose:
            logger.info(f"Device: {device}")
        return device

    def _select_device(self):
        """
        Auto-select the best device

        Returns:
            torch.device: selected device object
        """
        if self.prefer_gpu and torch.cuda.is_available():
            # GPU available, prefer GPU
            device = torch.device('cuda')
        else:
            # GPU not available or not preferred, use CPU
            device = torch.device('cpu')

        return device

    def _print_device_info(self):
        """Print device information"""
        if not self.verbose:
            return

        logger.info("=" * 60)
        logger.info("Device Information")
        logger.info("=" * 60)
        logger.info(f"Current device: {self.device}")

        if self.device.type == 'cuda':
            logger.info(f"GPU count: {torch.cuda.device_count()}")
            logger.info(f"Current GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"CUDA version: {torch.version.cuda}")

            total_memory = torch.cuda.get_device_properties(0).total_memory
            allocated_memory = torch.cuda.memory_allocated(0)
            cached_memory = torch.cuda.memory_reserved(0)

            logger.info(f"Total GPU memory: {total_memory / 1024**3:.2f} GB")
            logger.info(f"Allocated memory: {allocated_memory / 1024**3:.2f} GB")
            logger.info(f"Cached memory: {cached_memory / 1024**3:.2f} GB")

            if self._supports_amp():
                logger.info("Mixed precision training: Supported (AMP)")
            else:
                logger.info("Mixed precision training: Not supported")
        else:
            logger.info("Using CPU mode")
            logger.info("Mixed precision training: Not supported (GPU only)")

        logger.info("=" * 60)

    def _supports_amp(self):
        """
        Check if mixed precision training (AMP) is supported

        Returns:
            bool: whether AMP is supported
        """
        # AMP only supported on CUDA devices
        if self.device.type != 'cuda':
            return False

        # Check CUDA availability
        if not torch.cuda.is_available():
            return False

        # Check for torch.amp module
        try:
            from torch.amp import autocast, GradScaler
            return True
        except ImportError:
            return False

    def get_device(self):
        """
        Get current device

        Returns:
            torch.device: current device object
        """
        return self.device

    def is_cuda(self):
        """
        Check if current device is CUDA

        Returns:
            bool: whether it's a CUDA device
        """
        return self.device.type == 'cuda'

    def is_cpu(self):
        """
        Check if current device is CPU

        Returns:
            bool: whether it's a CPU device
        """
        return self.device.type == 'cpu'

    def supports_amp(self):
        """
        Check if mixed precision training is supported

        Returns:
            bool: whether AMP is supported
        """
        return self._supports_amp()

    def get_amp_context(self):
        """
        Get mixed precision training context manager

        Returns:
            context manager or None: returns autocast() if AMP supported, otherwise None
        """
        if self.supports_amp():
            from torch.amp import autocast
            return autocast('cuda')
        else:
            # Return a null context manager
            from contextlib import nullcontext
            return nullcontext()

    def get_grad_scaler(self):
        """
        Get gradient scaler (for mixed precision training)

        Returns:
            GradScaler or None: returns GradScaler() if AMP supported, otherwise None
        """
        if self.supports_amp():
            from torch.amp import GradScaler
            return GradScaler('cuda')
        else:
            return None

    def empty_cache(self):
        """Clear GPU cache (only effective on GPU)"""
        if self.is_cuda():
            torch.cuda.empty_cache()

    def get_memory_info(self):
        """
        Get GPU memory info (only effective on GPU)

        Returns:
            dict: dict containing memory info, returns None on CPU
        """
        if not self.is_cuda():
            return None

        total_memory = torch.cuda.get_device_properties(0).total_memory
        allocated_memory = torch.cuda.memory_allocated(0)
        cached_memory = torch.cuda.memory_reserved(0)
        free_memory = total_memory - cached_memory

        return {
            'total': total_memory,
            'allocated': allocated_memory,
            'cached': cached_memory,
            'free': free_memory,
            'total_gb': total_memory / 1024**3,
            'allocated_gb': allocated_memory / 1024**3,
            'cached_gb': cached_memory / 1024**3,
            'free_gb': free_memory / 1024**3
        }

    def set_device(self, device):
        """
        Set device (use with caution, manual setting is not recommended)

        Args:
            device (torch.device or str): device to set
        """
        if isinstance(device, str):
            device = torch.device(device)

        self.device = device
        if self.verbose:
            logger.info(f"Device changed to: {self.device}")


def get_device(prefer_gpu=True, verbose=True):
    """
    Convenience function: get device

    Args:
        prefer_gpu (bool): whether to prefer GPU, default True
        verbose (bool): whether to print device info, default True

    Returns:
        torch.device: device object
    """
    manager = DeviceManager(prefer_gpu=prefer_gpu, verbose=verbose)
    return manager.get_device()


def get_device_manager(prefer_gpu=True, verbose=True):
    """
    Convenience function: get device manager instance

    Args:
        prefer_gpu (bool): whether to prefer GPU, default True
        verbose (bool): whether to print device info, default True

    Returns:
        DeviceManager: device manager instance
    """
    return DeviceManager(prefer_gpu=prefer_gpu, verbose=verbose)


def check_cuda_available():
    """
    Check CUDA availability

    Returns:
        bool: whether CUDA is available
    """
    return torch.cuda.is_available()


def get_gpu_count():
    """
    Get GPU count

    Returns:
        int: number of GPUs
    """
    return torch.cuda.device_count() if torch.cuda.is_available() else 0


def get_gpu_name(device_id=0):
    """
    Get GPU name

    Args:
        device_id (int): GPU device ID, default 0

    Returns:
        str: GPU name, or None if no GPU
    """
    if torch.cuda.is_available() and device_id < torch.cuda.device_count():
        return torch.cuda.get_device_name(device_id)
    return None


def supports_amp():
    """
    Check if system supports mixed precision training

    Returns:
        bool: whether AMP is supported
    """
    if not torch.cuda.is_available():
        return False

    try:
        from torch.amp import autocast, GradScaler
        return True
    except ImportError:
        return False


def print_device_summary():
    """Print device summary information"""
    logger.info("=" * 60)
    logger.info("Device Summary")
    logger.info("=" * 60)
    logger.info(f"CUDA available: {check_cuda_available()}")

    if check_cuda_available():
        gpu_count = get_gpu_count()
        logger.info(f"GPU count: {gpu_count}")

        for i in range(gpu_count):
            gpu_name = get_gpu_name(i)
            logger.info(f"  GPU {i}: {gpu_name}")

        logger.info(f"Mixed precision support: {supports_amp()}")
    else:
        logger.info("Using CPU mode")
        logger.info("Mixed precision: Not supported (GPU only)")

    logger.info("=" * 60)


if __name__ == '__main__':
    # Test code
    print("Testing device manager...")

    # Create device manager
    device_manager = DeviceManager()

    # Get device
    device = device_manager.get_device()
    print(f"\nSelected device: {device}")

    # Check device type
    print(f"Is CUDA: {device_manager.is_cuda()}")
    print(f"Is CPU: {device_manager.is_cpu()}")

    # Check AMP support
    print(f"AMP supported: {device_manager.supports_amp()}")

    # Get memory info (if using GPU)
    if device_manager.is_cuda():
        memory_info = device_manager.get_memory_info()
        if memory_info:
            print(f"\nGPU Memory Info:")
            print(f"  Total: {memory_info['total_gb']:.2f} GB")
            print(f"  Allocated: {memory_info['allocated_gb']:.2f} GB")
            print(f"  Cached: {memory_info['cached_gb']:.2f} GB")
            print(f"  Free: {memory_info['free_gb']:.2f} GB")

    # Test convenience functions
    print("\nTesting convenience functions...")
    print_device_summary()
