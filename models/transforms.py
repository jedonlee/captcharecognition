# -*- coding: utf-8 -*-
"""
Data augmentation config (supports albumentations + document degradation)
Function: Improve training efficiency with geometric/color/noise/document augmentation

Augmentation strategies:
  - Priority: albumentations library (top GitHub project)
  - Supports document degradation: BadPhotoCopy, DirtyDrum, Letterpress, Markup, WaterMark etc.
  - Supports preprocessing: grayscale + OTSU binarization + median blur
  - Automatic fallback to torchvision.transforms when unavailable

Normalization strategy:
  - All paths (train/val/albumentations/torchvision) use ImageNet normalization
    mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225)

Config source: config.yaml (image size + augmentation config)
"""

import logging
import warnings
from typing import Optional, Tuple, Union

from utils.config_loader import get_config

logger = logging.getLogger(__name__)

# Attempt to import albumentations library
ALBUMENTATIONS_AVAILABLE = False
ALBUMENTATIONS_VERSION = None

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    ALBUMENTATIONS_AVAILABLE = True
    ALBUMENTATIONS_VERSION = A.__version__
    logger.info(f"Successfully loaded albumentations library, version: {ALBUMENTATIONS_VERSION}")
except ImportError:
    warnings.warn(
        "albumentations library is not installed or failed to import, falling back to torchvision.transforms. "
        "Install albumentations for better augmentation: pip install albumentations>=2.0.0",
        ImportWarning
    )
    ALBUMENTATIONS_AVAILABLE = False
    ToTensorV2 = None
    logger.warning("albumentations library not available, using torchvision.transforms fallback")

# Import torchvision.transforms as fallback
import torchvision.transforms as T
import torch
import numpy as np
import cv2


def _check_albumentations_version(min_version: str = "2.0.0") -> bool:
    """
    Check if albumentations version meets minimum requirements

    Args:
        min_version: minimum required version, default "2.0.0"

    Returns:
        bool: whether version meets requirements
    """
    if not ALBUMENTATIONS_AVAILABLE or ALBUMENTATIONS_VERSION is None:
        return False

    try:
        from packaging import version
        current_version = version.parse(ALBUMENTATIONS_VERSION)
        required_version = version.parse(min_version)
        return current_version >= required_version
    except ImportError:
        warnings.warn(
            "packaging library not installed, cannot precisely check albumentations version. "
            "Install with: pip install packaging",
            ImportWarning
        )
        return True
    except Exception as e:
        warnings.warn(f"Error checking albumentations version: {e}", RuntimeWarning)
        return True


def apply_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Apply grayscale conversion

    Args:
        image: input image (H, W, C)

    Returns:
        grayscale image
    """
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        # Convert back to 3-channel for pretrained model compatibility
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    return image


def apply_otsu_binarization(image: np.ndarray) -> np.ndarray:
    """
    OTSU adaptive binarization

    Args:
        image: input image

    Returns:
        binarized image
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    # OTSU binarization
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if len(image.shape) == 3:
        binary = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)

    return binary


def apply_median_blur(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    Median blur denoising

    Args:
        image: input image
        kernel_size: kernel size

    Returns:
        denoised image
    """
    return cv2.medianBlur(image, kernel_size)


def get_document_degradation_transforms(config):
    """
    Get document degradation transform list

    Args:
        config: config object

    Returns:
        list of document degradation transforms
    """
    doc_config = config.get_document_degradation_config()
    if not doc_config.get('enabled', False):
        return []

    intensity = doc_config.get('intensity', 0.5)
    transforms = []

    # Check if document degradation is available (some require extensions)
    try:
        # Attempt to add BadPhotoCopy (simulates photocopy effect)
        if doc_config.get('bad_photo_copy', {}).get('enabled', True):
            p = doc_config.get('bad_photo_copy', {}).get('p', 0.2) * intensity
            if hasattr(A, 'BadPhotoCopy'):
                transforms.append(A.BadPhotoCopy(alpha=(0.2, 0.6), noise_type=1, p=p))
    except Exception:
        pass

    try:
        # Attempt to add DirtyDrum (dirty drum effect)
        if doc_config.get('dirty_drum', {}).get('enabled', True):
            p = doc_config.get('dirty_drum', {}).get('p', 0.2) * intensity
            if hasattr(A, 'DirtyDrum'):
                transforms.append(A.DirtyDrum(p=p))
    except Exception:
        pass

    try:
        # Attempt to add Letterpress (letterpress effect)
        if doc_config.get('letterpress', {}).get('enabled', True):
            p = doc_config.get('letterpress', {}).get('p', 0.15) * intensity
            if hasattr(A, 'Letterpress'):
                transforms.append(A.Letterpress(p=p))
    except Exception:
        pass

    try:
        # Attempt to add Markup (markup effect)
        if doc_config.get('markup', {}).get('enabled', True):
            p = doc_config.get('markup', {}).get('p', 0.15) * intensity
            if hasattr(A, 'Markup'):
                transforms.append(A.Markup(p=p))
    except Exception:
        pass

    try:
        # Attempt to add WaterMark (watermark effect)
        if doc_config.get('watermark', {}).get('enabled', True):
            p = doc_config.get('watermark', {}).get('p', 0.1) * intensity
            if hasattr(A, 'WaterMark'):
                transforms.append(A.WaterMark(p=p))
    except Exception:
        pass

    try:
        # Attempt to add BookBinding (book binding effect)
        if doc_config.get('book_binding', {}).get('enabled', True):
            p = doc_config.get('book_binding', {}).get('p', 0.1) * intensity
            if hasattr(A, 'BookBinding'):
                transforms.append(A.BookBinding(p=p))
    except Exception:
        pass

    try:
        # Attempt to add Folding (folding effect)
        if doc_config.get('folding', {}).get('enabled', True):
            p = doc_config.get('folding', {}).get('p', 0.1) * intensity
            if hasattr(A, 'Folding'):
                transforms.append(A.Folding(p=p))
    except Exception:
        pass

    try:
        # Attempt to add SectionShift (section shift effect)
        if doc_config.get('section_shift', {}).get('enabled', True):
            p = doc_config.get('section_shift', {}).get('p', 0.1) * intensity
            if hasattr(A, 'SectionShift'):
                transforms.append(A.SectionShift(p=p))
    except Exception:
        pass

    return transforms


def get_train_transform(image_height: int = None, image_width: int = None) -> Union[A.Compose, T.Compose]:
    """
    Get training data augmentation transforms

    Prefers albumentations for more professional data augmentation

    Optimization strategy (2026-04-26):
    - Disable strong geometric augmentation, keep only weak affine
    - Disable color augmentation (equivalent to adding noise on grayscale)
    - Use BORDER_CONSTANT fill (no ghosting)

    Args:
        image_height: image height, default from config.yaml
        image_width: image width, default from config.yaml

    Returns:
        data augmentation transform object
    """
    config = get_config()

    # Get image size from config
    if image_height is None or image_width is None:
        image_height, image_width = config.get_preprocessed_image_size()

    # Get augmentation config
    aug_config = config.get('augmentation', {}).get('train', {})

    # Plan A: use albumentations (recommended)
    if ALBUMENTATIONS_AVAILABLE and _check_albumentations_version():
        try:
            transform_list = [
                # Resize to unified size
                A.Resize(height=image_height, width=image_width, interpolation=cv2.INTER_LINEAR),
            ]

            # Add geometric augmentation (weakened, Affine only)
            if aug_config.get('enabled', True):
                affine_cfg = aug_config.get('affine', {})
                shear_val = affine_cfg.get('shear_limit', 0)
                transform_list.append(A.Affine(
                    translate_percent=affine_cfg.get('translate_percent', 0.05),
                    scale=affine_cfg.get('scale_limit', (0.95, 1.05)),
                    rotate=affine_cfg.get('rotate_limit', 5),
                    shear=shear_val,
                    p=affine_cfg.get('p', 0.5)
                ))

            # Add brightness contrast
            bc_cfg = aug_config.get('brightness_contrast', {})
            bc_p = bc_cfg.get('p', 0.5)
            if bc_p > 0:
                transform_list.append(A.RandomBrightnessContrast(
                    brightness_limit=bc_cfg.get('brightness_limit', 0.15),
                    contrast_limit=bc_cfg.get('contrast_limit', 0.15),
                    p=bc_p,
                ))

            # Add Gaussian noise
            noise_cfg = aug_config.get('gauss_noise', {})
            noise_limit = noise_cfg.get('var_limit', (5.0, 25.0))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    transform_list.append(A.GaussNoise(var_limit=noise_limit, p=noise_cfg.get('p', 0.3)))
                except Exception:
                    pass

            # Add Gaussian blur
            blur_cfg = aug_config.get('gaussian_blur', {})
            blur_limit = blur_cfg.get('blur_limit', (3, 5))
            transform_list.append(A.GaussianBlur(blur_limit=blur_limit, p=blur_cfg.get('p', 0.2)))

            # Add random erasing (simulates character occlusion) — CoarseDropout as alternative to RandomErasing
            # RandomErasing unavailable in albumentations <1.4, use CoarseDropout for equivalent effect
            erasing_cfg = aug_config.get('random_erasing', {})
            erasing_p = erasing_cfg.get('p', 0.3)
            if erasing_p > 0:
                try:
                    if hasattr(A, 'RandomErasing'):
                        transform_list.append(A.RandomErasing(
                            scale=erasing_cfg.get('scale', (0.02, 0.15)),
                            ratio=erasing_cfg.get('ratio', (0.3, 3.0)),
                            p=erasing_p,
                        ))
                    else:
                        # Use CoarseDropout to simulate erasing effect
                        transform_list.append(A.CoarseDropout(
                            num_holes_range=(1, 4),
                            hole_height_range=(4, 8),
                            hole_width_range=(4, 8),
                            fill=0,
                            p=erasing_p * 0.6,
                        ))
                except Exception:
                    pass

            # Add CoarseDropout (simulates coarse occlusion, uses albumentations 2.x API)
            cd_cfg = aug_config.get('coarse_dropout', {})
            cd_p = cd_cfg.get('p', 0.2)
            if cd_p > 0:
                transform_list.append(A.CoarseDropout(
                    num_holes_range=(1, cd_cfg.get('max_holes', 4)),
                    hole_height_range=(4, cd_cfg.get('max_height', 12)),
                    hole_width_range=(4, cd_cfg.get('max_width', 12)),
                    fill=0,
                    p=cd_p,
                ))

            # Add GridDistortion
            gd_cfg = aug_config.get('grid_distortion', {})
            gd_p = gd_cfg.get('p', 0.3)
            if gd_p > 0:
                transform_list.append(A.GridDistortion(
                    num_steps=gd_cfg.get('num_steps', 5),
                    distort_limit=gd_cfg.get('distort_limit', 0.3),
                    p=gd_p,
                ))

            # Normalization
            transform_list.append(A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)
            ))

            # ToTensorV2
            transform_list.append(ToTensorV2())

            transform = A.Compose(transform_list)
            return transform

        except Exception as e:
            warnings.warn(f"Error creating transform with albumentations: {e}, falling back to torchvision.transforms", RuntimeWarning)

    # Plan B: use torchvision.transforms
    try:
        transform_list = [
            T.Resize((image_height, image_width)),
            T.RandomRotation(degrees=10),
            T.ColorJitter(brightness=0.2, contrast=0.2),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
        transform = T.Compose(transform_list)
        return transform
    except Exception as e:
        raise RuntimeError(f"Failed to create data augmentation transform: {e}")


def get_val_transform(image_height: int = None, image_width: int = None) -> Union[A.Compose, T.Compose]:
    """
    Get validation/test transform (normalization only, no augmentation)

    Args:
        image_height: image height
        image_width: image width

    Returns:
        transform object
    """
    config = get_config()

    if image_height is None or image_width is None:
        image_height, image_width = config.get_preprocessed_image_size()

    # Plan A: use albumentations
    if ALBUMENTATIONS_AVAILABLE and _check_albumentations_version():
        try:
            transform_list = [
                A.Resize(height=image_height, width=image_width, interpolation=cv2.INTER_LINEAR),
            ]

            # Normalization
            transform_list.append(A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)
            ))

            transform_list.append(ToTensorV2())

            transform = A.Compose(transform_list)
            return transform

        except Exception as e:
            warnings.warn(f"Error creating validation transform with albumentations: {e}, falling back to torchvision.transforms", RuntimeWarning)

    # Plan B: use torchvision.transforms
    try:
        transform = T.Compose([
            T.Resize((image_height, image_width)),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
        return transform
    except Exception as e:
        raise RuntimeError(f"Failed to create validation transform: {e}")


def get_transform_info() -> dict:
    """
    Get current data augmentation config info

    Returns:
        dict: dictionary with data augmentation config info
    """
    info = {
        "albumentations_available": ALBUMENTATIONS_AVAILABLE,
        "albumentations_version": ALBUMENTATIONS_VERSION,
        "version_check_passed": _check_albumentations_version() if ALBUMENTATIONS_AVAILABLE else False,
        "using_albumentations": ALBUMENTATIONS_AVAILABLE and _check_albumentations_version()
    }
    return info


if __name__ == "__main__":
    """Test data augmentation functionality"""
    print("=" * 60)
    print("Data Augmentation Configuration Test")
    print("=" * 60)
    
    info = get_transform_info()
    print(f"\n[Configuration Info]")
    print(f"  albumentations available: {info['albumentations_available']}")
    print(f"  albumentations version: {info['albumentations_version']}")
    print(f"  Version check passed: {info['version_check_passed']}")
    print(f"  Using albumentations: {info['using_albumentations']}")
    
    print(f"\n[Testing Train Transform]")
    try:
        train_transform = get_train_transform()
        print(f"  Train transform created: {type(train_transform).__name__}")
    except Exception as e:
        print(f"  Train transform failed: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n[Testing Val Transform]")
    try:
        val_transform = get_val_transform()
        print(f"  Val transform created: {type(val_transform).__name__}")
    except Exception as e:
        print(f"  Val transform failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("Test complete")
    print("=" * 60)


get_train_transforms = get_train_transform
get_val_transforms = get_val_transform
