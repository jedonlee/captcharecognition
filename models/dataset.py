# -*- coding: utf-8 -*-
"""
CAPTCHA Dataset Classes (Simplified)
Function: Read images from folder + companion .txt labels -> tensor + label indices (62 classes + blank=63)

Dual-mode data sources:
  Scenario A: Preprocessed/augmented data (uint8 PNG, 32x128 RGB)
    -> Skip normalization, keep uint8 for unified transform processing
  Scenario B: Raw unprocessed data
    -> unified_preprocess_for_deep_learning() online preprocessing -> [0,1] float32

Augmentation:
  Priority: albumentations (with ImageNet normalization), fallback to torchvision.transforms

Dependencies: CharMapper / config_loader / preprocess.unified_preprocess_for_deep_learning
"""

import os
import tempfile
import torch
import numpy as np
import cv2
from PIL import Image
from torch.utils.data import Dataset
from collections import OrderedDict
import gc
from typing import Dict, List, Tuple, Optional, Any
import threading
import weakref

from utils.chars import CharMapper
import logging
from utils.config_loader import get_config
from preprocess import unified_preprocess_for_deep_learning
from generate.generate_dataset import CaptchaGenerator

logger = logging.getLogger(__name__)

try:
    import albumentations as A
    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    ALBUMENTATIONS_AVAILABLE = False


def check_label_file_exists(image_dir, image_file):
    """
    Check if label file exists

    Returns:
        tuple: (exists, label_path, error_message)
    """
    label_file = image_file.replace('.png', '.txt')
    label_path = os.path.join(image_dir, label_file)

    if not os.path.exists(label_path):
        error_msg = f"Label file not found: {label_path}\n" \
                   f"Suggestion: Check if image file {image_file} has a corresponding label file"
        return False, label_path, error_msg

    return True, label_path, None


def validate_label_format(label_text, characters, max_length=6):
    """
    Validate label format correctness

    Returns:
        tuple: (is_valid, error_message)
    """
    if not label_text:
        return False, "Label content is empty\nSuggestion: Check if the label file contains valid text"

    if len(label_text) > max_length:
        return False, f"Label length exceeds limit: {len(label_text)} > {max_length}\n" \
                     f"Suggestion: Ensure label length does not exceed {max_length} characters"

    for i, char in enumerate(label_text):
        if char not in characters:
            return False, f"Label contains invalid character: '{char}' (position: {i+1})\n" \
                         f"Allowed characters: {characters}\n" \
                         f"Suggestion: Check the label file to ensure it only contains these characters"

    return True, None


def handle_label_error(image_file, error_type, error_msg):
    """
    Handle label error and provide recovery suggestions

    Returns:
        dict: dictionary containing error details and recovery suggestions
    """
    recovery_suggestions = {
        'file_not_found': [
            "1. Check if image and label filenames match (e.g., image.png corresponds to image.txt)",
            "2. Confirm the label file is in the same directory",
            "3. If the label file is missing, create it manually or regenerate the dataset",
            "4. Run the dataset check script: python check_data.py"
        ],
        'invalid_format': [
            "1. Check if the label file content is plain text",
            "2. Ensure the label only contains letters and digits (A-Z, a-z, 0-9)",
            "3. Label length should not exceed 6 characters",
            "4. Remove spaces, newlines, or other special characters from the label",
            "5. Open the label file with a text editor to check content"
        ],
        'other': [
            "1. Check if file permissions are correct",
            "2. Confirm the file encoding is UTF-8",
            "3. Try regenerating the label file",
            "4. View the full error log for more information"
        ]
    }

    suggestions = recovery_suggestions.get(error_type, recovery_suggestions['other'])

    error_info = {
        'image_file': image_file,
        'error_type': error_type,
        'error_message': error_msg,
        'recovery_suggestions': suggestions
    }

    return error_info


class CaptchaDataset(Dataset):
    """
    CAPTCHA Dataset (dual-mode + auto augmentation fallback)

    Core features:
    - Auto-detect Scenario A/B via .preprocessed_flag file
    - Dual transform engine: albumentations (primary) / torchvision.transforms (fallback)
    - Unified character mapping via CharMapper singleton
    - Output format: dict {image:(3,32,128), label_indices:(6,), label_length, label_text}
    """

    def __init__(self, image_dir=None, transform=None, max_length=None, cache_size=100, use_cache=True, data_dir=None):
        """
        Initialize dataset (memory-optimized, thread-safe)

        Args:
            image_dir: image directory (primary param)
            data_dir: image directory (backward compat, equivalent to image_dir)
            transform: data augmentation transforms
            max_length: max sequence length (default from config)
            cache_size: LRU cache size (default 100 images)
            use_cache: whether to use cache (default True, False for fully lazy loading)
        """
        # Determine image directory (prefer image_dir, fallback to data_dir)
        final_image_dir = None
        if image_dir is not None:
            final_image_dir = image_dir
        elif data_dir is not None:
            final_image_dir = data_dir
        
        # Strict null check
        if final_image_dir is None:
            raise ValueError(
                "Either image_dir or data_dir parameter must be provided\n"
                "Check data.train_dir and data.val_dir in config.yaml"
            )
        if not isinstance(final_image_dir, (str, bytes, os.PathLike)):
            raise TypeError(
                f"image_dir/data_dir must be str, bytes, or PathLike object, got: {type(final_image_dir)}\n"
                f"Value: {final_image_dir}"
            )
        
        self.image_dir = final_image_dir
        
        # Verify directory exists (friendly error)
        if not os.path.exists(self.image_dir):
            raise FileNotFoundError(
                f"Dataset directory not found: {self.image_dir}\n"
                "Please ensure:\n"
                "1. Data generation, cleaning, splitting, and preprocessing steps have been run\n"
                "2. Path configuration in config.yaml is correct\n"
                "3. Directory permissions are correct"
            )
        if not os.path.isdir(self.image_dir):
            raise NotADirectoryError(
                f"Path is not a directory: {self.image_dir}\n"
                f"Check path configuration in config.yaml"
            )
        
        self.transform = transform
        self.cache_size = cache_size
        self.use_cache = use_cache

        # Load charset config from unified config file
        config = get_config()
        chars_config = config.get_chars_config()

        # Use charset from config
        self.characters = chars_config['characters']
        self.num_classes = chars_config['num_classes']
        self.blank_index = chars_config['blank_index']

        # Use max_length from config, or default if unspecified
        self.max_length = max_length if max_length is not None else chars_config['max_length']

        # Use unified CharMapper (single source of truth for char mapping)
        # Replaces manual char_to_idx / idx_to_char dicts + LabelEncoder
        self.char_mapper = CharMapper.get_instance()
        self.characters = self.char_mapper.characters
        self.num_classes = self.char_mapper.num_classes
        self.blank_index = self.char_mapper.blank_index
        self.char_to_idx = self.char_mapper.char_to_idx
        self.idx_to_char = self.char_mapper.idx_to_char

        # Get all image files (lazy load: only filenames, no image data)
        self.image_files = [f for f in os.listdir(self.image_dir) if f.endswith('.png')]

        # Filter out images without matching labels, validate label format
        self.valid_samples = []
        self.invalid_samples = []

        for image_file in self.image_files:
            # Check if label file exists
            exists, label_path, error_msg = check_label_file_exists(self.image_dir, image_file)

            if not exists:
                error_info = handle_label_error(image_file, 'file_not_found', error_msg)
                self.invalid_samples.append(error_info)
                continue

            # Read and validate label format (text only, no image loading)
            try:
                with open(label_path, 'r', encoding='utf-8') as f:
                    label_text = f.read().strip()

                is_valid, format_error = validate_label_format(label_text, self.characters, self.max_length)

                if not is_valid:
                    error_info = handle_label_error(image_file, 'invalid_format', format_error)
                    self.invalid_samples.append(error_info)
                    continue

                # Save sample info (filename and label text, no image loaded)
                self.valid_samples.append({
                    'image_file': image_file,
                    'label_text': label_text
                })
            except Exception as e:
                error_msg = f"Error reading label file: {str(e)}"
                error_info = handle_label_error(image_file, 'other', error_msg)
                self.invalid_samples.append(error_info)
                continue

        # Initialize thread-safe LRU cache for loaded images
        # Thread lock ensures multi-thread safety
        self._cache_lock = threading.Lock() if use_cache else None
        self.image_cache = OrderedDict() if use_cache else None

        # Use weak references to track loaded images, avoid memory leaks
        self._weak_image_refs = weakref.WeakValueDictionary() if use_cache else None

        logger.info(f"Dataset initialized (memory-optimized, thread-safe):")
        logger.info(f"  Image directory: {image_dir}")
        logger.info(f"  Total images: {len(self.image_files)}")
        logger.info(f"  Valid samples: {len(self.valid_samples)}")
        logger.info(f"  Invalid samples: {len(self.invalid_samples)}")
        logger.info(f"  Characters: {self.characters}")
        logger.info(f"  Num classes: {self.num_classes}")
        logger.info(f"  Blank index: {self.blank_index}")
        logger.info(f"  Max length: {self.max_length}")
        logger.info(f"  Cache size: {cache_size if use_cache else 'disabled'}")
        logger.info(f"  Lazy loading: {'disabled' if not use_cache else 'partial (LRU cache)'}")
        logger.info(f"  Thread safety: {'enabled' if use_cache else 'not needed (cache disabled)'}")

        if self.invalid_samples:
            logger.warning(f"Found {len(self.invalid_samples)} invalid samples:")
            error_type_counts = {}
            for error_info in self.invalid_samples:
                error_type = error_info['error_type']
                error_type_counts[error_type] = error_type_counts.get(error_type, 0) + 1

            for error_type, count in error_type_counts.items():
                logger.warning(f"  - {error_type}: {count}")

            logger.warning(f"First 3 invalid sample details:")
            for i, error_info in enumerate(self.invalid_samples[:3]):
                logger.warning(f"  [{i+1}] File: {error_info['image_file']}")
                logger.warning(f"      Error type: {error_info['error_type']}")
                logger.warning(f"      Error message: {error_info['error_message']}")
                logger.warning(f"      Recovery suggestions:")
                for suggestion in error_info['recovery_suggestions'][:2]:
                    logger.warning(f"        {suggestion}")

        # 🔧 2026-04-06 P0fix: detect data source via marker file (replace fragile folder-name detection)
        # Detection rule: .preprocessed_flag exists in target dir -> preprocessed
        #                 No such file -> raw data (requires full unified_preprocess_for_deep_learning)
        # Advantage: independent of folder name, renaming won't misclassify
        self._is_preprocessed_data = self._detect_preprocessed_source(self.image_dir)
        source_type = "Preprocessed (offline)" if self._is_preprocessed_data else "Raw data (online)"
        logger.info(f"  Data source type: {source_type}")
        logger.info(f"  Preprocessing strategy: {'BGR->RGB+Normalize' if self._is_preprocessed_data else 'Full online preprocessing (RGB+CLAHE+Normalize)'}")

    def _detect_preprocessed_source(self, image_dir):
        """
        Detect if data source has been preprocessed (via marker file detection)

        Detection rules:
        - .preprocessed_flag exists in target dir or parent dir -> preprocessed
        - No such file -> raw data

        Args:
            image_dir: dataset directory path

        Returns:
            bool: True if preprocessed, False if raw data
        """
        import os

        # Strategy 1: check directly in target directory
        flag_file = os.path.join(image_dir, '.preprocessed_flag')
        if os.path.isfile(flag_file):
            return True

        # Strategy 2: check up to 2 levels of parent dirs (covers augmented_train/ under preprocessed/)
        current_dir = image_dir
        for _ in range(3):
            parent_dir = os.path.dirname(current_dir)
            if parent_dir == current_dir:
                break
            flag_file = os.path.join(parent_dir, '.preprocessed_flag')
            if os.path.isfile(flag_file):
                return True
            current_dir = parent_dir

        return False

    def _load_image(self, image_file):
        """
        Load image with caching mechanism (thread-safe)

        Args:
            image_file: image filename

        Returns:
            image: image numpy array
        """
        # If cache enabled, check cache first (thread-safe)
        if self.use_cache and self.image_cache is not None:
            with self._cache_lock:
                if image_file in self.image_cache:
                    # Cache hit, move to LRU end (most recently used)
                    self.image_cache.move_to_end(image_file)
                    cached_image = self.image_cache[image_file]
                    # Return a copy to prevent external modifications
                    return cached_image.copy()

        # Load image (lazy loading, on-demand)
        image_path = os.path.join(self.image_dir, image_file)

        try:
            # Use context manager for timely file handle closure
            with Image.open(image_path) as img:
                image = np.array(img.convert('RGB'))

            # If cache enabled, add to cache (thread-safe)
            if self.use_cache and self.image_cache is not None:
                with self._cache_lock:
                    # Re-check cache status (may have been modified by another thread during loading)
                    if len(self.image_cache) < self.cache_size or image_file not in self.image_cache:
                        self.image_cache[image_file] = image
                        # Use weak reference to track image object, avoid memory leaks
                        self._weak_image_refs[image_file] = image
                        # Evict LRU item if cache exceeds size limit
                        while len(self.image_cache) > self.cache_size:
                            removed_key, removed_image = self.image_cache.popitem(last=False)
                            # Remove from weak reference dict
                            if removed_key in self._weak_image_refs:
                                del self._weak_image_refs[removed_key]
                            # Explicitly delete reference to aid garbage collection
                            del removed_image
        except FileNotFoundError:
            logger.warning(f"Image file not found: {image_path}")
            image = np.zeros((64, 256, 3), dtype=np.uint8)
        except Exception as e:
            logger.error(f"Error loading image {image_file}: {str(e)}")
            # Return blank image as fallback
            image = np.zeros((64, 256, 3), dtype=np.uint8)

        return image

    def clear_cache(self):
        """Clear image cache and free memory (thread-safe)"""
        if self.image_cache is not None:
            with self._cache_lock:
                # Clear cache
                self.image_cache.clear()
                # Clear weak reference dict
                if self._weak_image_refs is not None:
                    self._weak_image_refs.clear()
            # Trigger garbage collection
            gc.collect()
            logger.info(f"Image cache cleared, memory released")

    def get_cache_stats(self):
        """
        Get cache statistics (thread-safe)

        Returns:
            dict: cache statistics
        """
        if not self.use_cache or self.image_cache is None:
            return {
                'enabled': False,
                'size': 0,
                'max_size': 0,
                'hit_rate': 0.0
            }

        # Thread lock for safe cache size reading
        with self._cache_lock:
            return {
                'enabled': True,
                'size': len(self.image_cache),
                'max_size': self.cache_size,
                'usage_rate': len(self.image_cache) / self.cache_size
            }

    def __len__(self):
        """Return dataset size"""
        return len(self.valid_samples)

    def __getitem__(self, idx):
        """
        Get single sample (memory-optimized version)

        Args:
            idx: index

        Returns:
            image: image tensor (C, H, W)
            label_indices: label index list
            label_length: label length
        """
        try:
            # Get sample info (filename + label text)
            sample_info = self.valid_samples[idx]
            image_file = sample_info['image_file']
            label_text = sample_info['label_text']

            # Build label path
            label_path = os.path.join(self.image_dir, image_file.replace('.png', '.txt'))

            # Re-check label file existence (defensive programming)
            exists, _, error_msg = check_label_file_exists(self.image_dir, image_file)
            if not exists:
                raise FileNotFoundError(error_msg)

            # Load image (force RGB, ensure 3 channels)
            image = self._load_image(image_file)
            
            # Ensure image is 3-channel (unified format)
            if len(image.shape) == 2:  # Grayscale (H, W)
                image = np.stack([image] * 3, axis=-1)  # Convert to (H, W, 3)
            elif len(image.shape) == 3 and image.shape[2] == 1:  # Single channel (H, W, 1)
                image = np.repeat(image, 3, axis=2)  # Convert to (H, W, 3)
            elif len(image.shape) == 3 and image.shape[2] != 3:
                # Other channel count, truncate to 3 or pad
                if image.shape[2] > 3:
                    image = image[:, :, :3]
                else:
                    image = np.repeat(image, 3 // image.shape[2] + 1, axis=2)[:, :, :3]
            
            # 🔧 2026-04-06 P0fix: select preprocessing strategy based on data source type, avoid double preprocessing
            # Root cause: preprocess_dataset.py (offline) and preprocess.py (online) use different pipelines
            #   - Offline: grayscale→binarization(OTSU)→median→gaussian→morph→CLAHE(grayscale)→BGR→resize
            #   - Online: BGR→RGB→CLAHE(YUV-Y channel)→resize→normalize[0,1]
            #   Applying CLAHE on already-binarized images produces unpredictable results
            if self._is_preprocessed_data:
                # Scenario A: data preprocessed by preprocess_dataset.py (offline) + augment.py
                # Data format: 32×128 uint8 RGB (PNG files, cv2.imread loads as uint8[0,255])
                # Strategy: keep uint8 format, let transform pipeline handle normalization+standardization
                
                # Step 1: BGR → RGB (OpenCV reads as BGR by default)
                if len(image.shape) == 3 and image.shape[2] == 3:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Step 2: Ensure correct size (defensive check + fallback)
                if image.shape[0] != 64 or image.shape[1] != 256:
                    import warnings
                    warnings.warn(
                        f"[Dataset] Image size mismatch: got {image.shape[:2]}, expected (64, 256). "
                        f"Auto-resizing to (64, 256). This may indicate a corrupted sample.",
                        UserWarning
                    )
                    image = cv2.resize(image, (256, 64), interpolation=cv2.INTER_LINEAR)
                
                # Keep uint8 [0,255] format, do not normalize here
                # Normalization and ImageNet standardization handled by transforms.py pipeline
                
            else:
                # Scenario B: raw data (split_captchas/ or raw_captchas/)
                # Full online preprocessing pipeline (RGB space + CLAHE + normalization)
                image = unified_preprocess_for_deep_learning(
                    image,
                    target_height=64,
                    target_width=256,
                    apply_clahe=True
                )
            
            # Convert to torch tensor and enforce (C, H, W) format (3, 64, 256)
            # Use .clone() to ensure tensor is resizable
            image = torch.from_numpy(image).permute(2, 0, 1).float().clone()
            
            # Force dimension validation (ensure all images are identical shape)
            assert image.shape == (3, 64, 256), f"Image dimension error: {image.shape}, expected: (3, 64, 256)"

            # Validate label format (defensive programming)
            is_valid, format_error = validate_label_format(label_text, self.characters, self.max_length)
            if not is_valid:
                raise ValueError(format_error)

            # Convert to character list
            chars = list(label_text)

            # Encode labels to indices (direct char_to_idx mapping)
            label_indices = [self.char_to_idx[char] for char in chars]

            # Strictly validate index range (0 ≤ idx < total_classes)
            total_classes = self.num_classes + 1  # Includes blank token
            for idx in label_indices:
                assert 0 <= idx < total_classes, f"Label index out of range: {idx}, valid range is [0, {total_classes-1}]"

            label_indices = torch.tensor(label_indices, dtype=torch.long).clone()

            # Label length
            label_length = len(label_indices)

            # Apply data augmentation (supports albumentations and torchvision.transforms)
            if self.transform:
                # Detect transform type and handle accordingly
                transform_type = type(self.transform).__module__
                image_numpy = image.permute(1, 2, 0).numpy()

                if 'albumentations' in transform_type and ALBUMENTATIONS_AVAILABLE:
                    # albumentations requires uint8[0,255] input (internally divides by 255 then normalizes)
                    if self._is_preprocessed_data:
                        image_numpy = image_numpy.astype(np.uint8)  # Scenario A: float32[0,255]→uint8
                    else:
                        image_numpy = (image_numpy * 255.0).astype(np.uint8)  # Scenario B: [0,1]→uint8
                    logger.debug(f"Albumentations input shape: HWC {image_numpy.shape}")

                    augmented = self.transform(image=image_numpy)
                    image = augmented['image']
                    logger.debug(f"Albumentations output shape: CHW {image.shape}")

                    if isinstance(image, torch.Tensor):
                        image = image.contiguous()
                    image = image.float()
                else:
                    # torchvision.transforms requires PIL Image input
                    if self._is_preprocessed_data:
                        image_numpy = image_numpy.astype(np.uint8)  # Scenario A: float32[0,255]→uint8
                    else:
                        image_numpy = (image_numpy * 255.0).astype(np.uint8)  # Scenario B: [0,1]→uint8
                    pil_image = Image.fromarray(image_numpy)
                    image = self.transform(pil_image)
                    if isinstance(image, torch.Tensor):
                        image = image.contiguous()
                    if isinstance(image, torch.Tensor) and image.dtype != torch.float32:
                        image = image.float()

            # Ensure tensor is contiguous
            if isinstance(image, torch.Tensor) and not image.is_contiguous():
                image = image.contiguous()

            # Re-validate dimensions after transform
            assert image.shape == (3, 64, 256), f"Image dimension error after augmentation: {image.shape}, expected: (3, 64, 256)"

            return {
                'image': image.contiguous() if isinstance(image, torch.Tensor) else image,
                'label_indices': label_indices.contiguous() if isinstance(label_indices, torch.Tensor) else label_indices,
                'label_length': label_length,
                'label_text': label_text
            }
        except FileNotFoundError as e:
            logger.error(f"Error: Label file missing for sample {idx}")
            logger.error(f"Error details: {str(e)}")
            error_info = handle_label_error(self.valid_samples[idx]['image_file'] if idx < len(self.valid_samples) else 'unknown', 'file_not_found', str(e))
            logger.error(f"Recovery suggestions:")
            for suggestion in error_info['recovery_suggestions'][:2]:
                logger.error(f"  {suggestion}")
            return self._get_dummy_sample()
        except ValueError as e:
            logger.error(f"Error: Invalid label format for sample {idx}")
            logger.error(f"Error details: {str(e)}")
            error_info = handle_label_error(self.valid_samples[idx]['image_file'] if idx < len(self.valid_samples) else 'unknown', 'invalid_format', str(e))
            logger.error(f"Recovery suggestions:")
            for suggestion in error_info['recovery_suggestions'][:2]:
                logger.error(f"  {suggestion}")
            return self._get_dummy_sample()
        except Exception as e:
            logger.warning(f"Warning: Error loading sample at index {idx}: {str(e)}")
            return self._get_dummy_sample()

    def _get_dummy_sample(self):
        """
        Return dummy sample for error recovery

        Returns:
            dict: dictionary with dummy data
        """
        # Get image size from config
        config = get_config()
        preprocessed_height, preprocessed_width = config.get_preprocessed_image_size()

        dummy_image = torch.zeros((3, preprocessed_height, preprocessed_width), dtype=torch.float32)
        dummy_label_indices = torch.zeros((6,), dtype=torch.long)
        dummy_label_length = 0
        dummy_label_text = ""
        return {
            'image': dummy_image,
            'label_indices': dummy_label_indices,
            'label_length': dummy_label_length,
            'label_text': dummy_label_text
        }

    def get_num_classes(self):
        """Get number of classes (62 + blank token)"""
        return self.num_classes + 1

    def get_blank_index(self):
        """Get blank token index (for CTC Loss)"""
        return self.blank_index


def collate_fn(batch):
    """
    Custom collate function for variable-length sequences

    Args:
        batch: batch data

    Returns:
        images: image tensor (B, C, H, W)
        label_indices: label index tensor (B, max_length)
        label_lengths: label length tensor (B,)
        label_texts: list of label texts
    """
    try:
        # Get max label length in batch
        max_length = max([item['label_length'] for item in batch])

        # Pad labels
        padded_labels = []
        label_lengths = []
        label_texts = []

        # Get blank token index from config
        config = get_config()
        blank_index = config.get_blank_index()
        total_classes = config.get_total_classes()

        for item in batch:
            label_indices = item['label_indices']
            label_length = item['label_length']
            label_text = item['label_text']

            # Strictly validate index range
            for idx in label_indices:
                assert 0 <= idx < total_classes, f"Label index out of range: {idx}, valid range is [0, {total_classes-1}]"

            # Pad to max_length using blank token index
            padding_length = max_length - label_length
            if padding_length > 0:
                padding = torch.full((padding_length,), blank_index, dtype=torch.long)
                padded_label = torch.cat([label_indices, padding])
            else:
                padded_label = label_indices

            padded_labels.append(padded_label)
            label_lengths.append(label_length)
            label_texts.append(label_text)

        # Stack images
        images = torch.stack([item['image'] for item in batch])

        # Stack labels
        padded_labels = torch.stack(padded_labels)

        # Convert label lengths to tensor
        label_lengths = torch.tensor(label_lengths, dtype=torch.long)

        return {
            'images': images,
            'label_indices': padded_labels,
            'label_lengths': label_lengths,
            'label_texts': label_texts
        }
    except Exception as e:
        logger.error(f"Error in collate_fn: {str(e)}")
        # Return empty batch to avoid training crash
        return {
            'images': torch.empty(0),
            'label_indices': torch.empty(0),
            'label_lengths': torch.empty(0),
            'label_texts': []
        }


class StreamCaptchaDataset(Dataset):
    """
    Streaming CAPTCHA Dataset (dynamic generation)

    Core features:
    - Generates fresh samples each epoch, no offline augmentation needed
    - Integrated CaptchaGenerator for on-demand CAPTCHA synthesis
    - Quality check + auto-retry mechanism
    - Output format identical to CaptchaDataset
    - Supports infinite samples (different data each epoch)
    """

    def __init__(self,
                 transform=None,
                 max_length=None,
                 cache_size=0,
                 use_cache=False,
                 num_samples_per_epoch=100000,
                 quality_config=None,
                 generation_config=None,
                 seed=None):
        """
        Initialize streaming dataset

        Args:
            transform: data augmentation transforms
            max_length: max sequence length (default from config)
            cache_size: cache size (default 0, cache not recommended for streaming mode)
            use_cache: whether to use cache (default False)
            num_samples_per_epoch: samples per epoch (default 100000)
            quality_config: quality check config
            generation_config: CAPTCHA generation config
            seed: random seed
        """
        self.transform = transform
        self.cache_size = cache_size
        self.use_cache = use_cache
        self.num_samples_per_epoch = num_samples_per_epoch
        self.seed = seed

        # Load charset config from unified config file
        config = get_config()
        chars_config = config.get_chars_config()

        # Use max_length from config
        self.max_length = max_length if max_length is not None else chars_config['max_length']

        # Use unified CharMapper
        self.char_mapper = CharMapper.get_instance()
        self.characters = self.char_mapper.characters
        self.num_classes = self.char_mapper.num_classes
        self.blank_index = self.char_mapper.blank_index
        self.char_to_idx = self.char_mapper.char_to_idx
        self.idx_to_char = self.char_mapper.idx_to_char

        # Load quality check config
        if quality_config is None:
            quality_config = config.get('data_pipeline', {}).get('generate', {}).get('quality_check', {
                'min_laplacian_variance': 30,
                'min_contrast': 15,
                'brightness_min': 15,
                'brightness_max': 245
            })

        self.quality_config = {
            'min_laplacian_variance': quality_config.get('min_laplacian_variance', 30),
            'min_contrast': quality_config.get('min_contrast', 15),
            'brightness_min': quality_config.get('brightness_min', 15),
            'brightness_max': quality_config.get('brightness_max', 245)
        }

        # Load generation config
        if generation_config is None:
            generation_config = config.get('captcha_generation', {})

        # Initialize CAPTCHA generator (use temp dir, no disk writes)
        try:
            temp_dir = tempfile.mkdtemp(prefix='captcha_stream_')
            self.generator = CaptchaGenerator(
                output_dir=temp_dir,
                num_samples=1,
                image_width=256,
                image_height=64,
                verbose=False,
                use_multiprocessing=False
            )
            logger.info(f"StreamCaptchaDataset: CAPTCHA generator initialized successfully")
            logger.info(f"  Font pool size: {len(self.generator._font_pool) if hasattr(self.generator, '_font_pool') else 'N/A'}")
        except Exception as e:
            logger.exception(f"StreamCaptchaDataset: CAPTCHA generator initialization failed: {str(e)}")
            raise
        
        # Initialize cache (optional)
        self._cache_lock = threading.Lock() if use_cache else None
        self.image_cache = OrderedDict() if use_cache else None
        self._weak_image_refs = weakref.WeakValueDictionary() if use_cache else None

        # Sample counter (for indexing)
        self._sample_counter = 0

        logger.info(f"StreamCaptchaDataset initialized (streaming generation mode):")
        logger.info(f"  Samples per epoch: {self.num_samples_per_epoch}")
        logger.info(f"  Characters: {self.characters}")
        logger.info(f"  Num classes: {self.num_classes}")
        logger.info(f"  Blank index: {self.blank_index}")
        logger.info(f"  Max length: {self.max_length}")
        logger.info(f"  Cache: {'enabled' if use_cache else 'disabled'}")
    
    def __len__(self):
        """Return samples per epoch"""
        return self.num_samples_per_epoch

    def _check_image_quality(self, image):
        """
        Check image quality

        Args:
            image: numpy array image (H, W, C) or (H, W)

        Returns:
            bool: True if quality passes, False otherwise
        """
        try:
            # Convert to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image

            # Laplacian variance (sharpness)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            if laplacian_var < self.quality_config['min_laplacian_variance']:
                return False

            # Contrast
            contrast = gray.std()
            if contrast < self.quality_config['min_contrast']:
                return False

            # Brightness
            brightness = gray.mean()
            if brightness < self.quality_config['brightness_min'] or brightness > self.quality_config['brightness_max']:
                return False

            return True
        except Exception as e:
            logger.warning(f"Quality check failed: {str(e)}")
            return False

    def _generate_captcha_with_retry(self, max_retries=5):
        """
        Generate CAPTCHA with quality check and retry

        Args:
            max_retries: max retry attempts

        Returns:
            tuple: (image, label_text) or (None, None) on failure
        """
        for attempt in range(max_retries):
            try:
                result = self.generator.generate_single_captcha()
                if len(result) == 3:
                    image, label_text, char_positions = result
                elif len(result) == 2:
                    image, label_text = result
                else:
                    logger.warning(f"Abnormal generation return value: {len(result)} elements")
                    continue

                # Check image quality
                if self._check_image_quality(image):
                    return image, label_text
                else:
                    logger.debug(f"Quality check failed, retrying (attempt {attempt + 1}/{max_retries})")
                    continue

            except Exception as e:
                logger.warning(f"CAPTCHA generation failed (attempt {attempt + 1}/{max_retries}): {str(e)}")
                if attempt == max_retries - 1:
                    return None, None
                continue

        return None, None

    def __getitem__(self, idx):
        """
        Get single sample (dynamically generated)

        Args:
            idx: index (for tracking sample position)

        Returns:
            dict: {
                'image': torch.Tensor (3, 64, 256),
                'label_indices': torch.Tensor (length,),
                'label_length': int,
                'label_text': str
            }
        """
        try:
            # Generate CAPTCHA with quality check and retry
            image, label_text = self._generate_captcha_with_retry(max_retries=5)

            # If generation fails, return dummy sample
            if image is None or label_text is None:
                logger.warning(f"Sample {idx} generation failed, returning dummy sample")
                return self._get_dummy_sample()

            # Validate label format
            is_valid = True
            for ch in label_text:
                if ch not in self.char_to_idx:
                    is_valid = False
                    break
            if not is_valid or len(label_text) == 0 or len(label_text) > self.max_length:
                logger.warning(f"Sample {idx} has invalid label format: {label_text}")
                return self._get_dummy_sample()

            # Ensure image is 3-channel
            if len(image.shape) == 2:  # Grayscale (H, W)
                image = np.stack([image] * 3, axis=-1)  # Convert to (H, W, 3)
            elif len(image.shape) == 3 and image.shape[2] == 1:  # Single channel (H, W, 1)
                image = np.repeat(image, 3, axis=2)
            elif len(image.shape) == 3 and image.shape[2] != 3:
                if image.shape[2] > 3:
                    image = image[:, :, :3]
                else:
                    image = np.repeat(image, 3 // image.shape[2] + 1, axis=2)[:, :, :3]

            # Ensure image dimensions are correct
            if image.shape[0] != 64 or image.shape[1] != 256:
                image = cv2.resize(image, (256, 64), interpolation=cv2.INTER_LINEAR)

            # Convert to torch tensor (C, H, W)
            image = torch.from_numpy(image).permute(2, 0, 1).float().clone()

            # Force dimension validation
            assert image.shape == (3, 64, 256), f"Image dimension error: {image.shape}, expected: (3, 64, 256)"

            # Encode labels to indices
            chars = list(label_text)
            label_indices = [self.char_to_idx[char] for char in chars]

            # Validate index range
            total_classes = self.num_classes + 1
            for idx_val in label_indices:
                assert 0 <= idx_val < total_classes, f"Label index out of range: {idx_val}, valid range is [0, {total_classes-1}]"

            label_indices = torch.tensor(label_indices, dtype=torch.long).clone()
            label_length = len(label_indices)

            # Apply data augmentation
            if self.transform:
                transform_type = type(self.transform).__module__
                image_numpy = image.permute(1, 2, 0).numpy()

                if 'albumentations' in transform_type and ALBUMENTATIONS_AVAILABLE:
                    # albumentations requires uint8[0,255] input
                    image_numpy = image_numpy.astype(np.uint8)

                    augmented = self.transform(image=image_numpy)
                    image = augmented['image']

                    if isinstance(image, torch.Tensor):
                        image = image.contiguous()
                    image = image.float()
                else:
                    # torchvision.transforms
                    image_numpy = image_numpy.astype(np.uint8)
                    pil_image = Image.fromarray(image_numpy)
                    image = self.transform(pil_image)
                    if isinstance(image, torch.Tensor):
                        image = image.contiguous()
                    if isinstance(image, torch.Tensor) and image.dtype != torch.float32:
                        image = image.float()

            # Ensure tensor is contiguous
            if isinstance(image, torch.Tensor) and not image.is_contiguous():
                image = image.contiguous()

            # Re-validate dimensions after transform
            assert image.shape == (3, 64, 256), f"Image dimension error after augmentation: {image.shape}, expected: (3, 64, 256)"

            return {
                'image': image.contiguous() if isinstance(image, torch.Tensor) else image,
                'label_indices': label_indices.contiguous() if isinstance(label_indices, torch.Tensor) else label_indices,
                'label_length': label_length,
                'label_text': label_text
            }

        except Exception as e:
            logger.exception(f"Exception while getting sample {idx}: {str(e)}")
            return self._get_dummy_sample()

    def _get_dummy_sample(self):
        """
        Return dummy sample for error recovery

        Returns:
            dict: dictionary with dummy data
        """
        dummy_image = torch.zeros((3, 64, 256), dtype=torch.float32)
        dummy_label_indices = torch.zeros((6,), dtype=torch.long)
        dummy_label_length = 0
        dummy_label_text = ""
        return {
            'image': dummy_image,
            'label_indices': dummy_label_indices,
            'label_length': dummy_label_length,
            'label_text': dummy_label_text
        }

    def get_num_classes(self):
        """Get number of classes (62 + blank token)"""
        return self.num_classes + 1

    def get_blank_index(self):
        """Get blank token index (for CTC Loss)"""
        return self.blank_index

    def clear_cache(self):
        """Clear image cache and free memory"""
        if self.image_cache is not None:
            with self._cache_lock:
                self.image_cache.clear()
                if self._weak_image_refs is not None:
                    self._weak_image_refs.clear()
            gc.collect()
            logger.info(f"Image cache cleared, memory released")

    def get_cache_stats(self):
        """Get cache statistics"""
        if not self.use_cache or self.image_cache is None:
            return {
                'enabled': False,
                'size': 0,
                'max_size': 0,
                'hit_rate': 0.0
            }

        with self._cache_lock:
            return {
                'enabled': True,
                'size': len(self.image_cache),
                'max_size': self.cache_size,
                'usage_rate': len(self.image_cache) / self.cache_size
            }
