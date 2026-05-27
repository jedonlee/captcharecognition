# -*- coding: utf-8 -*-
"""
Dataset Preprocessing Script
Function: Batch preprocess split_captchas/{train,val,test} to model input size (64x256) and save to preprocessed_{train,val,test}.
Preprocessing steps:
  1. Grayscale conversion (reduce computation, preserve character info)
  2. Median filter (lightweight denoising, preserve character structure)
  3. Gaussian filter (lightweight smoothing, no character blurring)
  4. CLAHE contrast enhancement (no over-enhancement)
  5. Size normalization (64x256)
  6. Convert to BGR format (maintain 3 channels)
Note: Target size from image_config.yaml; this script is for the "preprocessing and saving" step in the training pipeline.
"""

import logging
import os
import sys
import time
import argparse
import numpy as np
from tqdm import tqdm
import cv2

logger = logging.getLogger(__name__)

from utils.config_loader import get_config
from utils.common import PathValidator


def preprocess_image_to_size(image_bgr, target_h, target_w):
    """
    Full preprocessing pipeline (optimized: preserve grayscale, remove binarization)

    Args:
        image_bgr: BGR format image
        target_h: target height
        target_w: target width

    Returns:
        Preprocessed image
    """
    if image_bgr is None or image_bgr.size == 0:
        return None

    # Step 1: Grayscale (preserve multi-level grayscale, retain anti-aliased/subpixel edge info)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # Step 2: Lightweight denoising (median filter + Gaussian filter)
    denoised = cv2.medianBlur(gray, 3)
    denoised = cv2.GaussianBlur(denoised, (3, 3), 0)

    # Step 3: Convert back to 3 channels (for model input)
    final = cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)

    # Step 4: Size normalization to 64x256
    final = cv2.resize(final, (int(target_w), int(target_h)),
                      interpolation=cv2.INTER_AREA)

    return final


def preprocess_fixed_image(image_bgr, target_h, target_w):
    """
    Fixed dataset preprocessing pipeline: Grayscale -> Gaussian blur -> Resize(64,256) -> 3-channel RGB

    Args:
        image_bgr: BGR format image
        target_h: target height
        target_w: target width

    Returns:
        Preprocessed 3-channel RGB image
    """
    if image_bgr is None or image_bgr.size == 0:
        return None

    # Step 1: Grayscale
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # Step 2: Gaussian blur
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    # Step 3: Convert back to 3 channels (for model input)
    final = cv2.cvtColor(blurred, cv2.COLOR_GRAY2BGR)

    # Step 4: Size normalization
    final = cv2.resize(final, (int(target_w), int(target_h)),
                      interpolation=cv2.INTER_AREA)

    return final


def unified_preprocess_for_deep_learning(image_bgr, target_height=64, target_width=256, apply_clahe=True):
    """
    Online preprocessing pipeline (for streaming dataset/raw data)

    Steps:
      1. BGR -> RGB
      2. RGB -> YUV, apply CLAHE on Y channel
      3. YUV -> RGB
      4. Resize to target size
      5. Normalize to [0, 1]

    Args:
        image_bgr: BGR format image (numpy array)
        target_height: target height
        target_width: target width
        apply_clahe: whether to apply CLAHE

    Returns:
        Normalized RGB image float32 [0, 1]
    """
    if image_bgr is None or image_bgr.size == 0:
        return np.zeros((target_height, target_width, 3), dtype=np.float32)

    # Step 1: BGR → RGB
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # Step 2: CLAHE on Y channel
    if apply_clahe:
        yuv = cv2.cvtColor(rgb, cv2.COLOR_RGB2YUV)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        yuv[:, :, 0] = clahe.apply(yuv[:, :, 0])
        rgb = cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)

    # Step 3: Resize
    rgb = cv2.resize(rgb, (target_width, target_height), interpolation=cv2.INTER_LINEAR)

    # Step 4: Normalize to [0, 1]
    rgb = rgb.astype(np.float32) / 255.0

    return rgb


def preprocess_split(input_dir, output_dir, target_h, target_w):
    PathValidator.validate_path(input_dir, "Input directory")
    PathValidator.validate_path(output_dir, "Output directory", must_exist=False)
    os.makedirs(output_dir, exist_ok=True)

    image_files = [f for f in os.listdir(input_dir) if f.endswith('.png')]
    max_retries = 3
    retry_delay = 0.1
    for proc_idx, image_file in enumerate(tqdm(image_files, desc=f"Preprocessing {os.path.basename(input_dir)}")):
        src_image = os.path.join(input_dir, image_file)
        src_label = src_image.replace('.png', '.txt')
        if not os.path.exists(src_label):
            continue

        preprocessed = False
        for attempt in range(max_retries):
            try:
                img = cv2.imread(src_image, cv2.IMREAD_COLOR)
                if img is None:
                    logger.warning(f"Warning: Cannot read image, skipping: {src_image}")
                    break
                out = preprocess_image_to_size(img, target_h, target_w)
                if out is None:
                    break

                dst_image = os.path.join(output_dir, image_file)
                dst_label = dst_image.replace('.png', '.txt')
                cv2.imwrite(dst_image, out)
                with open(src_label, 'r', encoding='utf-8') as f_src, open(dst_label, 'w', encoding='utf-8') as f_dst:
                    f_dst.write(f_src.read().strip())

                preprocessed = True
                break

            except BaseException as e:
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                logger.exception(f"Error preprocessing sample (retried {max_retries} times): {image_file}, {e}")

        if (proc_idx + 1) % 1000 == 0:
            time.sleep(0.01)


def preprocess_fixed_set(input_dir, output_dir, target_h, target_w, label="Fixed dataset"):
    """
    Preprocess fixed dataset: Read PNG -> Grayscale -> Gaussian blur -> Resize(64,256) -> Save as 3-channel RGB

    Args:
        input_dir: input directory (containing .png and .txt files)
        output_dir: output directory
        target_h: target height
        target_w: target width
        label: progress description label
    """
    PathValidator.validate_path(input_dir, "Input directory")
    PathValidator.validate_path(output_dir, "Output directory", must_exist=False)
    os.makedirs(output_dir, exist_ok=True)

    image_files = [f for f in os.listdir(input_dir) if f.endswith('.png')]
    max_retries = 3
    retry_delay = 0.1
    for proc_idx, image_file in enumerate(tqdm(image_files, desc=f"Preprocessing {label}")):
        src_image = os.path.join(input_dir, image_file)
        src_label = src_image.replace('.png', '.txt')
        if not os.path.exists(src_label):
            continue

        for attempt in range(max_retries):
            try:
                img = cv2.imread(src_image, cv2.IMREAD_COLOR)
                if img is None:
                    logger.warning(f"Warning: Cannot read image, skipping: {src_image}")
                    break
                out = preprocess_fixed_image(img, target_h, target_w)
                if out is None:
                    break

                dst_image = os.path.join(output_dir, image_file)
                dst_label = dst_image.replace('.png', '.txt')
                cv2.imwrite(dst_image, out)
                with open(src_label, 'r', encoding='utf-8') as f_src, open(dst_label, 'w', encoding='utf-8') as f_dst:
                    f_dst.write(f_src.read().strip())

                break

            except BaseException as e:
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                logger.exception(f"Error preprocessing sample (retried {max_retries} times): {image_file}, {e}")

        if (proc_idx + 1) % 1000 == 0:
            time.sleep(0.01)


def main():
    config = get_config()

    parser = argparse.ArgumentParser(description='Batch preprocess dataset to model input size')
    parser.add_argument('--mode', type=str, default='split', choices=['split', 'fixed'],
                        help='Preprocessing mode: split (default, process split_captchas) or fixed (process fixed dataset)')
    parser.add_argument('--split_dir', type=str, default=config.get_data_dir('split'), help='Split data directory')
    parser.add_argument('--overwrite', action='store_true', help='Whether to overwrite existing output files')
    parser.add_argument('--fixed_val_input', type=str, default=None,
                        help='Fixed validation set input directory (default: data/fixed/val)')
    parser.add_argument('--fixed_test_input', type=str, default=None,
                        help='Fixed test set input directory (default: data/fixed/test)')
    parser.add_argument('--fixed_val_output', type=str, default=None,
                        help='Fixed validation set output directory (default: data/preprocessed/val)')
    parser.add_argument('--fixed_test_output', type=str, default=None,
                        help='Fixed test set output directory (default: data/preprocessed/test)')
    args = parser.parse_args()

    target_h, target_w = config.get_preprocessed_image_size()

    if args.mode == 'fixed':
        project_root = config.get_project_root()

        if args.fixed_val_input is not None:
            val_input = args.fixed_val_input
        else:
            val_input = str(project_root / "data" / "fixed" / "val")

        if args.fixed_test_input is not None:
            test_input = args.fixed_test_input
        else:
            test_input = str(project_root / "data" / "fixed" / "test")

        if args.fixed_val_output is not None:
            val_output = args.fixed_val_output
        else:
            val_output = config.get_data_dir('val')

        if args.fixed_test_output is not None:
            test_output = args.fixed_test_output
        else:
            test_output = config.get_data_dir('test')

        if args.overwrite:
            for out_dir in [val_output, test_output]:
                if os.path.exists(out_dir):
                    for f in os.listdir(out_dir):
                        if f.endswith('.png') or f.endswith('.txt'):
                            try:
                                os.remove(os.path.join(out_dir, f))
                            except Exception:
                                pass

        preprocess_fixed_set(val_input, val_output, target_h, target_w, label="Fixed validation set")
        preprocess_fixed_set(test_input, test_output, target_h, target_w, label="Fixed test set")

        preprocessed_root = config.get_data_dir('preprocessed')
        flag_path = os.path.join(preprocessed_root, '.preprocessed_flag')
        with open(flag_path, 'w') as f:
            f.write(f"preprocessed_at={time.strftime('%Y-%m-%d_%H:%M:%S')}\n")
            f.write(f"target_size={target_h}x{target_w}\n")
            f.write(f"mode=fixed\n")
        logger.info(f"Preprocessing marker file generated: {flag_path}")
    else:
        split_dir = args.split_dir

        for subset in ['train', 'val', 'test']:
            in_dir = os.path.join(split_dir, subset)
            out_dir = config.get_data_dir(subset)
            if args.overwrite and os.path.exists(out_dir):
                for f in os.listdir(out_dir):
                    if f.endswith('.png') or f.endswith('.txt'):
                        try:
                            os.remove(os.path.join(out_dir, f))
                        except Exception:
                            pass
            preprocess_split(in_dir, out_dir, target_h, target_w)

        preprocessed_root = config.get_data_dir('preprocessed')
        flag_path = os.path.join(preprocessed_root, '.preprocessed_flag')
        with open(flag_path, 'w') as f:
            f.write(f"preprocessed_at={time.strftime('%Y-%m-%d_%H:%M:%S')}\n")
            f.write(f"target_size={target_h}x{target_w}\n")
        logger.info(f"Preprocessing marker file generated: {flag_path}")


if __name__ == "__main__":
    main()
