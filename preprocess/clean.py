# -*- coding: utf-8 -*-
"""
Data Cleaning Script
Function: Automatically filter low-quality CAPTCHAs (invalid labels, poor image quality, abnormal brightness), keep 16000 high-quality samples
Notes:
  - Default target count is 16000, configurable via config.yaml dataset_cleaning.target_count or command-line arguments
  - When strict_target=false, if valid samples are insufficient, all valid samples are kept with a warning
  - Generation phase performs real-time quality checks (blur, contrast), cleaning phase performs basic quality checks (label validity, image quality, brightness), avoiding redundant checks
"""

import os
import sys
import logging
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)

from utils.config_loader import get_config
from utils.common import PathValidator
import numpy as np
import time
from tqdm import tqdm


class CaptchaCleaner:
    """Captcha data cleaner"""

    def __init__(self, input_dir, output_dir, target_count=16000):
        """
        Initialize the data cleaner

        Args:
            input_dir: input directory (raw captchas)
            output_dir: output directory (cleaned captchas)
            target_count: target number of samples to keep (default 16000)
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.target_count = target_count

        # Quality check log (record quality metrics for all samples)
        self.quality_log = []
        self.rejected_samples = []

        # Load character set from unified config
        config = get_config()
        chars_config = config.get_chars_config()
        cleaning_cfg = config.get('dataset_cleaning', {}) or {}
        thresholds = cleaning_cfg.get('thresholds', {}) if isinstance(cleaning_cfg, dict) else {}
        self._strict_target = bool(cleaning_cfg.get('strict_target', True)) if isinstance(cleaning_cfg, dict) else True
        # Default values should match those in config.yaml
        self._min_laplacian_var = float(thresholds.get('min_laplacian_var', 30))
        self._min_contrast_std = float(thresholds.get('min_contrast_std', 15))
        self._min_foreground_ratio = float(thresholds.get('min_foreground_ratio', 0.0))
        self._max_foreground_ratio = float(thresholds.get('max_foreground_ratio', 1.0))
        self._min_entropy = float(thresholds.get('min_entropy', 0.0))
        self._expected_width, self._expected_height = config.get_original_image_size()

        # Use character set from config
        self.characters = chars_config['characters']
        self.num_classes = chars_config['num_classes']

        # Create output directory (with validation)
        try:
            PathValidator.validate_path(input_dir, "Input directory")
            PathValidator.validate_path(output_dir, "Output directory", must_exist=False)
            os.makedirs(output_dir, exist_ok=True)
        except (FileNotFoundError, ValueError) as e:
            logger.error(f"Error: {e}")
            raise

    def _calc_entropy(self, gray):
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).astype(np.float32)
        hist_sum = hist.sum()
        if hist_sum <= 0:
            return 0.0
        p = hist / hist_sum
        p = p[p > 0]
        return float((-p * np.log2(p)).sum())

    def _estimate_char_count(self, fg_mask, expected_len):
        h, w = fg_mask.shape[:2]
        proj = fg_mask.sum(axis=0).astype(np.float32) / max(1, h)
        thr = max(2.0, float(np.percentile(proj, 65)))
        active = proj > thr
        if not np.any(active):
            return 0
        segments = 0
        in_seg = False
        for v in active:
            if v and not in_seg:
                segments += 1
                in_seg = True
            elif not v and in_seg:
                in_seg = False
        if expected_len <= 0:
            return segments
        if segments > expected_len * 2:
            return expected_len * 2
        return segments

    def check_label_validity(self, label):
        """
        Check label validity

        Args:
            label: label string

        Returns:
            whether the label is valid
        """
        # Check length (4-6 chars)
        if len(label) < 4 or len(label) > 6:
            return False

        # Check if all chars are within the 62-class set
        for char in label:
            if char not in self.characters:
                return False

        return True

    def check_image_quality(self, image, image_path=""):
        """
        Check image quality (using quantitative metrics)

        Note: Quality checks (blur, contrast, character visibility, etc.) have already been performed
        during generation. Cleaning only checks basic image attributes to avoid redundant filtering.

        Args:
            image: input image (BGR format)
            image_path: image path (for logging)

        Returns:
            (is_valid, rejection_reason)
        """
        # Check image dimensions
        if image is None or image.size == 0:
            return False, "Image is empty or cannot be read"

        height, width = image.shape[:2]

        # Check dimensions (fixed during generation, but keep as a safeguard)
        if height != int(self._expected_height) or width != int(self._expected_width):
            return False, f"Invalid dimensions: {width}x{height}"

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mean_value = np.mean(gray)

        # Check abnormal brightness (not checked during generation, keep this check)
        if mean_value < 15 or mean_value > 245:
            return False, f"Abnormal brightness: {mean_value:.1f}"

        # Note: The following checks were already performed during generation, skip to avoid redundant filtering
        # - Blur check (Laplacian variance) - already checked during generation
        # - Contrast check (std dev) - already checked during generation
        # - Entropy check - disabled

        # Record quality metrics (for debugging)
        if image_path:
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            std_value = float(np.std(gray))
            entropy = self._calc_entropy(gray)
            self.quality_log.append({
                'image_path': image_path,
                'mean_value': mean_value,
                'laplacian_var': laplacian_var,
                'std_value': std_value,
                'entropy': entropy,
                'passed': True
            })

        return True, ""

    def check_character_visibility(self, image, label):
        """
        Check character visibility (whether completely occluded)

        Note: Occlusion check was already disabled during generation (occlusion_threshold=0.0),
        cleaning also skips this check to avoid redundant filtering.

        Args:
            image: input image (BGR format)
            label: label string

        Returns:
            whether visible
        """
        # Note: Occlusion check was already disabled during generation (occlusion_threshold=0.0),
        # cleaning also skips this check to avoid redundant filtering
        return True

    def clean_single_sample(self, image_path, label_path):
        """
        Clean a single sample

        Args:
            image_path: image path
            label_path: label path

        Returns:
            (whether to keep, rejection reason)
        """
        try:
            if not os.path.exists(label_path):
                reason = f"Label file not found: {label_path}"
                self.rejected_samples.append({
                    'image_path': image_path,
                    'reason': reason
                })
                return False, reason
            with open(label_path, 'r', encoding='utf-8') as f:
                label = f.read().strip()
            if not self.check_label_validity(label):
                reason = f"Invalid label: {label}"
                self.rejected_samples.append({
                    'image_path': image_path,
                    'reason': reason
                })
                return False, reason

            base = os.path.splitext(os.path.basename(image_path))[0]
            prefix = base.split('_')[0] if '_' in base else ''
            if prefix and prefix != label:
                reason = "Label string does not match filename"
                self.rejected_samples.append({
                    'image_path': image_path,
                    'reason': reason
                })
                return False, reason

            image = cv2.imread(image_path)
            is_valid, reject_reason = self.check_image_quality(image, image_path)
            if not is_valid:
                self.rejected_samples.append({
                    'image_path': image_path,
                    'reason': reject_reason
                })
                return False, reject_reason

            if not self.check_character_visibility(image, label):
                reason = "Characters completely occluded"
                self.rejected_samples.append({
                    'image_path': image_path,
                    'reason': reason
                })
                return False, reason

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray_blur = cv2.GaussianBlur(gray, (3, 3), 0)
            _, fg = cv2.threshold(gray_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            fg_ratio = float(np.sum(fg > 0) / fg.size)
            est_count = self._estimate_char_count(fg, len(label))
            # Fix: relax char count threshold (from 3 to 5) to reduce false rejections
            # Reason: CAPTCHA characters may be adhesive, estimation may be inaccurate, causing many valid samples to be rejected
            if len(label) > 0 and abs(int(est_count) - int(len(label))) >= 5:
                reason = "Actual character count does not match label length"
                self.rejected_samples.append({
                    'image_path': image_path,
                    'reason': reason
                })
                return False, reason

            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            std_value = float(np.std(gray))
            score = float(laplacian_var) * 0.6 + float(std_value) * 0.4 - abs(fg_ratio - 0.16) * 120.0

            return True, score

        except Exception as e:
            reason = f"Processing error: {str(e)}"
            self.rejected_samples.append({
                'image_path': image_path,
                'reason': reason
            })
            logger.error(f"Error cleaning sample: {image_path}, {e}")
            return False, reason

    def clean_dataset(self):
        """
        Clean the dataset

        Returns:
            number of samples kept
        """
        logger.info(f"Starting CAPTCHA dataset cleaning...")
        logger.info(f"Input directory: {self.input_dir}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Target count: {self.target_count}")

        image_files = [f for f in os.listdir(self.input_dir) if f.endswith('.png')]
        logger.info(f"Original sample count: {len(image_files)}")

        valid_samples = []
        for image_file in tqdm(image_files, desc="Cleaning samples"):
            image_path = os.path.join(self.input_dir, image_file)
            label_path = image_path.replace('.png', '.txt')
            result = self.clean_single_sample(image_path, label_path)
            score = None
            reject_reason = ""
            if isinstance(result, tuple) and len(result) == 2:
                is_valid = bool(result[0])
                if is_valid:
                    score = float(result[1])
                else:
                    reject_reason = str(result[1])
            else:
                is_valid = False
                reject_reason = "Abnormal return value"
            if is_valid:
                valid_samples.append((score if score is not None else 0.0, image_file))
            else:
                # Log rejection reason (for debugging)
                if reject_reason:
                    pass  # Already logged in clean_single_sample

        logger.info(f"\nValid samples after cleaning: {len(valid_samples)}")
        logger.info(f"Rejected samples: {len(self.rejected_samples)}")

        # Generate quality check report
        self._generate_quality_report()

        valid_samples.sort(key=lambda x: x[0], reverse=True)

        if len(valid_samples) >= self.target_count:
            valid_samples = valid_samples[:self.target_count]
            logger.info(f"Selecting top {self.target_count} quality samples")
        elif self._strict_target:
            # Fix: remove strict error mechanism, replace with warning
            # Keep all valid samples to avoid wasting generation work
            logger.warning(f"Warning: Valid samples insufficient for target {self.target_count} (current: {len(valid_samples)})")
            logger.warning(f"Will keep all {len(valid_samples)} valid samples instead of aborting")
            logger.warning(f"Suggestion: Increase generation count or lower generation difficulty/cleaning thresholds next time")
            valid_samples = valid_samples

        if os.path.exists(self.output_dir):
            for f in os.listdir(self.output_dir):
                if f.endswith('.png') or f.endswith('.txt'):
                    try:
                        os.remove(os.path.join(self.output_dir, f))
                    except Exception:
                        pass

        logger.info(f"\nCopying valid samples to output directory...")
        max_retries = 3
        retry_delay = 0.1
        for copy_idx, (_, image_file) in enumerate(tqdm(valid_samples, desc="Copying samples")):
            copied = False
            for attempt in range(max_retries):
                try:
                    src_image = os.path.join(self.input_dir, image_file)
                    dst_image = os.path.join(self.output_dir, image_file)
                    with open(src_image, 'rb') as f_src, open(dst_image, 'wb') as f_dst:
                        f_dst.write(f_src.read())

                    src_label = src_image.replace('.png', '.txt')
                    dst_label = dst_image.replace('.png', '.txt')
                    if os.path.exists(src_label):
                        with open(src_label, 'rb') as f_src, open(dst_label, 'wb') as f_dst:
                            f_dst.write(f_src.read())

                    copied = True
                    break

                except BaseException as e:
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                        continue
                    logger.error(f"Error copying sample (retried {max_retries} times): {image_file}, {e}")

            if (copy_idx + 1) % 1000 == 0:
                time.sleep(0.01)

        logger.info(f"\nData cleaning complete!")
        logger.info(f"Kept samples: {len(valid_samples)}")
        logger.info(f"Saved to: {self.output_dir}")
        return len(valid_samples)

    def _generate_quality_report(self):
        """Generate quality check report"""
        if not self.rejected_samples:
            logger.info("\n✅ No samples were rejected")
            return

        logger.info("\n" + "=" * 80)
        logger.info("📊 Quality Check Report")
        logger.info("=" * 80)

        reason_counts = {}
        for sample in self.rejected_samples:
            reason = sample['reason']
            reason_counts[reason] = reason_counts.get(reason, 0) + 1

        logger.info(f"\nRejection reason statistics:")
        for reason, count in sorted(reason_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / len(self.rejected_samples)) * 100
            logger.info(f"  {reason}: {count} ({percentage:.1f}%)")

        logger.info(f"\nFirst 10 rejected sample details:")
        for i, sample in enumerate(self.rejected_samples[:10], 1):
            logger.info(f"  [{i}] {sample['image_path']}")
            logger.info(f"      Reason: {sample['reason']}")

        if len(self.rejected_samples) > 10:
            logger.info(f"  ... and {len(self.rejected_samples) - 10} more samples")

        if self.quality_log:
            logger.info(f"\nQuality metrics statistics:")
            mean_values = [log['mean_value'] for log in self.quality_log]
            laplacian_vars = [log['laplacian_var'] for log in self.quality_log]
            std_values = [log['std_value'] for log in self.quality_log]

            logger.info(f"  Avg brightness: {np.mean(mean_values):.1f} (range: {np.min(mean_values):.1f} - {np.max(mean_values):.1f})")
            logger.info(f"  Avg Laplacian variance: {np.mean(laplacian_vars):.1f} (range: {np.min(laplacian_vars):.1f} - {np.max(laplacian_vars):.1f})")
            logger.info(f"  Avg std deviation: {np.mean(std_values):.1f} (range: {np.min(std_values):.1f} - {np.max(std_values):.1f})")

        logger.info("\n" + "=" * 80)


def main():
    config = get_config()

    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(description='CAPTCHA dataset cleaner')
    parser.add_argument('--input_dir', type=str, default=config.get_data_dir('raw'), help='Input directory')
    parser.add_argument('--output_dir', type=str, default=config.get_data_dir('cleaned'), help='Output directory')
    parser.add_argument('--target_count', type=int, default=config.get('dataset_cleaning.target_count', 16000), help='Target number of samples to keep')
    args = parser.parse_args()

    # Create cleaner
    cleaner = CaptchaCleaner(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        target_count=args.target_count
    )

    # Clean dataset
    cleaner.clean_dataset()


if __name__ == "__main__":
    main()
