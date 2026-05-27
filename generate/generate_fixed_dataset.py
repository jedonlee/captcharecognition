# -*- coding: utf-8 -*-
"""
Fixed Dataset Generation Script - Generate permanent evaluation benchmarks (validation/test sets)

Generate a fixed number of CAPTCHA images for permanent model evaluation benchmarks.
- Validation set: default 4000 images, saved to data/fixed/val/
- Test set: default 4000 images, saved to data/fixed/test/

Usage:
  python generate/generate_fixed_dataset.py --val_samples 4000 --test_samples 4000
"""

import os
import random
import uuid
import argparse
import logging
from pathlib import Path

from generate.generate_dataset import CaptchaGenerator
from utils.config_loader import get_config
from utils.common import PathValidator


def setup_logging():
    """Set up logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def generate_fixed_set(generator: CaptchaGenerator, output_dir: Path, num_samples: int, logger: logging.Logger):
    """
    Generate a fixed number of captchas to the specified directory

    Args:
        generator: CaptchaGenerator instance
        output_dir: output directory
        num_samples: number of samples to generate
        logger: logger instance
    """
    PathValidator.validate_path(str(output_dir), "Output directory", must_exist=False)
    output_dir.mkdir(parents=True, exist_ok=True)

    existing_count = len(list(output_dir.glob("*.png")))
    if existing_count >= num_samples:
        logger.info(f"Directory {output_dir} already has {existing_count} images, skipping generation (target: {num_samples})")
        return 0

    remaining = num_samples - existing_count
    logger.info(f"Starting generation of {remaining} CAPTCHAs to {output_dir} (existing: {existing_count})")

    generated = 0
    for _ in range(remaining):
        try:
            image, text, char_positions = generator.generate_single_captcha()
            filename = f"{text}_{uuid.uuid4().hex[:8]}"
            image_path = output_dir / f"{filename}.png"
            label_path = output_dir / f"{filename}.txt"

            import cv2
            cv2.imwrite(str(image_path), image)
            with open(label_path, 'w', encoding='utf-8') as f:
                f.write(text)

            generated += 1
            if generated % 500 == 0:
                logger.info(f"Generated {generated}/{remaining}")
        except Exception as e:
            logger.exception(f"Error generating CAPTCHA: {e}")
            continue

    logger.info(f"Done: Generated {generated} CAPTCHAs to {output_dir}")
    return generated


def main():
    parser = argparse.ArgumentParser(description='Generate fixed evaluation benchmark dataset')
    parser.add_argument('--val_samples', type=int, default=4000, help='Number of validation set samples (default: 4000)')
    parser.add_argument('--test_samples', type=int, default=4000, help='Number of test set samples (default: 4000)')
    args = parser.parse_args()

    setup_logging()
    logger = logging.getLogger(__name__)

    config = get_config()

    random.seed(42)

    original_width, original_height = config.get_original_image_size()

    project_root = config.get_project_root()
    fixed_val_dir = project_root / "data" / "fixed" / "val"
    fixed_test_dir = project_root / "data" / "fixed" / "test"

    generator = CaptchaGenerator(
        output_dir=str(fixed_val_dir),
        num_samples=args.val_samples,
        image_width=original_width,
        image_height=original_height,
        verbose=False,
        use_multiprocessing=False,
    )

    logger.info(f"Fixed dataset generation - Validation: {args.val_samples}, Test: {args.test_samples}")
    logger.info(f"Validation set output directory: {fixed_val_dir}")
    logger.info(f"Test set output directory: {fixed_test_dir}")

    generate_fixed_set(generator, fixed_val_dir, args.val_samples, logger)
    generate_fixed_set(generator, fixed_test_dir, args.test_samples, logger)

    logger.info("Fixed dataset generation complete")


if __name__ == "__main__":
    main()
