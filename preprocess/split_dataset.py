# -*- coding: utf-8 -*-
"""
Dataset Split Script
Function: Randomly split cleaned_captchas into train/val/test and copy to split_captchas/{train,val,test}.
Notes:
  - Images and their matching .txt labels are copied together (source files are not deleted)
  - Default ratios and random seed come from config.yaml (can be overridden via command-line arguments)
"""

import os
import sys
import random
import shutil
import time
from tqdm import tqdm

from utils.config_loader import get_config


class DatasetSplitter:
    """Dataset splitter"""
    def __init__(self, input_dir, output_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, random_seed=42):
        """
        Initialize dataset splitter

        Args:
            input_dir: input directory (cleaned CAPTCHAs)
            output_dir: output directory (split dataset)
            train_ratio: training set ratio (default 0.8, i.e., 80%)
            val_ratio: validation set ratio (default 0.1, i.e., 10%)
            test_ratio: test set ratio (default 0.1, i.e., 10%)
            random_seed: random seed (default 42)
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.random_seed = random_seed

        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("Sum of train, val, and test ratios must be 1.0")

        self.train_dir = os.path.join(output_dir, 'train')
        self.val_dir = os.path.join(output_dir, 'val')
        self.test_dir = os.path.join(output_dir, 'test')

        os.makedirs(self.train_dir, exist_ok=True)
        os.makedirs(self.val_dir, exist_ok=True)
        os.makedirs(self.test_dir, exist_ok=True)

        random.seed(random_seed)

    def get_all_samples(self):
        """
        Get all samples

        Returns:
            list of sample file names (image filenames)
        """
        image_files = [f for f in os.listdir(self.input_dir) if f.endswith('.png')]

        valid_samples = []
        for image_file in image_files:
            label_file = image_file.replace('.png', '.txt')
            if os.path.exists(os.path.join(self.input_dir, label_file)):
                valid_samples.append(image_file)

        return valid_samples

    def split_samples(self, samples):
        """
        Split samples

        Args:
            samples: list of samples

        Returns:
            train_samples, val_samples, test_samples
        """
        random.shuffle(samples)

        total = len(samples)
        train_end = int(total * self.train_ratio)
        val_end = train_end + int(total * self.val_ratio)

        train_samples = samples[:train_end]
        val_samples = samples[train_end:val_end]
        test_samples = samples[val_end:]

        return train_samples, val_samples, test_samples

    def copy_sample(self, sample, target_dir):
        """
        Copy a single sample (image + label), with retry mechanism

        Args:
            sample: sample file name
            target_dir: target directory

        Returns:
            whether the operation succeeded
        """
        max_retries = 3
        retry_delay = 0.1

        for attempt in range(max_retries):
            try:
                src_image = os.path.join(self.input_dir, sample)
                dst_image = os.path.join(target_dir, sample)
                shutil.copy2(src_image, dst_image)

                src_label = src_image.replace('.png', '.txt')
                dst_label = dst_image.replace('.png', '.txt')
                if os.path.exists(src_label):
                    shutil.copy2(src_label, dst_label)

                return True

            except BaseException as e:
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                logger.error(f"Error copying sample (retried {max_retries} times): {sample}, {e}")
                return False

        return False

    def copy_samples(self, samples, target_dir, desc="Copying samples"):
        """
        Batch copy samples (with rate-limiting measures)

        Args:
            samples: list of samples
            target_dir: target directory
            desc: progress bar description
        """
        for idx, sample in enumerate(tqdm(samples, desc=desc)):
            self.copy_sample(sample, target_dir)
            if (idx + 1) % 1000 == 0:
                time.sleep(0.01)

    def split_dataset(self):
        """
        Split dataset

        Returns:
            train_count, val_count, test_count
        """
        logger.info(f"Starting dataset splitting...")
        logger.info(f"Input directory: {self.input_dir}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Split ratios: train={self.train_ratio}, val={self.val_ratio}, test={self.test_ratio}")
        logger.info(f"Random seed: {self.random_seed}")

        samples = self.get_all_samples()
        logger.info(f"Total samples: {len(samples)}")

        train_samples, val_samples, test_samples = self.split_samples(samples)

        logger.info(f"Training set samples: {len(train_samples)}")
        logger.info(f"Validation set samples: {len(val_samples)}")
        logger.info(f"Test set samples: {len(test_samples)}")

        logger.info(f"Copying samples to output directory...")
        self.copy_samples(train_samples, self.train_dir, desc="Copying train set")
        self.copy_samples(val_samples, self.val_dir, desc="Copying val set")
        self.copy_samples(test_samples, self.test_dir, desc="Copying test set")

        logger.info(f"\nDataset splitting complete!")
        logger.info(f"Train set: {self.train_dir} ({len(train_samples)} samples)")
        logger.info(f"Val set: {self.val_dir} ({len(val_samples)} samples)")
        logger.info(f"Test set: {self.test_dir} ({len(test_samples)} samples)")

        return len(train_samples), len(val_samples), len(test_samples)


def main():
    """Main function"""
    config = get_config()

    """Main function"""
    import argparse
    parser.add_argument('--input_dir', type=str, default=config.get_data_dir('cleaned'), help='Input directory')
    parser.add_argument('--output_dir', type=str, default=config.get_data_dir('split'), help='Output directory')
    parser.add_argument('--train_ratio', type=float, default=config.get('data.train_ratio', 0.8), help='Training set ratio')
    parser.add_argument('--val_ratio', type=float, default=config.get('data.val_ratio', 0.1), help='Validation set ratio')
    parser.add_argument('--test_ratio', type=float, default=config.get('data.test_ratio', 0.1), help='Test set ratio')
    parser.add_argument('--random_seed', type=int, default=config.get_seed(), help='Random seed')
    args = parser.parse_args()

    splitter = DatasetSplitter(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_seed=args.random_seed
    )

    splitter.split_dataset()


if __name__ == "__main__":
    main()
