# -*- coding: utf-8 -*-
"""
CAPTCHA Recognition Project - Entry Point (Unified Scheduler)

One-click or step-by-step execution of: fixed dataset generation, streaming training,
evaluation, comparison, and ablation studies.

Usage:
    Method 1 - Run all 8 steps with one command:
        python main.py --mode full

    Method 2 - Run individual steps:
        python main.py --step 0          # Generate + preprocess fixed dataset
        python main.py --step 1          # Stream train core model
        python main.py --step 2          # Stream train baseline model

    Method 3 - Run sub-scripts directly:
        python -m generate.generate_fixed_dataset --val_samples 4000 --test_samples 4000
        python -m models.train --model_type captcha --streaming
        python evaluate_all_models.py

Notes:
    - main.py uses subprocess.run to call sub-scripts with real-time output
    - Shows real-time tqdm progress bars and log output
    - Automatically chains 8 steps, checks success, stops on failure

Parameters from config.yaml; CLI args can override.
"""

import os
import sys
import logging
import argparse
import subprocess
from pathlib import Path
from utils.config_loader import get_config

logger = logging.getLogger(__name__)


class ProjectRunner:
    """
    Project Runner (Unified Scheduler)

    Chains 8 sub-scripts into a complete pipeline, checks each step's
    success, and stops on failure. Supports --mode and --step invocation.
    """

    def __init__(self, project_root=None):
        if project_root is None:
            self.project_root = Path(__file__).parent
        else:
            self.project_root = Path(project_root)

        self.config = get_config()
        logger.info(f"Project root directory: {self.project_root}")

    def _check_gpu_memory(self):
        """Safety Check: GPU memory check; refuses to start training if free memory < 2GB."""
        try:
            result = subprocess.run(
                "nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits",
                shell=True, capture_output=True, text=True
            )
            if result.returncode != 0:
                logger.info("Non-GPU environment or nvidia-smi unavailable, skipping memory check")
                return
            
            free_mb = int(result.stdout.strip())
            logger.info(f"GPU free memory: {free_mb} MB")
            
            if free_mb < 2000:
                logger.error(f"\n{'='*60}")
                logger.error(f"❌ Insufficient GPU memory (<2GB)")
                logger.error(f"   Current free: {free_mb} MB")
                logger.error(f"   Minimum required: 2000 MB")
                logger.error(f"   Please clean up residual training processes first")
                logger.error(f"{'='*60}\n")
                sys.exit("❌ Insufficient GPU memory, training refused.")
            else:
                logger.info(f"✅ Sufficient GPU memory: {free_mb} MB")
        except Exception as e:
            logger.warning(f"GPU memory check failed: {e}, skipping check")

    def run_command(self, command, description):
        """
        Run a sub-command via subprocess.

        Output from the sub-script (including tqdm progress) is
        forwarded directly to the current process stdout.
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Executing: {description}")
        logger.info(f"Command: {command}")
        logger.info(f"{'='*60}")

        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=self.project_root,
                check=False,
            )

            if result.returncode != 0:
                logger.error(f"\n❌ Command failed: {description}")
                logger.error(f"Return code: {result.returncode}")
                raise subprocess.CalledProcessError(
                    result.returncode, command
                )
            else:
                logger.info(f"\n✅ Command succeeded: {description}")
                return True
        except subprocess.CalledProcessError:
            raise
        except Exception as e:
            logger.error(f"\n❌ Command exception: {description}")
            logger.error(f"Exception: {str(e)}")
            raise

    def generate_fixed_dataset(self, val_samples=None, test_samples=None):
        """Generate fixed validation/test sets."""
        if val_samples is None:
            val_samples = int(self.config.get('fixed_data.num_samples', 4000))
        if test_samples is None:
            test_samples = int(self.config.get('fixed_data.num_samples', 4000))
        command = f"python -m generate.generate_fixed_dataset --val_samples {val_samples} --test_samples {test_samples}"
        description = f"Generate fixed validation set ({val_samples} images) + test set ({test_samples} images)"
        return self.run_command(command, description)

    def preprocess_fixed_dataset(self):
        """Preprocess the fixed validation/test sets (grayscale -> Gaussian blur -> Resize -> save)."""
        command = "python -m preprocess.preprocess_dataset --mode fixed"
        description = "Preprocess fixed dataset (grayscale → Gaussian blur → resize → save)"
        return self.run_command(command, description)

    def train_core_model(self, num_epochs=None, batch_size=None):
        """Stream train the core model (ConvNeXt V2-Tiny + CBAM + BiLSTM + CTC/CE)."""
        if num_epochs is None:
            num_epochs = int(self.config.get('training.num_epochs', 80))
        if batch_size is None:
            batch_size = int(self.config.get('training.batch_size', 64))

        # Auto cleanup: kill any lingering training processes
        logger.info("Cleaning up residual training processes...")
        subprocess.run(["pkill", "-9", "-f", "models/train"], capture_output=True)
        subprocess.run(["pkill", "-9", "-f", "train_baseline_vgg.py"], capture_output=True)
        subprocess.run(["pkill", "-9", "-f", "python main.py"], capture_output=True)
        import time
        time.sleep(2)

        # GPU memory check (refuse to start if insufficient)
        self._check_gpu_memory()

        command = f"python -m models.train --num_epochs {num_epochs} --batch_size {batch_size} --streaming"
        description = f"Stream train core model ({num_epochs} epochs, batch_size={batch_size})"
        return self.run_command(command, description)

    def train_baseline_model(self, num_epochs=None, batch_size=None):
        """Stream train the baseline model (VGG + BiLSTM + CTC/CE)."""
        if num_epochs is None:
            num_epochs = int(self.config.get('training.num_epochs', 80))
        if batch_size is None:
            batch_size = int(self.config.get('training.batch_size', 256))

        # Auto cleanup: kill any lingering training processes
        logger.info("Cleaning up residual training processes...")
        subprocess.run(["pkill", "-9", "-f", "models/train"], capture_output=True)
        subprocess.run(["pkill", "-9", "-f", "train_baseline_vgg.py"], capture_output=True)
        subprocess.run(["pkill", "-9", "-f", "python main.py"], capture_output=True)
        import time
        time.sleep(2)

        # GPU memory check (refuse to start if insufficient)
        self._check_gpu_memory()

        command = f"python -m train_baseline_vgg --num_epochs {num_epochs} --batch_size {batch_size} --streaming"
        description = f"Stream train baseline model ({num_epochs} epochs, batch_size={batch_size})"
        return self.run_command(command, description)

    def evaluate_all_models(self, test_dir=None):
        """Evaluate all models (core + baseline)."""
        if test_dir is None:
            test_dir = self.config.get_data_dir('test')
        command = f"python evaluate_all_models.py --test_dir {test_dir}"
        description = "Evaluate all models (core + baseline)"
        return self.run_command(command, description)

    def run_traditional_methods(self):
        """Run traditional method comparison experiments."""
        command = "python evaluate_traditional.py"
        description = "Traditional method evaluation: OpenCV contour detection + KNN classification"
        return self.run_command(command, description)

    def generate_comparison_report(self):
        """Generate the four-model comparison report."""
        command = "python generate_comparison_report.py"
        description = "Generate four-model comparison report (bar charts + Markdown + JSON)"
        return self.run_command(command, description)

    def run_ablation_experiments(self, num_epochs=None):
        """Run ablation experiments (5 experiments)."""
        if num_epochs is None:
            num_epochs = int(self.config.get('ablation.num_epochs', 15))
        command = f"python run_ablation.py --experiments all"
        description = f"Ablation experiments (5 groups, {num_epochs} epochs/group)"
        return self.run_command(command, description)

    def mlops_fine_tune(self, hard_sample_dir=None, threshold=None, lr=None, epochs=None):
        """
        MLOps automated fine-tuning: collect hard samples, build mixed dataset,
        fine-tune, evaluate, and apply quality gating.
        """
        # GPU memory check (refuse to start if insufficient)
        self._check_gpu_memory()

        import shutil
        import cv2
        import tempfile
        import torch
        from torch.utils.data import DataLoader, Dataset
        from pathlib import Path
        from PIL import Image
        import numpy as np
        import random
        from datetime import datetime
        from collections import Counter

        # Import dependencies
        from generate.generate_dataset import CaptchaGenerator
        from models.dataset import collate_fn as dataset_collate

        if hard_sample_dir is None:
            hard_sample_dir = str(self.project_root / 'data' / 'hard_samples')
        if threshold is None:
            threshold = 500
        if lr is None:
            lr = 1e-6
        if epochs is None:
            epochs = 5

        logger.info(f"\n{'=' * 60}")
        logger.info(f"MLOps Automated Fine-tuning")
        logger.info(f"{'=' * 60}")

        # Check number of hard samples
        hard_sample_path = Path(hard_sample_dir)
        hard_sample_path.mkdir(parents=True, exist_ok=True)

        image_extensions = {'.png', '.jpg', '.jpeg', '.bmp'}
        image_files = [f for f in hard_sample_path.iterdir() if f.suffix.lower() in image_extensions]
        sample_count = len(image_files)

        logger.info(f"Hard sample directory: {hard_sample_dir}")
        logger.info(f"Current sample count: {sample_count}")
        logger.info(f"Trigger threshold: {threshold}")

        if sample_count < threshold:
            logger.warning(f"\n⚠️ Hard sample count ({sample_count}) below threshold ({threshold}), exiting fine-tuning")
            return False

        logger.info(f"\n✅ Hard sample count exceeds threshold, starting fine-tuning...")

        # Error sample analysis
        from collections import Counter

        log_dir = self.project_root / 'logs'
        log_dir.mkdir(parents=True, exist_ok=True)
        analysis_log_path = log_dir / 'mlops_analysis.log'

        hard_sample_pairs = []
        image_extensions_set = {'.png', '.jpg', '.jpeg', '.bmp'}

        for img_file in hard_sample_path.iterdir():
            if img_file.suffix.lower() not in image_extensions_set:
                continue
            txt_file = img_file.with_suffix('.txt')
            if txt_file.exists():
                with open(txt_file, 'r', encoding='utf-8') as f:
                    label_text = f.read().strip()
                pred_text = img_file.stem.split('_pred_')[-1] if '_pred_' in img_file.stem else None
                hard_sample_pairs.append({
                    'image': img_file.name,
                    'label': label_text,
                    'pred': pred_text
                })

        if hard_sample_pairs:
            all_label_chars = Counter()
            all_pred_chars = Counter()
            length_distribution = Counter()
            confusion_pairs = Counter()

            for sample in hard_sample_pairs:
                label = sample['label']
                pred = sample['pred']
                length_distribution[len(label)] += 1

                for ch in label:
                    all_label_chars[ch] += 1

                if pred and len(label) == len(pred):
                    for lc, pc in zip(label, pred):
                        if lc != pc:
                            confusion_pairs[f"{pc}→{lc}"] += 1

            top_error_chars = all_label_chars.most_common(10)
            sorted_lengths = sorted(length_distribution.items())
            top_confusion = confusion_pairs.most_common(5)

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            total = len(hard_sample_pairs)

            error_chars_str = ", ".join([f"{ch}({cnt})" for ch, cnt in top_error_chars])
            length_str = ", ".join([f"{length}-char({cnt})" for length, cnt in sorted_lengths])
            confusion_str = ", ".join([f"{pair}({cnt})" for pair, cnt in top_confusion])

            with open(analysis_log_path, 'a', encoding='utf-8') as f:
                f.write(f"\n{'=' * 60}\n")
                f.write(f"[{timestamp}] Error Sample Analysis\n")
                f.write(f"{'=' * 60}\n")
                f.write(f"- Total samples: {total}\n")
                f.write(f"- Top 10 error-prone characters: {error_chars_str}\n")
                f.write(f"- Length distribution: {length_str}\n")
                f.write(f"- Top 5 confusion pairs: {confusion_str}\n")
                f.write(f"{'=' * 60}\n\n")

            logger.info(f"\n{'=' * 60}")
            logger.info(f"Error Sample Analysis")
            logger.info(f"{'=' * 60}")
            logger.info(f"Total samples: {total}")
            logger.info(f"Top 10 error-prone characters: {error_chars_str}")
            logger.info(f"Length distribution: {length_str}")
            logger.info(f"Top 5 confusion pairs: {confusion_str}")
            logger.info(f"Analysis report written to: {analysis_log_path}")
            logger.info(f"{'=' * 60}\n")

        # Load the best model
        checkpoint_path = self.project_root / 'checkpoints' / 'best_model.pth'
        if not checkpoint_path.exists():
            logger.error(f"\n❌ Best model not found: {checkpoint_path}")
            return False

        # Build mixed dataset (80% streaming + 20% hard samples)
        class MixedDataset(Dataset):
            def __init__(self, hard_sample_dir, stream_ratio=0.8,
                        num_samples_per_epoch=10000, transform=None):
                self.hard_sample_dir = Path(hard_sample_dir)
                self.transform = transform

                self.hard_samples = []
                for img_file in self.hard_sample_dir.iterdir():
                    if img_file.suffix.lower() not in image_extensions:
                        continue
                    txt_file = img_file.with_suffix('.txt')
                    if txt_file.exists():
                        self.hard_samples.append((img_file, txt_file))

                random.shuffle(self.hard_samples)

                temp_dir = tempfile.mkdtemp()
                self.generator = CaptchaGenerator(output_dir=temp_dir, num_samples=1, verbose=False)
                self.stream_ratio = stream_ratio
                self.num_samples = num_samples_per_epoch
                self.num_hard = int(num_samples_per_epoch * (1 - stream_ratio))
                self.num_stream = num_samples_per_epoch - self.num_hard

            def __len__(self):
                return self.num_samples

            def __getitem__(self, idx):
                if idx < self.num_hard and self.hard_samples:
                    img_file, txt_file = self.hard_samples[idx % len(self.hard_samples)]
                    img = Image.open(img_file).convert('RGB')
                    with open(txt_file, 'r', encoding='utf-8') as f:
                        label_text = f.read().strip()
                else:
                    cv_img, label_text, _ = self.generator.generate_single_captcha()
                    img = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))

                if self.transform:
                    img_array = np.array(img)
                    augmented = self.transform(image=img_array)
                    img = augmented['image']

                chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'
                char_to_idx = {c: i for i, c in enumerate(chars)}
                label_indices = [char_to_idx[c] for c in label_text if c in char_to_idx]

                return {
                    'image': img,
                    'label_indices': torch.tensor(label_indices, dtype=torch.long),
                    'label_text': label_text
                }

            def collate_fn(self, batch):
                images = torch.stack([item['image'] for item in batch])
                label_indices = [item['label_indices'] for item in batch]
                label_lengths = torch.tensor([len(idx) for idx in label_indices], dtype=torch.long)
                max_len = max(len(idx) for idx in label_indices)
                padded = torch.zeros(len(batch), max_len, dtype=torch.long)
                for i, idx in enumerate(label_indices):
                    padded[i, :len(idx)] = idx
                return {
                    'images': images,
                    'label_indices': padded,
                    'label_lengths': label_lengths,
                    'label_texts': [item['label_text'] for item in batch]
                }

        # Import training modules
        from models.transforms import get_train_transform
        from models.model import CaptchaModel
        from models.hybrid_loss_fixed import HybridCTCELoss
        from utils.training_utils import set_seed
        from utils.device_manager import DeviceManager
        from torch.amp import GradScaler, autocast as amp_autocast
        import torch.optim as optim
        from tqdm import tqdm

        set_seed(42)
        device = DeviceManager().device

        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model = CaptchaModel(num_chars=62, pretrained=False).to(device)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        logger.info("✅ Best model loaded successfully")

        # Backup the current best model
        checkpoint_dir = checkpoint_path.parent
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        backup_pattern = "best_model_backup_*.pth"
        existing_backups = sorted(checkpoint_dir.glob(backup_pattern))

        if checkpoint_path.exists():
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = checkpoint_dir / f"best_model_backup_{timestamp}.pth"

            logger.info(f"\n{'=' * 60}")
            logger.info(f"Model Backup")
            logger.info(f"{'=' * 60}")
            logger.info(f"Source file: {checkpoint_path}")
            logger.info(f"Backup file: {backup_path}")

            shutil.copy2(str(checkpoint_path), str(backup_path))
            logger.info(f"✅ Backup successful: {backup_path.name}")

            # Keep only the 3 most recent backups
            all_backups = sorted(checkpoint_dir.glob(backup_pattern))
            if len(all_backups) > 3:
                backups_to_delete = all_backups[:-3]
                for old_backup in backups_to_delete:
                    old_backup.unlink()
                    logger.info(f"🗑️  Deleted old backup: {old_backup.name}")

            logger.info(f"Current backup count: {len(all_backups)}")
            logger.info(f"{'=' * 60}\n")

        train_dataset = MixedDataset(
            hard_sample_dir=hard_sample_dir,
            stream_ratio=0.8,
            num_samples_per_epoch=10000,
            transform=get_train_transform()
        )
        batch_size = int(self.config.get('training.batch_size', 64))
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=2, pin_memory=True, collate_fn=train_dataset.collate_fn
        )

        # Import dataset collate_fn
        from models.dataset import collate_fn as dataset_collate

        # Fine-tuning
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-5)
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=lr * 10, total_steps=epochs * len(train_loader),
            pct_start=0.1, anneal_strategy='cos', final_div_factor=100
        )
        ctc_weight = float(self.config.get('training.ctc_weight', 0.6))
        ce_weight = float(self.config.get('training.ce_weight', 0.4))
        label_smoothing = float(self.config.get('training.label_smoothing', 0.1))
        criterion = HybridCTCELoss(num_chars=62, ctc_weight=ctc_weight, ce_weight=ce_weight, label_smoothing=label_smoothing).to(device)
        scaler = GradScaler()

        logger.info(f"\nStarting fine-tuning: lr={lr}, epochs={epochs}")
        for epoch in range(epochs):
            model.train()
            total_loss = 0.0
            batches = 0
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
                images = batch['images'].to(device)
                label_indices = batch['label_indices'].to(device)
                label_lengths = batch['label_lengths'].to(device)
                targets = torch.cat([label_indices[i][:label_lengths[i]] for i in range(len(label_lengths))])

                with amp_autocast('cuda'):
                    encoder_out, decoder_out = model(images)
                    bsz = images.size(0)
                    input_lengths = torch.full((bsz,), encoder_out.size(0), dtype=torch.long, device=device)
                    loss, _, _ = criterion(encoder_out, decoder_out, targets, input_lengths, label_lengths)

                if torch.isnan(loss) or torch.isinf(loss):
                    continue

                optimizer.zero_grad()
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

                total_loss += loss.item()
                batches += 1

            avg_loss = total_loss / batches if batches > 0 else 0
            logger.info(f"  Epoch {epoch+1} - Avg Loss: {avg_loss:.4f}")

        # Evaluate on preprocessed test set
        logger.info("\nEvaluating new model...")
        test_dir_path = self.project_root / 'data' / 'preprocessed' / 'test'

        if not test_dir_path.exists():
            logger.error(f"❌ Test set not found: {test_dir_path}")
            return False

        from models.dataset import CaptchaDataset
        from models.transforms import get_val_transform
        from utils.decoder import beam_search_decode, postprocess_text_list
        from utils.chars import CharMapper

        test_dataset = CaptchaDataset(image_dir=str(test_dir_path), transform=get_val_transform())
        from models.dataset import collate_fn as dataset_collate
        val_batch_size = int(self.config.get('training.val_batch_size', 256))
        test_loader = DataLoader(
            test_dataset, batch_size=val_batch_size, shuffle=False,
            num_workers=2, pin_memory=True, collate_fn=dataset_collate
        )

        model.eval()
        correct = 0
        total = 0
        mapper = CharMapper.get_instance()

        with torch.no_grad():
            for batch in test_loader:
                images = batch['images'].to(device)
                label_texts = batch['label_texts']
                encoder_out = model(images)
                if isinstance(encoder_out, tuple):
                    encoder_out = encoder_out[0]
                logits = encoder_out.permute(1, 0, 2).cpu()
                preds = beam_search_decode(logits.clone(), mapper, beam_width=10)
                preds = postprocess_text_list(preds)
                for pred, target in zip(preds, label_texts):
                    total += 1
                    if pred == target:
                        correct += 1

        new_accuracy = correct / total if total > 0 else 0
        logger.info(f"New model accuracy: {new_accuracy:.4f}")

        old_accuracy = checkpoint.get('best_val_accuracy', 0)
        logger.info(f"Old model accuracy: {old_accuracy:.4f}")

        # Quality gating: only replace if new model is better
        checkpoint_dir = self.project_root / 'checkpoints'

        if new_accuracy > old_accuracy:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = checkpoint_dir / f'best_model_backup_{timestamp}.pth'
            shutil.copy(checkpoint_path, backup_path)
            logger.info(f"✅ Old model backed up to {backup_path}")

            config_snapshot = self.config.get_config().copy() if self.config.get_config() else {}
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epochs,
                'best_val_accuracy': new_accuracy,
                'config': config_snapshot
            }, checkpoint_path)
            logger.info(f"✅ New model saved as best_model.pth (accuracy: {new_accuracy:.4f} vs {old_accuracy:.4f})")
            return True
        else:
            logger.warning("⚠️ New model did not meet online standards, keeping original model")
            return False

    def run_full_pipeline(self, skip_fixed_data=False, skip_ablation=False):
        """
        Run the complete 8-step pipeline.

        Steps:
            0: Generate + preprocess fixed validation/test sets (4000 val + 4000 test)
            1: Stream train core model (ConvNeXt V2-Tiny + CBAM + BiLSTM + CTC/CE)
            2: Stream train baseline model (VGG + BiLSTM + CTC/CE)
            3: Evaluate core model (on fixed test set)
            4: Evaluate baseline model (on fixed test set)
            5: Run traditional method comparison
            6: Generate four-model comparison report
            7: Run ablation experiments (5 experiments)

        Each step checks the return value; stops subsequent steps on failure.
        """
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Running complete pipeline (8 steps)")
        logger.info(f"{'=' * 60}")

        logger.info("✅ Full mode: complete workflow (parameters from config.yaml)")

        # Step 0: Generate + preprocess fixed dataset
        if not skip_fixed_data:
            if not self.generate_fixed_dataset():
                logger.error("❌ Fixed dataset generation failed")
                return False
            if not self.preprocess_fixed_dataset():
                logger.error("❌ Fixed dataset preprocessing failed")
                return False
        else:
            logger.info("⏭️ Skipping fixed dataset generation step")

        # Step 1: Stream train core model
        if not self.train_core_model():
            logger.error("❌ Core model training failed")
            return False

        # Step 2: Stream train baseline model
        if not self.train_baseline_model():
            logger.error("❌ Baseline model training failed")
            return False

        # Step 3: Evaluate all models
        if not self.evaluate_all_models():
            logger.error("❌ Model evaluation failed")
            return False

        # Step 4: Run traditional methods
        if not self.run_traditional_methods():
            logger.error("❌ Traditional method evaluation failed")
            return False

        # Step 5: Generate comparison report
        if not self.generate_comparison_report():
            logger.error("❌ Comparison report generation failed")
            return False

        # Step 6: Run ablation experiments
        if not skip_ablation:
            if not self.run_ablation_experiments():
                logger.error("❌ Ablation experiment failed")
                return False
        else:
            logger.info("⏭️ Skipping ablation experiment step")

        logger.info(f"\n{'=' * 60}")
        logger.info(f"✅ Complete pipeline executed successfully!")
        logger.info(f"{'=' * 60}")
        return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='CAPTCHA Recognition Project - Unified Entry Point')
    parser.add_argument('--mode', type=str, default='full',
                       choices=['full', 'generate_fixed', 'preprocess_fixed',
                               'train_core', 'train_baseline', 'evaluate',
                               'traditional', 'comparison', 'ablation', 'mlops'],
                       help='Operation mode')
    parser.add_argument('--step', type=int, choices=range(0, 8),
                       help='Run the specified step (0-7)')
    parser.add_argument('--skip_fixed_data', action='store_true',
                       help='Skip fixed dataset generation')
    parser.add_argument('--skip_ablation', action='store_true',
                       help='Skip ablation experiments')
    parser.add_argument('--num_epochs', type=int, default=None,
                       help='Number of epochs (train/ablation mode)')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Batch size (train mode only)')
    parser.add_argument('--val_samples', type=int, default=None,
                       help='Validation set sample count (generate_fixed mode)')
    parser.add_argument('--test_samples', type=int, default=None,
                       help='Test set sample count (generate_fixed mode)')
    parser.add_argument('--test_dir', type=str, default=None,
                       help='Test data directory (evaluate mode)')
    parser.add_argument('--hard_sample_dir', type=str, default=None,
                       help='Hard sample directory (mlops mode, reads from config by default)')
    parser.add_argument('--threshold', type=int, default=500,
                       help='Hard sample threshold to trigger fine-tuning (mlops mode, default: 500)')
    parser.add_argument('--lr', type=float, default=1e-6,
                       help='Fine-tuning learning rate (mlops mode, default: 1e-6)')
    parser.add_argument('--epochs', type=int, default=5,
                       help='Fine-tuning epochs (mlops mode, default: 5)')

    args = parser.parse_args()

    # Create project runner
    runner = ProjectRunner()

    # Run by step number
    if args.step is not None:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Running step {args.step}")
        logger.info(f"{'=' * 60}")

        try:
            if args.step == 0:
                runner.generate_fixed_dataset(val_samples=args.val_samples, test_samples=args.test_samples)
                runner.preprocess_fixed_dataset()
            elif args.step == 1:
                runner.train_core_model(num_epochs=args.num_epochs, batch_size=args.batch_size)
            elif args.step == 2:
                runner.train_baseline_model(num_epochs=args.num_epochs, batch_size=args.batch_size)
            elif args.step == 3:
                runner.evaluate_all_models(test_dir=args.test_dir)
            elif args.step == 4:
                runner.run_traditional_methods()
            elif args.step == 5:
                runner.generate_comparison_report()
            elif args.step == 6:
                runner.run_ablation_experiments(num_epochs=args.num_epochs)
            elif args.step == 7:
                runner.run_full_pipeline(skip_fixed_data=args.skip_fixed_data, skip_ablation=args.skip_ablation)
        except subprocess.CalledProcessError as e:
            logger.error(f"\n❌ Step {args.step} failed")
            sys.exit(1)

        logger.info(f"\n{'=' * 60}")
        logger.info(f"Step {args.step} completed!")
        logger.info(f"{'=' * 60}")
        return

    # Run by mode
    if args.mode == 'full':
        runner.run_full_pipeline(skip_fixed_data=args.skip_fixed_data, skip_ablation=args.skip_ablation)
    elif args.mode == 'generate_fixed':
        runner.generate_fixed_dataset(val_samples=args.val_samples, test_samples=args.test_samples)
    elif args.mode == 'preprocess_fixed':
        runner.preprocess_fixed_dataset()
    elif args.mode == 'train_core':
        runner.train_core_model(num_epochs=args.num_epochs, batch_size=args.batch_size)
    elif args.mode == 'train_baseline':
        runner.train_baseline_model(num_epochs=args.num_epochs, batch_size=args.batch_size)
    elif args.mode == 'evaluate':
        runner.evaluate_all_models(test_dir=args.test_dir)
    elif args.mode == 'traditional':
        runner.run_traditional_methods()
    elif args.mode == 'comparison':
        runner.generate_comparison_report()
    elif args.mode == 'ablation':
        runner.run_ablation_experiments(num_epochs=args.num_epochs)
    elif args.mode == 'mlops':
        runner.mlops_fine_tune(
            hard_sample_dir=args.hard_sample_dir,
            threshold=args.threshold,
            lr=args.lr,
            epochs=args.epochs
        )


if __name__ == '__main__':
    main()
