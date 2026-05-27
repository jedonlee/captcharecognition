# -*- coding: utf-8 -*-
"""
Unified Model Evaluation Script - Evaluate all trained models on the fixed test set

Evaluates:
  1. Core model (ConvNeXt V2-Tiny + BiLSTM) - checkpoints/best_model.pth
  2. Baseline model (VGG CNN + BiLSTM)        - checkpoints/baseline_vgg_best.pth

Outputs:
  - Image accuracy, character accuracy, inference speed, throughput
  - JSON results saved to results/ with timestamp

Usage:
  python evaluate_all_models.py [--batch_size BATCH_SIZE] [--gpu]
"""

import os
import sys
import json
import time
import logging
import argparse
from pathlib import Path
from datetime import datetime
import shutil

import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.model import CaptchaModel
from models.baseline_vgg_cnn_lstm import BaselineVGGCNNBiLSTM
from models.dataset import CaptchaDataset, collate_fn
from models.transforms import get_val_transform
from utils.config_loader import get_config
from utils.device_manager import DeviceManager
from utils.chars import CharMapper
from utils.decoder import (
    greedy_decode,
    beam_search_decode,
    calculate_accuracy,
    postprocess_captcha,
    postprocess_text_list,
)
from utils.metrics import print_metrics_report

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def setup_logging(log_level=logging.INFO):
    logger = logging.getLogger("evaluate_all_models")
    if logger.handlers:
        return logger
    logger.setLevel(log_level)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    logging.getLogger("utils.metrics").handlers.clear()
    logging.getLogger("utils.metrics").addHandler(console_handler)
    return logger

logger = setup_logging()

# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------

MODEL_REGISTRY = {
    "core": {
        "name": "Core (ConvNeXt V2-Tiny + BiLSTM)",
        "checkpoint_key": "checkpoint.checkpoint_dir",
        "checkpoint_filename": "best_model.pth",
        "model_class": CaptchaModel,
        "use_ctc": True,
    },
    "baseline": {
        "name": "Baseline (VGG CNN + BiLSTM)",
        "checkpoint_key": "model.baseline",
        "checkpoint_filename": "baseline_vgg_best.pth",
        "model_class": BaselineVGGCNNBiLSTM,
        "use_ctc": True,
    },
}

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def resolve_checkpoint(model_config, config_loader):
    """Resolve full checkpoint path from model config entry."""
    checkpoint_dir = config_loader.get_checkpoint_dir()
    filename = Path(checkpoint_dir) / model_config["checkpoint_filename"]
    return str(filename)

# ---------------------------------------------------------------------------
# Single-model evaluation
# ---------------------------------------------------------------------------

def evaluate_single_model(
    model,
    dataloader,
    device,
    model_name,
    use_ctc=True,
    warmup_batches=2,
    collect_hard_samples=False,
    hard_samples_dir=None,
    dataset=None,
):
    """
    Run inference on one model and return metrics.

    Returns:
        dict with keys:
          image_accuracy_greedy, char_accuracy_greedy,
          image_accuracy_beam,    char_accuracy_beam,
          image_accuracy_beam_pp, char_accuracy_beam_pp,
          avg_time_ms, std_time_ms, throughput,
          best_name, best_image_accuracy
    """

    if hard_samples_dir is None:
        config = get_config()
        hard_samples_dir = str(config.get_project_root() / 'data' / 'hard_samples')
    model.eval()

    all_logits = []
    all_targets = []
    all_times = []
    total_samples = 0

    if collect_hard_samples:
        hard_samples_path = Path(hard_samples_dir)
        hard_samples_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Hard samples will be saved to: {hard_samples_path}")

    with torch.no_grad():
        # Warmup
        warmup_count = 0
        for batch in dataloader:
            if warmup_count >= warmup_batches:
                break
            images = batch["images"].to(device)
            _ = model(images)
            warmup_count += 1

        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"  Evaluating {model_name}", leave=False)):
            images = batch["images"].to(device)
            label_indices = batch["label_indices"].to(device)
            label_texts = batch["label_texts"]

            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()

            outputs = model(images)
            if isinstance(outputs, tuple):
                encoder_out = outputs[0]
            else:
                encoder_out = outputs

            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.time()

            elapsed_ms = (t1 - t0) * 1000.0
            all_times.append(elapsed_ms)
            total_samples += images.size(0)

            # Store (batch, seq_len, classes) for decoding
            all_logits.append(encoder_out.permute(1, 0, 2).cpu())
            all_targets.extend(label_texts)

            # Collect hard samples
            if collect_hard_samples and dataset is not None:
                # Decode to check for misclassifications
                logits_batch_current = encoder_out.permute(1, 0, 2).cpu()
                char_mapper = CharMapper.get_instance()
                preds = beam_search_decode(logits_batch_current.clone(), char_mapper, beam_width=10)
                preds = postprocess_text_list(preds)

                for i, (pred, target) in enumerate(zip(preds, label_texts)):
                    if pred != target:
                        # Get image index in dataset
                        img_idx = batch_idx * dataloader.batch_size + i
                        if img_idx < len(dataset):
                            # Get sample info from dataset
                            sample_info = dataset.valid_samples[img_idx]
                            img_path = Path(dataset.image_dir) / sample_info['image_file']
                            label = sample_info['label_text']
                            src_path = Path(img_path)
                            if src_path.exists():
                                # Create unique filename
                                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                dst_name = f"{timestamp}_{src_path.stem}_pred_{pred}.png"
                                dst_path = hard_samples_path / dst_name
                                
                                # Copy image
                                shutil.copy(str(src_path), str(dst_path))
                                
                                # Save label
                                label_path = hard_samples_path / f"{dst_path.stem}.txt"
                                with open(str(label_path), 'w', encoding='utf-8') as f:
                                    f.write(target)

    # Concatenate logits
    logits_batch = torch.cat(all_logits, dim=0)  # (N, seq_len, classes)
    # Transpose to (seq_len, N, classes) for decoders
    logits_seq = logits_batch.permute(1, 0, 2)

    char_mapper = CharMapper.get_instance()

    # Greedy decode
    pred_greedy = greedy_decode(logits_batch.clone(), char_mapper, use_ctc=use_ctc)
    img_acc_greedy, char_acc_greedy = calculate_accuracy(pred_greedy, all_targets)

    # Beam Search decode (beam_width=10)
    pred_beam = beam_search_decode(logits_batch.clone(), char_mapper, beam_width=10)
    img_acc_beam, char_acc_beam = calculate_accuracy(pred_beam, all_targets)

    # Beam Search + post-processing
    pred_beam_pp = postprocess_text_list(pred_beam)
    img_acc_beam_pp, char_acc_beam_pp = calculate_accuracy(pred_beam_pp, all_targets)

    # Timing stats
    avg_time = float(np.mean(all_times))
    std_time = float(np.std(all_times))
    batch_size = dataloader.batch_size
    throughput = batch_size / (avg_time / 1000.0)

    # Determine best decoding strategy
    candidates = {
        "Greedy": img_acc_greedy,
        "Beam3": img_acc_beam,
        "Beam3+PostProcess": img_acc_beam_pp,
    }
    best_name = max(candidates, key=candidates.get)
    best_img_acc = candidates[best_name]

    logger.info(f"  Greedy               -> img_acc: {img_acc_greedy*100:.2f}%, char_acc: {char_acc_greedy*100:.2f}%")
    logger.info(f"  Beam3                -> img_acc: {img_acc_beam*100:.2f}%, char_acc: {char_acc_beam*100:.2f}%")
    logger.info(f"  Beam3 + PostProcess  -> img_acc: {img_acc_beam_pp*100:.2f}%, char_acc: {char_acc_beam_pp*100:.2f}%")
    logger.info(f"  Best strategy: {best_name} ({best_img_acc*100:.2f}%)")
    logger.info(f"  Avg inference: {avg_time:.2f} ms, Throughput: {throughput:.1f} samples/s")

    print_metrics_report(pred_beam_pp, all_targets, chars=char_mapper.characters)

    return {
        "image_accuracy_greedy": round(float(img_acc_greedy), 6),
        "char_accuracy_greedy": round(float(char_acc_greedy), 6),
        "image_accuracy_beam": round(float(img_acc_beam), 6),
        "char_accuracy_beam": round(float(char_acc_beam), 6),
        "image_accuracy_beam_postprocess": round(float(img_acc_beam_pp), 6),
        "char_accuracy_beam_postprocess": round(float(char_acc_beam_pp), 6),
        "avg_time_ms": round(avg_time, 4),
        "std_time_ms": round(std_time, 4),
        "throughput": round(throughput, 2),
        "best_strategy": best_name,
        "best_image_accuracy": round(float(best_img_acc), 6),
        "total_samples": total_samples,
    }

# ---------------------------------------------------------------------------
# Checkpoint loader
# ---------------------------------------------------------------------------

def load_model(model_entry, checkpoint_path, device):
    """
    Create model instance and load weights.

    Returns:
        (model, model_info_dict)
    """
    model_class = model_entry["model_class"]
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        return None, None

    try:
        checkpoint = torch.load(str(checkpoint_path), map_location=device, weights_only=True)
    except Exception as exc:
        logger.error(f"  Failed to load checkpoint {checkpoint_path}: {exc}")
        return None, None

    state_dict = checkpoint.get("model_state_dict", checkpoint)
    use_ctc = model_entry.get("use_ctc", True)

    if model_class is BaselineVGGCNNBiLSTM:
        num_classes = checkpoint.get("num_classes", 63)
        model = BaselineVGGCNNBiLSTM(num_classes=num_classes).to(device)
        model_type = "baseline"
    else:
        fc_weight = state_dict.get("fc.weight", state_dict.get("decoder.fc.weight"))
        if fc_weight is not None:
            num_classes_from_ckpt = fc_weight.shape[0]
            num_chars = num_classes_from_ckpt - 1
        else:
            config = get_config()
            num_chars = config.get_total_classes() - 1
        model = CaptchaModel(num_chars=num_chars, pretrained=False).to(device)
        model_type = "core"

    try:
        model.load_state_dict(state_dict)
        logger.info(f"  Loaded weights from {checkpoint_path}")
    except Exception as exc:
        logger.error(f"  State dict mismatch: {exc}")
        return None, None

    info = {
        "model_type": model_type,
        "num_params": sum(p.numel() for p in model.parameters()),
        "checkpoint_path": str(checkpoint_path),
    }
    return model, info

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate all trained captcha recognition models")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size for evaluation")
    parser.add_argument("--gpu", action="store_true", help="Force GPU usage")
    parser.add_argument("--test_dir", type=str, default=None, help="Path to test image directory")
    parser.add_argument("--models", type=str, nargs="+", default=None,
                        help="Model keys to evaluate (e.g. core baseline). Default: all")
    parser.add_argument("--collect_hard_samples", action="store_true",
                        help="Collect misclassified images as hard samples for MLOps fine-tuning")
    parser.add_argument("--hard_samples_dir", type=str, default=None,
                        help="Directory to save hard samples (default from config)")
    args = parser.parse_args()

    config = get_config()

    # Device
    device_manager = DeviceManager(prefer_gpu=args.gpu, verbose=True)
    device = device_manager.get_device()

    # Test data directory
    if args.test_dir is not None:
        test_dir = args.test_dir
    else:
        test_dir = config.get_data_dir("test")

    if not os.path.isdir(test_dir):
        logger.error(f"Test directory does not exist: {test_dir}")
        sys.exit(1)

    # Batch size
    batch_size = args.batch_size or config.get("evaluation.batch_size", 256)

    logger.info("=" * 60)
    logger.info("Unified Model Evaluation")
    logger.info("=" * 60)
    logger.info(f"Device:         {device}")
    logger.info(f"Test directory: {test_dir}")
    logger.info(f"Batch size:     {batch_size}")

    # Dataset
    try:
        test_dataset = CaptchaDataset(
            image_dir=test_dir,
            transform=get_val_transform(),
        )
    except Exception as exc:
        logger.exception(f"Failed to create test dataset: {exc}")
        sys.exit(1)

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        drop_last=False,
        num_workers=0 if os.name == "nt" else 2,
        pin_memory=device_manager.is_cuda(),
    )
    logger.info(f"Test samples: {len(test_dataset)}")

    # Determine which models to evaluate
    if args.models is not None:
        model_keys = [k for k in args.models if k in MODEL_REGISTRY]
    else:
        model_keys = list(MODEL_REGISTRY.keys())

    if not model_keys:
        logger.error("No valid model keys specified")
        sys.exit(1)

    results = {}
    all_results = {
        "timestamp": datetime.now().isoformat(),
        "device": str(device),
        "test_dir": test_dir,
        "test_samples": len(test_dataset),
        "batch_size": batch_size,
        "models": {},
    }

    for key in model_keys:
        model_entry = MODEL_REGISTRY[key]
        logger.info(f"\n{'='*60}")
        logger.info(f"Model: {model_entry['name']} ({key})")
        logger.info(f"{'='*60}")

        checkpoint_path = resolve_checkpoint(model_entry, config)
        logger.info(f"Checkpoint: {checkpoint_path}")

        model, model_info = load_model(model_entry, checkpoint_path, device)
        if model is None:
            logger.warning(f"  Skipping {key} - checkpoint not loaded")
            all_results["models"][key] = {
                "status": "error",
                "checkpoint": checkpoint_path,
                "message": "Checkpoint not found or failed to load",
            }
            continue

        metrics = evaluate_single_model(
            model=model,
            dataloader=test_loader,
            device=device,
            model_name=model_entry["name"],
            use_ctc=model_entry.get("use_ctc", True),
            warmup_batches=2,
            collect_hard_samples=args.collect_hard_samples,
            hard_samples_dir=args.hard_samples_dir,
            dataset=test_dataset,
        )

        result_entry = {
            "status": "ok",
            "name": model_entry["name"],
            "model_type": model_info["model_type"],
            "num_params": model_info["num_params"],
            "checkpoint": checkpoint_path,
            "metrics": metrics,
        }
        all_results["models"][key] = result_entry

        # Clean up
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("Evaluation Summary")
    logger.info(f"{'='*60}")

    for key, entry in all_results["models"].items():
        if entry.get("status") != "ok":
            logger.info(f"  {key}: FAILED - {entry.get('message', 'unknown')}")
            continue
        m = entry["metrics"]
        logger.info(f"  {key} ({entry['name']})")
        logger.info(f"    Best: {m['best_strategy']} -> {m['best_image_accuracy']*100:.2f}% image acc")
        logger.info(f"    Avg time: {m['avg_time_ms']:.2f} ms, Throughput: {m['throughput']:.1f} samples/s")

    # Save JSON
    results_dir = config.get_results_dir()
    os.makedirs(results_dir, exist_ok=True)
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(results_dir, f"evaluate_all_{timestamp_str}.json")

    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        logger.info(f"\nResults saved to: {output_path}")
    except Exception as exc:
        logger.exception(f"Failed to save results: {exc}")

    logger.info("Done.")


if __name__ == "__main__":
    main()
