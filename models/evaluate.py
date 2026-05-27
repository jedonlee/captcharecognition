# -*- coding: utf-8 -*-
"""
Model Evaluation Script - Evaluate model performance on test set, calculate accuracy and latency

Evaluation Flow:
+------------------------------------------------------------------+
| 1. Initialization                                                 |
|    - Load config (config.yaml)                                    |
|    - Validate checkpoint and data paths                           |
|    - Load model weights                                           |
|    - Create dataset and data loader                               |
|    - Initialize device (GPU/CPU)                                  |
+------------------------------------------------------------------+
                            |
+------------------------------------------------------------------+
| 2. Evaluation Phase                                               |
|    - Set model to eval mode                                       |
|    - Iterate over test data loader                                |
|    - Forward pass: compute model output                           |
|    - Decode predictions (Greedy/Beam Search)                      |
|    - Calculate evaluation metrics (accuracy, latency, etc.)       |
|    - Record predictions and labels                                |
+------------------------------------------------------------------+
                            |
+------------------------------------------------------------------+
| 3. Result Output Phase                                            |
|    - Print evaluation results (image accuracy, char accuracy, inference time, etc.) |
|    - Save evaluation results to JSON file (optional)              |
|    - Generate confusion matrix (optional)                         |
|    - Generate error analysis report (optional)                    |
+------------------------------------------------------------------+

Evaluation Metrics:
1. Image Accuracy:
   - Exact match count / total captcha count
   - Reflects overall recognition ability
   - Strict metric, requires all characters correct

2. Character Accuracy:
   - Correctly recognized characters / total characters
   - Reflects character recognition ability
   - Relaxed metric, tolerates partial errors

3. Inference Time:
   - Average inference time per captcha
   - Reflects inference efficiency
   - Includes forward pass and decoding time

Decoding Strategies:
1. Greedy Decoding (default):
   - Pick the most probable character at each time step
   - Simple and fast, may not be optimal
   - Suitable for real-time scenarios

2. Beam Search Decoding (optional):
   - Keep top-k candidate sequences
   - More accurate, but higher computation
   - Suitable for accuracy-critical scenarios

Error Handling:
1. Checkpoint file not found: detailed error info and suggestions
2. Dataset directory not found: detailed error info and suggestions
3. Dataset directory empty: detailed error info and suggestions
4. Model loading failed: log error, skip model
5. Inference failed: log error, return empty result
"""

import os
import sys
import logging
import argparse
import time
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import json

from models.model import CaptchaModel
from models.baseline_vgg_cnn_lstm import BaselineVGGCNNBiLSTM
from models.dataset import CaptchaDataset, collate_fn
from models.transforms import get_val_transform
from utils.config_loader import get_config
from utils.device_manager import DeviceManager
from utils.directory_manager import DirectoryManager
from utils.chars import CharMapper
from utils.decoder import greedy_decode as _greedy_decode, beam_search_decode as _beam_search_decode, calculate_accuracy as _calculate_accuracy

logger = logging.getLogger(__name__)


def validate_paths(checkpoint_path, data_path):
    """
    Validate that paths exist, provide detailed error messages and suggestions

    Raises:
        FileNotFoundError: when a path does not exist
    """
    # Validate checkpoint path
    if not os.path.exists(checkpoint_path):
        error_msg = f"\n{'='*60}\n"
        error_msg += f"Error: Model checkpoint file not found\n"
        error_msg += f"{'='*60}\n"
        error_msg += f"Checkpoint path: {checkpoint_path}\n\n"
        error_msg += f"Possible causes:\n"
        error_msg += f"1. Model training has not been completed\n"
        error_msg += f"2. Checkpoint file path is misconfigured\n"
        error_msg += f"3. Checkpoint file has been moved or deleted\n\n"
        error_msg += f"Suggestions:\n"
        error_msg += f"1. Run the training script: python models/train.py\n"
        error_msg += f"2. Use --checkpoint to specify the correct checkpoint path\n"
        error_msg += f"3. Check the checkpoints directory for available model files\n"
        error_msg += f"4. Use view_checkpoint.py to view available checkpoints\n"
        error_msg += f"{'='*60}\n"
        raise FileNotFoundError(error_msg)

    # Validate dataset path
    if not os.path.exists(data_path):
        error_msg = f"\n{'='*60}\n"
        error_msg += f"Error: Dataset directory not found\n"
        error_msg += f"{'='*60}\n"
        error_msg += f"Dataset path: {data_path}\n\n"
        error_msg += f"Possible causes:\n"
        error_msg += f"1. Dataset has not been generated\n"
        error_msg += f"2. Dataset path is misconfigured\n"
        error_msg += f"3. Dataset directory has been moved or deleted\n\n"
        error_msg += f"Suggestions:\n"
        error_msg += f"1. Run the dataset generation script: python generate/generate_dataset.py\n"
        error_msg += f"2. Use --data to specify the correct dataset path\n"
        error_msg += f"3. Check the data directory for available datasets\n"
        error_msg += f"{'='*60}\n"
        raise FileNotFoundError(error_msg)

    if not os.listdir(data_path):
        error_msg = f"\n{'='*60}\n"
        error_msg += f"Error: Dataset directory is empty\n"
        error_msg += f"{'='*60}\n"
        error_msg += f"Dataset path: {data_path}\n\n"
        error_msg += f"Suggestions:\n"
        error_msg += f"1. Run the dataset generation script: python generate/generate_dataset.py\n"
        error_msg += f"2. Use --data to specify the correct dataset path\n"
        error_msg += f"3. Check the data directory for available datasets\n"
        error_msg += f"{'='*60}\n"
        raise FileNotFoundError(error_msg)

    logger.info(f"Path validation passed")
    logger.info(f"  Model checkpoint: {checkpoint_path}")
    logger.info(f"  Dataset directory: {data_path}")


def decode_predictions(predictions, chars, use_ctc=True, use_beam_search=False, beam_width=3):
    """Unified decoding: delegates to utils.decoder.greedy_decode / beam_search_decode"""
    if use_beam_search and use_ctc:
        char_mapper = CharMapper.get_instance()
        return _beam_search_decode(predictions, char_mapper, beam_width)
    char_mapper = CharMapper.get_instance()
    return _greedy_decode(predictions, char_mapper, use_ctc=use_ctc)


def beam_search_decode(predictions, chars, beam_width=3):
    """Unified Beam Search: delegates to utils.decoder.beam_search_decode"""
    char_mapper = CharMapper.get_instance()
    return _beam_search_decode(predictions, char_mapper, beam_width)


def calculate_accuracy(predictions, targets, chars, use_ctc=True, use_beam_search=False, beam_width=3):
    """Unified accuracy calculation: delegates to utils.decoder.calculate_accuracy"""
    pred_texts = decode_predictions(predictions, chars, use_ctc, use_beam_search, beam_width)
    target_texts = []
    for target_indices in targets:
        valid = [i.item() if torch.is_tensor(i) else i for i in target_indices
                 if 0 <= (i.item() if torch.is_tensor(i) else i) < len(chars)]
        target_texts.append(''.join(chars[i] for i in valid))
    return _calculate_accuracy(pred_texts, target_texts)


def evaluate(model, dataloader, device, chars, warmup_batches=2, use_ctc=True):
    """
    Evaluate model performance on test set (uses unified CharMapper)

    Args:
        model: model to evaluate
        dataloader: test data loader
        device: compute device (cuda/cpu)
        chars: character set string
        warmup_batches: number of warmup batches (default 2)
        use_ctc: whether to use CTC decoding (default True)

    Returns:
        dict: dictionary containing evaluation metrics
    """
    model.eval()

    all_predictions = []
    all_targets = []
    all_times = []
    failure_cases = []

    with torch.inference_mode():
        logger.info(f"Starting warmup（{warmup_batches}batches）...")
        warmup_count = 0
        for batch_idx, batch in enumerate(dataloader):
            if warmup_count >= warmup_batches:
                break
            images = batch['images'].to(device)
            outputs = model(images)
            if isinstance(outputs, tuple):
                _ = outputs[1]
            warmup_count += 1

        logger.info(f"Warmup complete, starting evaluation...")

        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
            images = batch['images'].to(device)
            label_indices = batch['label_indices'].to(device)

            if device.type == 'cuda':
                torch.cuda.synchronize()

            start_time = time.time()

            # ⚠️ Fix: Compatible with different model outputs (CaptchaModel returns tuple, Baseline returns tensor)
            # 🔧 2026-04-12 Fix: Use encoder_out (consistent with train.py), not decoder_out
            outputs = model(images)
            if isinstance(outputs, tuple):
                encoder_out = outputs[0]
            else:
                encoder_out = outputs

            if device.type == 'cuda':
                torch.cuda.synchronize()

            end_time = time.time()
            inference_time = (end_time - start_time) * 1000
            all_times.append(inference_time)

            # encoder_out shape: (32, batch, 63)
            # Convert to (batch, 32, 63) before saving for easier concatenation
            all_predictions.append(encoder_out.permute(1, 0, 2).cpu())
            all_targets.append(label_indices.cpu())

    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    # Convert back to (32, total_batch, 63) format for decoding (consistent with train.py)
    all_predictions = all_predictions.permute(1, 0, 2)

    # 🔧 2026-04-06 Fix: Pass chars parameter instead of label_encoder
    image_accuracy, char_accuracy = calculate_accuracy(all_predictions, all_targets, chars, use_ctc=use_ctc)

    # 🔧 Compute target and predicted texts first (shared by greedy and beam)
    # 🔧 2026-04-12 Fix: Move earlier to avoid pred_texts undefined error
    target_texts = []
    for target_indices in all_targets:
        target_indices_filtered = [i for i in target_indices if 0 <= i < len(chars)]
        target_text = ''.join(chars[idx] for idx in target_indices_filtered)
        target_texts.append(target_text)
    
    # 🔧 2026-04-06 Fix: Use chars parameter for decoding
    pred_texts = decode_predictions(all_predictions, chars)

    # 🔧 2026-04-06 Refine-7: Beam Search accuracy (core model uses beam=3)
    char_mapper = CharMapper.get_instance()
    beam_texts = _beam_search_decode(all_predictions, char_mapper, beam_width=10)
    beam_img_acc, beam_char_acc = _calculate_accuracy(beam_texts, target_texts)
    
    # 🔧 New: Apply post-processing, calculate accuracy
    from utils.decoder import postprocess_text_list
    beam3_post_texts = postprocess_text_list(beam_texts)
    beam3_post_img_acc, beam3_post_char_acc = _calculate_accuracy(beam3_post_texts, target_texts)

    logger.info(f"\n  Beam Search decoding results (beam_width=10):")
    logger.info(f"     Beam Search(3)        -> Image Accuracy: {beam_img_acc*100:.2f}%, Char Accuracy: {beam_char_acc*100:.2f}%")
    logger.info(f"     Beam Search(3)+Post  -> Image Accuracy: {beam3_post_img_acc*100:.2f}%, Char Accuracy: {beam3_post_char_acc*100:.2f}%")
    
    best_acc = beam_img_acc
    best_name = "Beam3"
    if beam3_post_img_acc > best_acc:
        best_acc = beam3_post_img_acc
        best_name = "Beam3+Post"
    
    logger.info(f"\n  Best combination: {best_name} -> Image Accuracy: {best_acc*100:.2f}%")

    avg_time = np.mean(all_times)
    std_time = np.std(all_times)

    # Throughput = batch_size / average inference time (seconds)
    batch_size = dataloader.batch_size
    throughput = batch_size / (avg_time / 1000)

    # Collect and analyze failure cases
    for idx, (pred_text, target_text) in enumerate(zip(pred_texts, target_texts)):
        if pred_text != target_text:
            error_type = analyze_error_type(pred_text, target_text)

            failure_cases.append({
                'index': idx,
                'prediction': pred_text,
                'target': target_text,
                'error_type': error_type,
                'length_diff': len(pred_text) - len(target_text)
            })

    return {
        'image_accuracy': image_accuracy,
        'char_accuracy': char_accuracy,
        'avg_time_ms': avg_time,
        'std_time_ms': std_time,
        'throughput': throughput,
        'predictions': pred_texts,
        'targets': target_texts,
        'failure_cases': failure_cases
    }


def analyze_error_type(pred_text, target_text):
    """
    Analyze prediction error type to help understand model failure reasons

    Returns:
        str: error type string, possible values:
            - 'length_mismatch': predicted length differs from target
            - 'confusion_X_Y': confusion pair error (X misidentified as Y)
            - 'single_char_error': single character error
            - 'double_char_error': two character errors
            - 'multiple_char_errors': multiple character errors
    """
    # Length mismatch
    if len(pred_text) != len(target_text):
        return 'length_mismatch'

    # Confusable character pairs
    confusion_pairs = [
        ('o', '0'), ('0', 'o'),
        ('l', '1'), ('1', 'l'),
        ('I', '1'), ('1', 'I'),
        ('I', 'l'), ('l', 'I'),
        ('O', '0'), ('0', 'O'),
        ('S', '5'), ('5', 'S'),
        ('Z', '2'), ('2', 'Z'),
        ('B', '8'), ('8', 'B'),
        ('G', '6'), ('6', 'G'),
    ]

    # Check for confusion pair errors
    for pc, tc in zip(pred_text, target_text):
        if pc != tc:
            if (pc, tc) in confusion_pairs:
                return f'confusion_{pc}_{tc}'

    # Count character errors
    char_errors = sum(1 for pc, tc in zip(pred_text, target_text) if pc != tc)

    # Classify by error count
    if char_errors == 1:
        return 'single_char_error'
    elif char_errors == 2:
        return 'double_char_error'
    else:
        return 'multiple_char_errors'


def main():
    """
    Main function: execute model evaluation pipeline
    Includes path validation, model loading, dataset loading, and result output
    """
    config_loader = get_config()

    parser = argparse.ArgumentParser(description='Model evaluation')
    parser.add_argument('--checkpoint', type=str,
                       help='Model checkpoint path')
    parser.add_argument('--data', type=str,
                       help='Test dataset path')
    parser.add_argument('--batch_size', type=int,
                       help='Batch size')
    parser.add_argument('--warmup_batches', type=int, default=2,
                       help='Number of warmup batches for GPU stability (default: 2)')
    parser.add_argument('--output', type=str,
                       help='Evaluation result output path')
    parser.add_argument('--save_result', action='store_true',
                       help='Save evaluation results to JSON file (for comparison report generation)')

    args = parser.parse_args()

    # Get default values from config using config_loader (absolute paths)
    if args.checkpoint is None:
        args.checkpoint = os.path.join(config_loader.get_checkpoint_dir(), 'best_model.pth')
    if args.data is None:
        args.data = config_loader.get_data_dir('test')
    if args.batch_size is None:
        args.batch_size = config_loader.get('training.batch_size', 64)
    if args.output is None:
        args.output = 'evaluation_results.json'

    logger.info(f"\n{'='*60}")
    logger.info(f"Starting path validation...")
    logger.info(f"{'='*60}")
    try:
        validate_paths(args.checkpoint, args.data)
    except FileNotFoundError as e:
        logger.error(f"\n{str(e)}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n{'='*60}")
        logger.error(f"Error: Unknown error during path validation")
        logger.error(f"{'='*60}")
        logger.error(f"Error message: {str(e)}")
        logger.error(f"\nSuggestions:")
        logger.error(f"1. Check if the input paths are correct")
        logger.error(f"2. Ensure sufficient file system permissions")
        logger.error(f"3. Contact technical support")
        logger.error(f"{'='*60}")
        sys.exit(1)

    device_manager = DeviceManager(prefer_gpu=True, verbose=True)
    device = device_manager.get_device()
    logger.info(f"Using device: {device}")

    dir_manager = DirectoryManager(verbose=True)

    logger.info(f"\n{'='*60}")
    logger.info(f"Loading dataset...")
    logger.info(f"{'='*60}")
    try:
        dataset = CaptchaDataset(
            image_dir=args.data,
            transform=get_val_transform()
        )
    except Exception as e:
        logger.error(f"\n{'='*60}")
        logger.error(f"Error: Failed to load dataset")
        logger.error(f"{'='*60}")
        logger.error(f"Error message: {str(e)}")
        logger.error(f"\nPossible causes:")
        logger.error(f"1. Dataset format is incorrect")
        logger.error(f"2. Dataset files are corrupted")
        logger.error(f"3. Dataset preprocessing configuration error")
        logger.error(f"\nSuggestions:")
        logger.error(f"1. Regenerate the dataset: python generate/generate_dataset.py")
        logger.error(f"2. Check if dataset files are complete")
        logger.error(f"3. Verify transforms configuration")
        logger.error(f"{'='*60}")
        sys.exit(1)

    try:
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            drop_last=True,
            num_workers=0 if os.name == 'nt' else 4,
            pin_memory=True if device_manager.is_cuda() else False,
            persistent_workers=False,
            prefetch_factor=2 if (os.name != 'nt' and device_manager.is_cuda()) else None,
            timeout=30
        )
    except Exception as e:
        logger.error(f"\n{'='*60}")
        logger.error(f"Error: Failed to create DataLoader")
        logger.error(f"{'='*60}")
        logger.error(f"Error message: {str(e)}")
        logger.error(f"\nPossible causes:")
        logger.error(f"1. Batch size too large, exceeds memory limit")
        logger.error(f"2. Dataset is empty")
        logger.error(f"3. collate_fn configuration error")
        logger.error(f"\nSuggestions:")
        logger.error(f"1. Reduce batch size: --batch_size 32")
        logger.error(f"2. Check if dataset contains valid data")
        logger.error(f"3. Verify collate_fn function is correct")
        logger.error(f"{'='*60}")
        sys.exit(1)

    logger.info(f"Dataset size: {len(dataset)}")
    logger.info(f"Character set: {dataset.characters}")
    logger.info(f"Batch size: {args.batch_size}")

    logger.info(f"\n{'='*60}")
    logger.info(f"Loading model...")
    logger.info(f"{'='*60}")

    model_type = 'captcha'
    try:
        if os.path.exists(args.checkpoint):
            try:
                checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
                state_dict = checkpoint.get('model_state_dict', checkpoint)
                model_type = checkpoint.get('model_type', 'auto')
                logger.info(f"Loading model from {args.checkpoint}")
                
                # Auto-detect model type via checkpoint keys
                if model_type == 'auto':
                    keys = list(state_dict.keys())
                    if any('conv_blocks' in k for k in keys):
                        model_type = 'baseline'
                        logger.info(f"  Auto-detected: Baseline model (VGG CNN + BiLSTM)")
                    elif any('backbone' in k for k in keys):
                        model_type = 'captcha'
                        logger.info(f"  Auto-detected: Core model (ConvNeXt V2-Tiny + BiLSTM)")
                    else:
                        logger.warning(f"  Cannot auto-detect model type, using default core model")
                        model_type = 'captcha'

                if model_type == 'baseline':
                    num_classes = checkpoint.get('num_classes', 63)
                    model = BaselineVGGCNNBiLSTM(
                        num_classes=num_classes
                    ).to(device)
                else:
                    # Core model - get correct num_classes from checkpoint
                    fc_weight_shape = state_dict.get('fc.weight', state_dict.get('decoder.fc.weight')).shape
                    num_classes_from_checkpoint = fc_weight_shape[0]
                    model = CaptchaModel(
                        num_chars=num_classes_from_checkpoint - 1,
                        pretrained=False
                    ).to(device)

                model.load_state_dict(state_dict)
                logger.info(f"Model loaded successfully")
            except Exception as e:
                logger.warning(f"Warning: Failed to load checkpoint: {str(e)}")
                logger.warning(f"Using untrained model")
                model = CaptchaModel(
                    num_chars=len(dataset.characters) + 1,
                    pretrained=False
                ).to(device)
        else:
            logger.warning(f"Warning: Checkpoint {args.checkpoint} not found, using untrained model")
            model = CaptchaModel(
                num_chars=len(dataset.characters) + 1,
                pretrained=False
            ).to(device)
    except Exception as e:
        logger.error(f"\n{'='*60}")
        logger.error(f"Error: Failed to load model")
        logger.error(f"{'='*60}")
        logger.error(f"Error message: {str(e)}")
        logger.error(f"\nPossible causes:")
        logger.error(f"1. Checkpoint file is corrupted")
        logger.error(f"2. Model architecture mismatch")
        logger.error(f"3. Insufficient device memory")
        logger.error(f"\nSuggestions:")
        logger.error(f"1. Retrain the model: python models/train.py")
        logger.error(f"2. Check checkpoint file integrity")
        logger.error(f"3. Use view_checkpoint.py to view checkpoint info")
        logger.error(f"4. Try using a smaller batch size")
        logger.error(f"{'='*60}")
        sys.exit(1)

    logger.info(f"\n{'='*60}")
    logger.info(f"Starting evaluation...")
    logger.info(f"{'='*60}")

    # ⚠️ Fix: Determine decoding strategy based on model type
    use_ctc = True
    if 'model_type' in locals() and model_type == 'baseline':
        use_ctc = False
        logger.info("Baseline model detected, using fixed-length decoding strategy")

    try:
        # 🔧 2026-04-06 Fix: Pass dataset.characters instead of label_encoder
        results = evaluate(model, dataloader, device, dataset.characters,
                          warmup_batches=args.warmup_batches, use_ctc=use_ctc)
    except Exception as e:
        logger.error(f"\n{'='*60}")
        logger.error(f"Error: Evaluation process failed")
        logger.error(f"{'='*60}")
        logger.error(f"Error message: {str(e)}")
        logger.error(f"\nPossible causes:")
        logger.error(f"1. Model inference error")
        logger.error(f"2. Data format mismatch")
        logger.error(f"3. Insufficient memory")
        logger.error(f"\nSuggestions:")
        logger.error(f"1. Check if model and dataset are compatible")
        logger.error(f"2. Reduce batch size")
        logger.error(f"3. Check device memory usage")
        logger.error(f"{'='*60}")
        sys.exit(1)

    logger.info(f"\n{'='*60}")
    logger.info(f"Evaluation Results")
    logger.info(f"{'='*60}")
    logger.info(f"Image Accuracy: {results['image_accuracy']:.4f} ({results['image_accuracy']*100:.2f}%)")
    logger.info(f"Char Accuracy: {results['char_accuracy']:.4f} ({results['char_accuracy']*100:.2f}%)")
    logger.info(f"Avg Inference Time: {results['avg_time_ms']:.2f} ms")
    logger.info(f"Inference Time Std Dev: {results['std_time_ms']:.2f} ms")
    logger.info(f"Throughput: {results['throughput']:.2f} samples/sec")
    logger.info(f"{'='*60}")

    logger.info(f"\n{'='*60}")
    logger.info(f"Failure Case Analysis")
    logger.info(f"{'='*60}")
    logger.info(f"Total failure cases: {len(results['failure_cases'])}")

    if len(results['failure_cases']) > 0:
        error_types = {}
        for case in results['failure_cases']:
            error_type = case['error_type']
            error_types[error_type] = error_types.get(error_type, 0) + 1

        logger.info(f"\nError type distribution:")
        for error_type, count in sorted(error_types.items(), key=lambda x: x[1], reverse=True):
            percentage = count / len(results['failure_cases']) * 100
            logger.info(f"  {error_type}: {count} ({percentage:.2f}%)")

        logger.info(f"\nTypical failure cases (first 10):")
        for i, case in enumerate(results['failure_cases'][:10], 1):
            logger.info(f"  {i}. Index: {case['index']}")
            logger.info(f"     Prediction: {case['prediction']}")
            logger.info(f"     Target: {case['target']}")
            logger.info(f"     Error type: {case['error_type']}")
            logger.info(f"     Length diff: {case['length_diff']:+d}")

    logger.info(f"{'='*60}")

    if args.save_result:
        # Save alongside model checkpoint for easier comparison report loading
        checkpoint_dir = os.path.dirname(args.checkpoint)
        checkpoint_name = os.path.splitext(os.path.basename(args.checkpoint))[0]
        output_path = os.path.join(checkpoint_dir, f'{checkpoint_name}_evaluation.json')
        logger.info(f"\nSaving evaluation results to: {output_path}")
    else:
        parent_dir = str(config_loader.get_project_root())
        output_path = os.path.join(parent_dir, args.output)

    output_dir = os.path.dirname(output_path)
    if output_dir:
        dir_manager.ensure_directory(output_dir)

    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump({
                'image_accuracy': results['image_accuracy'],
                'char_accuracy': results['char_accuracy'],
                'avg_time_ms': results['avg_time_ms'],
                'std_time_ms': results['std_time_ms'],
                'throughput': results['throughput'],
                'predictions': results['predictions'],
                'targets': results['targets'],
                'failure_cases': results['failure_cases']
            }, f, ensure_ascii=False, indent=2)
        logger.info(f"\nEvaluation results saved to: {output_path}")
    except Exception as e:
        logger.error(f"\n{'=' * 60}")
        logger.error(f"Warning: Failed to save evaluation results")
        logger.error(f"{'=' * 60}")
        logger.error(f"Error message: {str(e)}")
        logger.error(f"\nSuggestions:")
        logger.error(f"1. Check if the output path has write permission")
        logger.error(f"2. Check if there is sufficient disk space")
        logger.error(f"3. Try using another output path: --output /path/to/output.json")
        logger.error(f"{'=' * 60}")
        # Do not exit the program as evaluation is complete


def evaluate_traditional_method(recognizer, test_images, test_labels, chars):
    """
    Unified traditional method evaluation wrapper

    Returns:
        dict: dictionary with unified metrics
            - image_accuracy: image-level accuracy
            - char_accuracy: character-level accuracy
            - avg_time_ms: average inference time
            - std_time_ms: inference time standard deviation
    """
    predictions = []
    inference_times = []

    for image, label in zip(test_images, test_labels):
        start_time = time.time()
        pred = recognizer.recognize(image)
        end_time = time.time()
        inference_time = (end_time - start_time) * 1000

        predictions.append(pred)
        inference_times.append(inference_time)

    correct_images = sum(1 for p, t in zip(predictions, test_labels) if p == t)
    image_accuracy = correct_images / len(predictions)

    total_chars = 0
    correct_chars = 0
    for pred, target in zip(predictions, test_labels):
        for pc, tc in zip(pred, target):
            total_chars += 1
            if pc == tc:
                correct_chars += 1

    char_accuracy = correct_chars / total_chars if total_chars > 0 else 0.0
    avg_time = np.mean(inference_times)
    std_time = np.std(inference_times)

    return {
        'image_accuracy': image_accuracy,
        'char_accuracy': char_accuracy,
        'avg_time_ms': avg_time,
        'std_time_ms': std_time
    }


if __name__ == '__main__':
    main()
