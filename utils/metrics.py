# -*- coding: utf-8 -*-
"""
Evaluation metrics calculation utilities
Function: compute various evaluation metrics for CAPTCHA recognition
"""

import logging
import numpy as np
from typing import List, Tuple, Dict
from utils.chars import CharMapper
from utils.decoder import normalize_equiv

logger = logging.getLogger(__name__)


def calculate_metrics(predictions: List[str], targets: List[str]) -> Tuple[float, float]:
    """
    Calculate evaluation metrics (image-level accuracy and character accuracy)

    Args:
        predictions: list of prediction results
        targets: list of target labels

    Returns:
        image_accuracy: image-level accuracy
        char_accuracy: character-level accuracy
    """
    if len(predictions) != len(targets):
        raise ValueError(f"Mismatched predictions and targets: {len(predictions)} vs {len(targets)}")

    if len(predictions) == 0:
        return 0.0, 0.0

    # Calculate image-level accuracy
    preds_norm = [normalize_equiv(p) for p in predictions]
    targets_norm = [normalize_equiv(t) for t in targets]

    correct_images = sum(1 for p, t in zip(preds_norm, targets_norm) if p == t)
    image_accuracy = correct_images / len(predictions)

    # Calculate character accuracy
    total_chars = 0
    correct_chars = 0

    for pred_text, target_text in zip(preds_norm, targets_norm):
        for p_char, t_char in zip(pred_text, target_text):
            total_chars += 1
            if p_char == t_char:
                correct_chars += 1

    char_accuracy = correct_chars / total_chars if total_chars > 0 else 0.0

    return image_accuracy, char_accuracy


def calculate_precision_recall_f1(predictions: List[str], targets: List[str]) -> Dict[str, float]:
    """
    Calculate precision, recall, F1 score

    Args:
        predictions: list of prediction results
        targets: list of target labels

    Returns:
        dict containing precision, recall, F1 score
    """
    if len(predictions) != len(targets):
        raise ValueError(f"Number of predictions and targets mismatch: {len(predictions)} vs {len(targets)}")

    if len(predictions) == 0:
        return {
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0
        }

    # Calculate character-level precision, recall, F1
    total_chars = 0
    correct_chars = 0
    predicted_chars = 0
    target_chars = 0

    for pred_text, target_text in zip(predictions, targets):
        for p_char, t_char in zip(pred_text, target_text):
            total_chars += 1
            if p_char == t_char:
                correct_chars += 1

        predicted_chars += len(pred_text)
        target_chars += len(target_text)

    # Calculate precision, recall, F1
    precision = correct_chars / predicted_chars if predicted_chars > 0 else 0.0
    recall = correct_chars / target_chars if target_chars > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def calculate_character_accuracy(predictions: List[str], targets: List[str]) -> Dict[str, float]:
    """
    Calculate accuracy per character

    Args:
        predictions: list of prediction results
        targets: list of target labels

    Returns:
        dict containing accuracy per character
    """
    if len(predictions) != len(targets):
        raise ValueError(f"Number of predictions and targets mismatch: {len(predictions)} vs {len(targets)}")

    # Count accuracy per character
    preds_norm = [normalize_equiv(p) for p in predictions]
    targets_norm = [normalize_equiv(t) for t in targets]

    char_stats = {}

    for pred_text, target_text in zip(preds_norm, targets_norm):
        for p_char, t_char in zip(pred_text, target_text):
            if t_char not in char_stats:
                char_stats[t_char] = {'correct': 0, 'total': 0}

            char_stats[t_char]['total'] += 1
            if p_char == t_char:
                char_stats[t_char]['correct'] += 1

    # Calculate accuracy for each character
    char_accuracy = {}
    for char, stats in char_stats.items():
        accuracy = stats['correct'] / stats['total'] if stats['total'] > 0 else 0.0
        char_accuracy[char] = accuracy

    return char_accuracy


def calculate_confusion_matrix(predictions: List[str], targets: List[str], chars: str) -> np.ndarray:
    """
    Calculate confusion matrix

    Args:
        predictions: list of prediction results
        targets: list of target labels
        chars: character set

    Returns:
        confusion matrix (num_classes, num_classes)
    """
    if len(predictions) != len(targets):
        raise ValueError(f"Number of predictions and targets mismatch: {len(predictions)} vs {len(targets)}")

    num_classes = len(chars)
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int32)

    # Use unified CharMapper (instead of local char_to_idx)
    char_mapper = CharMapper.get_instance()

    # Calculate confusion matrix (using normalized equiv-tolerant mapping)
    preds_norm = [normalize_equiv(p) for p in predictions]
    targets_norm = [normalize_equiv(t) for t in targets]

    for pred_text, target_text in zip(preds_norm, targets_norm):
        for p_char, t_char in zip(pred_text, target_text):
            if p_char in char_mapper.char_to_idx and t_char in char_mapper.char_to_idx:
                pred_idx = char_mapper.char_to_idx[p_char]
                target_idx = char_mapper.char_to_idx[t_char]
                confusion_matrix[target_idx, pred_idx] += 1

    return confusion_matrix


def calculate_error_rate(predictions: List[str], targets: List[str]) -> Dict[str, float]:
    """
    Calculate error rate

    Args:
        predictions: list of prediction results
        targets: list of target labels

    Returns:
        dict containing various error rates
    """
    if len(predictions) != len(targets):
        raise ValueError(f"Number of predictions and targets mismatch: {len(predictions)} vs {len(targets)}")

    if len(predictions) == 0:
        return {
            'image_error_rate': 0.0,
            'char_error_rate': 0.0,
            'length_error_rate': 0.0
        }

    # Calculate image error rate
    incorrect_images = sum(1 for p, t in zip(predictions, targets) if p != t)
    image_error_rate = incorrect_images / len(predictions)

    # Calculate character error rate
    total_chars = 0
    incorrect_chars = 0

    for pred_text, target_text in zip(predictions, targets):
        for p_char, t_char in zip(pred_text, target_text):
            total_chars += 1
            if p_char != t_char:
                incorrect_chars += 1

    char_error_rate = incorrect_chars / total_chars if total_chars > 0 else 0.0

    # Calculate length error rate
    length_errors = sum(1 for p, t in zip(predictions, targets) if len(p) != len(t))
    length_error_rate = length_errors / len(predictions)

    return {
        'image_error_rate': image_error_rate,
        'char_error_rate': char_error_rate,
        'length_error_rate': length_error_rate
    }


def print_metrics_report(predictions: List[str], targets: List[str], chars: str = None):
    """
    Print evaluation metrics report

    Args:
        predictions: list of prediction results
        targets: list of target labels
        chars: character set (optional, for computing confusion matrix)
    """
    logger.info("=" * 80)
    logger.info("Evaluation Metrics Report")
    logger.info("=" * 80)
    logger.info("")

    image_accuracy, char_accuracy = calculate_metrics(predictions, targets)

    logger.info(f"Sample count: {len(predictions)}")
    logger.info(f"Image accuracy: {image_accuracy:.4f} ({image_accuracy*100:.2f}%)")
    logger.info(f"Character accuracy: {char_accuracy:.4f} ({char_accuracy*100:.2f}%)")
    logger.info("")

    prf = calculate_precision_recall_f1(predictions, targets)
    logger.info(f"Precision: {prf['precision']:.4f} ({prf['precision']*100:.2f}%)")
    logger.info(f"Recall: {prf['recall']:.4f} ({prf['recall']*100:.2f}%)")
    logger.info(f"F1 score: {prf['f1']:.4f}")
    logger.info("")

    error_rates = calculate_error_rate(predictions, targets)
    logger.info(f"Image error rate: {error_rates['image_error_rate']:.4f} ({error_rates['image_error_rate']*100:.2f}%)")
    logger.info(f"Character error rate: {error_rates['char_error_rate']:.4f} ({error_rates['char_error_rate']*100:.2f}%)")
    logger.info(f"Length error rate: {error_rates['length_error_rate']:.4f} ({error_rates['length_error_rate']*100:.2f}%)")
    logger.info("")

    char_accuracy_dict = calculate_character_accuracy(predictions, targets)
    logger.info("Character accuracy (Top 10):")
    sorted_chars = sorted(char_accuracy_dict.items(), key=lambda x: x[1], reverse=True)
    for i, (char, accuracy) in enumerate(sorted_chars[:10], 1):
        logger.info(f"  {i}. '{char}': {accuracy:.4f} ({accuracy*100:.2f}%)")
    logger.info("")

    if chars is not None:
        confusion_matrix = calculate_confusion_matrix(predictions, targets, chars)
        logger.info(f"Confusion matrix shape: {confusion_matrix.shape}")
        logger.info("")

    logger.info("=" * 80)


if __name__ == '__main__':
    # Test evaluation metrics calculation
    print("Testing evaluation metrics calculation...")

    # Create test data
    predictions = [
        "ABC123",
        "XYZ789",
        "ABC123",
        "XYZ789",
        "ABC123"
    ]
    targets = [
        "ABC123",
        "XYZ789",
        "ABC124",
        "XYZ788",
        "ABC123"
    ]

    # Print evaluation report
    print_metrics_report(predictions, targets, chars="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")

    # Test individual functions
    print("\nTesting individual functions...")

    image_accuracy, char_accuracy = calculate_metrics(predictions, targets)
    print(f"Image accuracy: {image_accuracy:.4f}")
    print(f"Character accuracy: {char_accuracy:.4f}")

    prf = calculate_precision_recall_f1(predictions, targets)
    print(f"Precision: {prf['precision']:.4f}")
    print(f"Recall: {prf['recall']:.4f}")
    print(f"F1 score: {prf['f1']:.4f}")

    print("\nTest passed!")


# ============================================================
# Backward compatibility (for old code)
# ============================================================
def calculate_accuracy(pred_texts, target_texts):
    """
    Calculate accuracy (backward compatibility)
    
    Args:
        pred_texts: list of predicted texts
        target_texts: list of target texts
    
    Returns:
        tuple[float, float]: (image_accuracy, char_accuracy)
    """
    return calculate_metrics(pred_texts, target_texts)
