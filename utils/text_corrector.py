# -*- coding: utf-8 -*-
"""
CAPTCHA text corrector - post-processing correction based on model confidence and confusion rules

Core principle:
- Use model output per-position log probabilities
- Check Top-1 prediction confidence at each position
- If confidence is low and confusion chars exist, try replacing with Top-2/Top-3 confusion chars
- Select the replacement that maximizes overall sequence probability

Usage:
    from utils.text_corrector import TextCorrector
    
    corrector = TextCorrector()
    corrected_text = corrector.correct_from_logits(
        text='aB3dEf',
        log_probs=log_probs_tensor,  # (seq_len, num_classes)
        char_mapper=char_mapper
    )
"""

import torch
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


class TextCorrector:
    """CAPTCHA text corrector"""
    
    def __init__(self):
        """
        Initialize the corrector
        """
        # Confusion character pairs - based on visual similarity (most common in CAPTCHAs)
        self.confusion_groups = [
            {'O', '0'},
            {'l', '1', 'I'},
            {'b', '6'},
            {'g', '9', 'q'},
            {'Z', '2'},
            {'S', '5'},
            {'u', 'v'},
            {'c', 'e'},
            {'r', 'v'},
            {'B', '8'},
            {'D', '0'},
            {'G', '6'},
        ]
        
        # Build confusion map: char -> other chars in the same group
        self.confusion_map = defaultdict(set)
        for group in self.confusion_groups:
            for c in group:
                for other in group:
                    if c != other:
                        self.confusion_map[c].add(other)
        
        logger.info(f"TextCorrector initialized, {len(self.confusion_groups)} confusion pairs, "
                    f"covering {len(self.confusion_map)} characters")
    
    def _build_char_to_idx(self, char_mapper):
        """Build character to index mapping"""
        return {c: i for i, c in enumerate(char_mapper.characters)}
    
    def _get_top_k_chars(self, log_probs_at_position, char_mapper, k=5):
        """Get Top-K characters and their probabilities at a position"""
        probs = torch.exp(log_probs_at_position)
        top_probs, top_indices = torch.topk(probs, min(k, len(probs)))
        
        results = []
        for prob, idx in zip(top_probs, top_indices):
            idx_item = idx.item()
            if idx_item < len(char_mapper.characters):
                char = char_mapper.characters[idx_item]
                results.append((char, prob.item(), idx_item))
        
        return results
    
    def correct_from_logits(self, text, log_probs, char_mapper, confidence_threshold=0.7):
        """
        Correction method based on per-position logits
        
        Args:
            text: beam search output text
            log_probs: per-position log probabilities, shape (seq_len, num_classes)
            char_mapper: CharMapper instance
            confidence_threshold: confidence threshold, attempt correction below this
            
        Returns:
            str: corrected text
        """
        if not text or len(text) == 0:
            return text
        
        char_to_idx = self._build_char_to_idx(char_mapper)
        
        # Calculate original sequence log probability
        original_log_prob = 0.0
        position_chars = []
        
        for pos, char in enumerate(text):
            if pos >= len(log_probs):
                position_chars.append((char, 0.0, -1))
                continue
            
            char_idx = char_to_idx.get(char, -1)
            if char_idx >= 0 and char_idx < len(log_probs[pos]):
                char_log_prob = log_probs[pos][char_idx].item()
                original_log_prob += char_log_prob
                position_chars.append((char, char_log_prob, char_idx))
            else:
                position_chars.append((char, -99.0, -1))
        
        # Attempt correction for each low-confidence position
        best_text = text
        best_log_prob = original_log_prob
        
        for pos, (char, char_log_prob, char_idx) in enumerate(position_chars):
            if char_idx < 0:
                continue
            
            # Check confidence (convert to probability)
            confidence = torch.exp(torch.tensor(char_log_prob)).item()
            
            # If confidence is high, no correction needed
            if confidence > confidence_threshold:
                continue
            
            # Check if char has confusion candidates
            if char not in self.confusion_map:
                continue
            
            # Get Top-K candidates
            top_k = self._get_top_k_chars(log_probs[pos], char_mapper, k=5)
            
            # Try replacing with confusion chars from Top-K
            for cand_char, cand_prob, cand_idx in top_k:
                if cand_char == char:
                    continue
                if cand_char not in self.confusion_map.get(char, set()):
                    continue
                
                # Calculate new sequence log probability after replacement
                new_log_prob = original_log_prob - char_log_prob + log_probs[pos][cand_idx].item()
                
                if new_log_prob > best_log_prob:
                    best_log_prob = new_log_prob
                    best_text = text[:pos] + cand_char + text[pos+1:]
        
        return best_text
    
    def correct(self, primary_text, candidates, log_probs):
        """
        Correct candidate texts (compatible with old interface, but limited effect)
        
        Args:
            primary_text: beam search best text
            candidates: beam search candidate text list (Top-K)
            log_probs: corresponding log probabilities from beam search
            
        Returns:
            str: corrected best text
        """
        # This interface lacks per-position logits, effect is limited
        # Directly return primary_text, actual correction done by correct_from_logits
        return primary_text
    
    def apply_confusion_rules(self, text):
        """
        Apply confusion character rules to text (simple rules without logits)
        
        Args:
            text: input text
            
        Returns:
            str: rule-corrected text
        """
        result = list(text)
        
        for i in range(len(result)):
            c = result[i]
            if self.is_confusable(c):
                confusion_group = self.get_confusion_group(c)
                has_digit = any(ch.isdigit() for ch in confusion_group)
                has_alpha = any(ch.isalpha() for ch in confusion_group)
                
                if has_digit and has_alpha:
                    neighbors = []
                    if i > 0:
                        neighbors.append(text[i-1])
                    if i < len(text) - 1:
                        neighbors.append(text[i+1])
                    
                    neighbor_digits = sum(1 for n in neighbors if n.isdigit())
                    neighbor_alphas = sum(1 for n in neighbors if n.isalpha())
                    
                    if neighbor_digits > neighbor_alphas:
                        for ch in confusion_group:
                            if ch.isdigit():
                                result[i] = ch
                                break
                    elif neighbor_alphas > neighbor_digits:
                        for ch in confusion_group:
                            if ch.isalpha():
                                result[i] = ch
                                break
        
        return ''.join(result)
    
    def get_confusion_group(self, char):
        """Get the confusion group for a character"""
        for group in self.confusion_groups:
            if char in group:
                return group
        return {char}
    
    def is_confusable(self, char):
        """Check if a character has confusion potential"""
        return char in self.confusion_map


# Singleton pattern
_corrector_instance = None

def get_corrector():
    """Get TextCorrector singleton"""
    global _corrector_instance
    if _corrector_instance is None:
        _corrector_instance = TextCorrector()
    return _corrector_instance
