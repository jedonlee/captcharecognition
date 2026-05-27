# -*- coding: utf-8 -*-
"""
Fixed hybrid loss function (HybridLossFixed)
Function: Combines CTC Loss and CrossEntropy Loss for CAPTCHA recognition

Fix history:
1. P0 fix: Smart detection of log-probability input, avoid repeated log_softmax
   - model.py applies stable_log_softmax() internally
   - This module auto-detects max_val ≤ 0 to determine if input is log probability
2. 🔧 2026-04-06 H-2: dynamic blank parameter
   - Default changed from hardcode=0 to None → auto-fetches from CharMapper.blank_index(62)
   - Prevents blank token from occupying character '0' index, causing digit 0 recognition failure
3. Input assertion: encoder_out max ≤ 0 check, prevents non-log_softmax input

Weight config:
- Default ctc_weight=0.35, ce_weight=0.65 (aligned with config.yaml / config_loader defaults)
- train.py reads config at runtime
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class HybridLossFixed(nn.Module):
    def __init__(self, blank=None, ctc_weight=0.6, ce_weight=0.4, label_smoothing=0.1, num_chars=None, *args, **kwargs):
        """
        Initialize hybrid loss function

        Args:
            blank: blank token index (default None, auto-fetch blank_index=62 from CharMapper)
            ctc_weight: CTC loss weight (default 0.6, aligned with config.yaml)
            ce_weight: CE loss weight (default 0.4, aligned with config.yaml)
            label_smoothing: label smoothing factor (default 0.1)
            num_chars: number of character classes (backward compatibility)
        """
        super().__init__()
        
        if blank is None:
            from utils.chars import CharMapper
            blank = CharMapper.get_instance().blank_index
        
        self.ctc_weight = ctc_weight
        if ce_weight is not None:
            self.ce_weight = ce_weight
        else:
            self.ce_weight = 1.0 - ctc_weight
        self.blank = blank
        # CTC loss with dynamic blank (avoid hardcode=0 occupying '0' index)
        self.ctc_loss = nn.CTCLoss(blank=blank, zero_infinity=True)
        # CE loss with label smoothing
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        
    def forward(self, *args):
        """
        Parameter parsing:
        - train.py passes 5 args: encoder_out, decoder_out, targets, input_lengths, targets_lengths
        - We need:
          CTC: logits (encoder_out transposed), targets (1D), input_lengths, target_lengths
          CE: decoder_out, padded_targets (B, T_dec)
        """
        if len(args) == 5:
            # Parameter order: encoder_out, decoder_out, targets, input_lengths, targets_lengths
            encoder_out, decoder_out, targets, input_lengths, target_lengths = args
            
            # Validate input dimensions
            if len(encoder_out.shape) != 3:
                raise ValueError(f"encoder_out should be 3D (T, B, C), got {encoder_out.shape}")
            if len(decoder_out.shape) != 3:
                raise ValueError(f"decoder_out should be 3D (B, T_dec, C), got {decoder_out.shape}")
            
            T, B, C = encoder_out.shape
            B_dec, T_dec, C_dec = decoder_out.shape
            
            if B != B_dec:
                raise ValueError(f"Batch size mismatch: encoder_out B={B}, decoder_out B={B_dec}")
            if C != C_dec:
                raise ValueError(f"Number of classes mismatch: encoder_out C={C}, decoder_out C={C_dec}")

            # 🔧 2026-04-06 Bug prevention: input validation assertion
            # Detect if encoder_out is log probability (model.py applies stable_log_softmax)
            # If max>0, input is raw logits not log probability, raise error immediately
            enc_max = encoder_out.max().item()
            assert enc_max <= 0.0 + 1e-6, (
                f"[HybridLoss] encoder_out max={enc_max:.4f} > 0! "
                f"Input must be log_softmax output (all values ≤ 0). "
                f"Check model.py: stable_log_softmax() is applied before returning encoder_out."
            )

            # encoder_out is (T, B, C), transpose to (B, T, C) for CTC logits
            ctc_logits = encoder_out.permute(1, 0, 2)  # (B, T, C)
            
            # Validate targets dimension (should be 1D concatenated sequence)
            if targets.dim() > 1:
                targets = targets.reshape(-1)
            
            # Validate input_lengths and target_lengths
            if input_lengths.shape[0] != B:
                raise ValueError(f"input_lengths length ({input_lengths.shape[0]}) must equal batch_size ({B})")
            if target_lengths.shape[0] != B:
                raise ValueError(f"target_lengths length ({target_lengths.shape[0]}) must equal batch_size ({B})")
            
            # Validate input_lengths do not exceed time steps
            if (input_lengths > T).any():
                raise ValueError(f"input_lengths contains values exceeding time steps ({T}): {input_lengths}")
            
            # Compute CTC loss
            # 🔧 2026-04-06 Fix: smart detection of log-probability input
            # Detection principle:
            #   - If input is raw logits (direct linear layer output), range is (-∞, +∞), typically has positive values
            #   - If input is log softmax output, range is (-∞, 0], all values ≤ 0
            #   - model.py:679 applies stable_log_softmax(), so encoder_out is already log probability
            #
            # Safety strategy:
            #   if max_val <= 0 and min_val > -100 → log probability → use directly
            #   else → raw logits → apply log_softmax

            max_val = ctc_logits.max().item()
            min_val = ctc_logits.min().item()

            is_log_probability = (max_val <= 0.0) and (min_val > -100.0)

            if is_log_probability:
                # Input is already log probability (from model.py's stable_log_softmax), use directly
                ctc_log_probs = ctc_logits.permute(1, 0, 2).float()
            else:
                # Input is raw logits, apply log_softmax
                ctc_log_probs = F.log_softmax(ctc_logits, dim=2).permute(1, 0, 2).float()
            ctc_loss = self.ctc_loss(ctc_log_probs, targets, input_lengths, target_lengths)
            
            # Compute CE loss
            # Need to create padded targets for decoder_out
            # targets is concatenated 1D sequence, pad to (B, T_dec) based on target_lengths
            padded_targets = torch.full((B, T_dec), self.ctc_loss.blank, dtype=torch.long, device=decoder_out.device)
            
            # Fill targets into padded_targets
            targets_cpu = targets.cpu() if targets.is_cuda else targets
            target_lengths_cpu = target_lengths.cpu() if target_lengths.is_cuda else target_lengths
            
            offset = 0
            for i in range(B):
                length = target_lengths_cpu[i].item()
                if length > 0:
                    # Only fill valid length, not exceeding T_dec
                    valid_length = min(length, T_dec)
                    padded_targets[i, :valid_length] = targets_cpu[offset:offset+valid_length]
                    offset += length
            
            # Compute CE loss
            ce_loss = self.ce_loss(decoder_out.reshape(-1, C_dec), padded_targets.reshape(-1))
            
            # Hybrid loss
            total_loss = self.ctc_weight * ctc_loss + self.ce_weight * ce_loss
            
            return total_loss, ctc_loss, ce_loss
            
        elif len(args) == 4:
            # Parameter order: logits, targets, target_lengths, input_lengths
            logits, targets, target_lengths, input_lengths = args
            
            # Simplified version, CTC loss only
            if logits.dim() != 3:
                raise ValueError(f"logits should be 3D (B, T, C), got {logits.shape}")
            
            B, T, C = logits.shape
            
            # Validate targets dimension
            if targets.dim() > 1:
                targets = targets.reshape(-1)
            
            # Validate input_lengths and target_lengths
            if input_lengths.shape[0] != B:
                raise ValueError(f"input_lengths length ({input_lengths.shape[0]}) must equal batch_size ({B})")
            if target_lengths.shape[0] != B:
                raise ValueError(f"target_lengths length ({target_lengths.shape[0]}) must equal batch_size ({B})")
            
            # Compute CTC loss
            # 🔧 2026-04-06 Fix: smart detection of log-probability input (consistent with 5-param version)
            max_val = logits.max().item()
            min_val = logits.min().item()

            is_log_probability = (max_val <= 0.0) and (min_val > -100.0)

            if is_log_probability:
                # Input is already log probability, use directly
                ctc_log_probs = logits.permute(1, 0, 2).float()
            else:
                # Input is raw logits, apply log_softmax
                ctc_log_probs = F.log_softmax(logits, dim=2).permute(1, 0, 2).float()
            ctc_loss = self.ctc_loss(ctc_log_probs, targets, input_lengths, target_lengths)
            
            # Simplified version returns CTC loss only
            total_loss = ctc_loss
            ce_loss = torch.tensor(0.0, device=logits.device)
            
            return total_loss, ctc_loss, ce_loss
            
        else:
            raise ValueError(f"HybridLossFixed expects 4 or 5 arguments, but got {len(args)}")


# Test function
def test_hybrid_loss():
    """Test fixed hybrid loss function"""
    logger.info("=" * 60)
    logger.info("Testing fixed HybridLossFixed")
    logger.info("=" * 60)
    
    # Simulate train.py data
    B = 2  # Batch size
    T = 4  # encoder_out time steps
    T_dec = 6  # decoder_out time steps
    C = 63  # Number of classes
    blank_idx = 62
    
    # Create mock data
    encoder_out = torch.randn(T, B, C)
    decoder_out = torch.randn(B, T_dec, C)
    
    # Simulate labels: two samples, lengths 4 and 6
    target_lengths = torch.tensor([4, 6])
    targets = torch.cat([
        torch.randint(0, C-1, (4,)),
        torch.randint(0, C-1, (6,))
    ])
    input_lengths = torch.full((B,), T, dtype=torch.long)
    
    # Create loss function
    criterion = HybridLossFixed(blank=blank_idx, ctc_weight=0.35, label_smoothing=0.1)
    
    try:
        # Test 5-param call
        total_loss, ctc_loss, ce_loss = criterion(
            encoder_out, decoder_out, targets, input_lengths, target_lengths
        )
        
        logger.info(f"Test passed!")
        logger.info(f"  total_loss: {total_loss.item():.4f}")
        logger.info(f"  ctc_loss: {ctc_loss.item():.4f}")
        logger.info(f"  ce_loss: {ce_loss.item():.4f}")
        logger.info(f"  Parameter compatibility: ✓")

    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    test_hybrid_loss()


# ============================================================
# Backward compatibility alias (for legacy code)
# ============================================================
HybridCTCELoss = HybridLossFixed