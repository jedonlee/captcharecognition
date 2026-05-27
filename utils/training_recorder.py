#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training recorder - for recording and saving training metrics
Functions:
- Record training and validation metrics (loss, accuracy, etc.)
- Save training records to JSON files
- Load training records for subsequent analysis
- Support append mode (resume training)
"""

import os
import json
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


class TrainingRecorder:
    """Training recorder"""

    def __init__(self, experiment_name: str = None, output_dir: str = 'experiments/records'):
        """
        Initialize training recorder

        Args:
            experiment_name: experiment name (default: current timestamp)
            output_dir: output directory (default: experiments/records)
        """
        if experiment_name is None:
            experiment_name = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.experiment_name = experiment_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize record data
        self.record = {
            'experiment_name': experiment_name,
            'start_time': datetime.now().isoformat(),
            'end_time': None,
            'epochs': [],
            'train_loss': [],
            'train_ctc_loss': [],
            'train_ce_loss': [],
            'val_loss': [],
            'val_ctc_loss': [],
            'val_ce_loss': [],
            'val_image_acc': [],
            'val_char_acc': [],
            'best_val_image_acc': 0.0,
            'best_val_char_acc': 0.0,
            'best_epoch': 0,
            'learning_rate': [],
            'config': {}
        }
        
        logger.info(f"Training recorder initialized")
        logger.info(f"  Experiment: {experiment_name}")
        logger.info(f"  Output dir: {self.output_dir.absolute()}")

    def set_config(self, config: Dict[str, Any]):
        """
        Set training config

        Args:
            config: config dict
        """
        self.record['config'] = config

    def record_epoch(self, epoch: int, 
                     train_loss: float, train_ctc_loss: float, train_ce_loss: float,
                     val_loss: float, val_ctc_loss: float, val_ce_loss: float,
                     val_image_acc: float, val_char_acc: float,
                     learning_rate: float = None):
        """
        Record metrics for one epoch

        Args:
            epoch: epoch number (starting from 0)
            train_loss: total training loss
            train_ctc_loss: training CTC loss
            train_ce_loss: training CE loss
            val_loss: total validation loss
            val_ctc_loss: validation CTC loss
            val_ce_loss: validation CE loss
            val_image_acc: validation image accuracy
            val_char_acc: validation character accuracy
            learning_rate: learning rate (optional)
        """
        self.record['epochs'].append(epoch + 1)  # Convert to 1-based
        self.record['train_loss'].append(float(train_loss))
        self.record['train_ctc_loss'].append(float(train_ctc_loss))
        self.record['train_ce_loss'].append(float(train_ce_loss))
        self.record['val_loss'].append(float(val_loss))
        self.record['val_ctc_loss'].append(float(val_ctc_loss))
        self.record['val_ce_loss'].append(float(val_ce_loss))
        self.record['val_image_acc'].append(float(val_image_acc))
        self.record['val_char_acc'].append(float(val_char_acc))
        
        if learning_rate is not None:
            self.record['learning_rate'].append(float(learning_rate))
        
        # Update best accuracy
        if val_image_acc > self.record['best_val_image_acc']:
            self.record['best_val_image_acc'] = float(val_image_acc)
            self.record['best_val_char_acc'] = float(val_char_acc)
            self.record['best_epoch'] = epoch + 1
        
        # Auto-save
        self.save()

    def save(self, filename: str = None):
        """
        Save training record to file

        Args:
            filename: filename (default: {experiment_name}.json)
        """
        if filename is None:
            filename = f"{self.experiment_name}.json"
        
        file_path = self.output_dir / filename
        
        # Update end time
        self.record['end_time'] = datetime.now().isoformat()
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(self.record, f, ensure_ascii=False, indent=2)
            logger.info(f"Training record saved: {file_path}")
            return str(file_path)
        except Exception as e:
            logger.warning(f"Failed to save training record: {e}")
            return None

    @classmethod
    def load(cls, file_path: str):
        """
        Load training record from file

        Args:
            file_path: file path

        Returns:
            TrainingRecorder instance or None (on failure)
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                record_data = json.load(f)
            
            # Create new instance and populate data
            recorder = cls(
                experiment_name=record_data.get('experiment_name'),
                output_dir=os.path.dirname(file_path)
            )
            recorder.record = record_data
            
            logger.info(f"Training record loaded: {file_path}")
            return recorder
        except Exception as e:
            logger.warning(f"Failed to load training record: {e}")
            return None

    def get_summary(self) -> Dict[str, Any]:
        """
        Get training summary

        Returns:
            summary dict
        """
        return {
            'experiment_name': self.record['experiment_name'],
            'start_time': self.record['start_time'],
            'end_time': self.record['end_time'],
            'num_epochs': len(self.record['epochs']),
            'best_epoch': self.record['best_epoch'],
            'best_val_image_acc': self.record['best_val_image_acc'],
            'best_val_char_acc': self.record['best_val_char_acc'],
            'final_val_image_acc': self.record['val_image_acc'][-1] if self.record['val_image_acc'] else 0,
            'final_val_char_acc': self.record['val_char_acc'][-1] if self.record['val_char_acc'] else 0,
        }

    def print_summary(self):
        """Print training summary"""
        summary = self.get_summary()
        logger.info("\n" + "=" * 80)
        logger.info("Training Summary")
        logger.info("=" * 80)
        logger.info(f"Experiment: {summary['experiment_name']}")
        logger.info(f"Start time: {summary['start_time']}")
        logger.info(f"End time: {summary['end_time']}")
        logger.info(f"Epochs: {summary['num_epochs']}")
        logger.info(f"Best epoch: {summary['best_epoch']}")
        logger.info(f"Best image accuracy: {summary['best_val_image_acc']:.4f} ({summary['best_val_image_acc']*100:.2f}%)")
        logger.info(f"Best char accuracy: {summary['best_val_char_acc']:.4f} ({summary['best_val_char_acc']*100:.2f}%)")
        logger.info(f"Final image accuracy: {summary['final_val_image_acc']:.4f} ({summary['final_val_image_acc']*100:.2f}%)")
        logger.info(f"Final char accuracy: {summary['final_val_char_acc']:.4f} ({summary['final_val_char_acc']*100:.2f}%)")
        logger.info("=" * 80 + "\n")
