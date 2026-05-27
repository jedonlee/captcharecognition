# -*- coding: utf-8 -*-
"""
Unified log management system
Function: provides structured logging, supports console output, file output, TensorBoard
"""

import os
import sys
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from datetime import datetime
import yaml


class LoggerManager:
    """Logger manager"""

    _instance = None
    _initialized = False

    def __new__(cls):
        """Singleton pattern"""
        if cls._instance is None:
            cls._instance = super(LoggerManager, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize logger manager"""
        if self._initialized:
            return

        self.loggers = {}
        self.config = self._load_config()
        self._initialized = True

    def _load_config(self):
        """
        Load config file

        Returns:
            config dict
        """
        config_path = Path(__file__).parent.parent / "config.yaml"

        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                return config.get('logging', {})
        else:
            # Default config
            return {
                'log_dir': 'logs',
                'tensorboard': {
                    'enabled': True,
                    'log_dir': 'logs/tensorboard'
                },
                'console': {
                    'level': 'INFO',
                    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                },
                'file': {
                    'enabled': True,
                    'log_dir': 'logs/file',
                    'filename': 'training.log',
                    'level': 'INFO',
                    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    'max_bytes': 10485760,
                    'backup_count': 5
                }
            }

    def get_logger(self, name='captcha_recognition'):
        """
        Get logger

        Args:
            name: logger name

        Returns:
            logger
        """
        if name in self.loggers:
            return self.loggers[name]

        # Create logger
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)

        # Avoid duplicate handlers
        if logger.handlers:
            return logger

        # Console handler
        console_handler = self._create_console_handler()
        logger.addHandler(console_handler)

        # File handler
        if self.config.get('file', {}).get('enabled', True):
            file_handler = self._create_file_handler(name)
            logger.addHandler(file_handler)

        self.loggers[name] = logger
        return logger

    def _create_console_handler(self):
        """
        Create console handler

        Returns:
            console handler
        """
        console_config = self.config.get('console', {})
        level = getattr(logging, console_config.get('level', 'INFO'))
        log_format = console_config.get('format',
                                      '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        handler.setFormatter(logging.Formatter(log_format))

        return handler

    def _create_file_handler(self, name):
        """
        Create file handler

        Args:
            name: logger name

        Returns:
            file handler
        """
        default_log_dir = Path(__file__).parent.parent / 'logs' / 'file'
        file_config = self.config.get('file', {})
        log_dir = file_config.get('log_dir', default_log_dir)
        filename = file_config.get('filename', 'training.log')
        level = getattr(logging, file_config.get('level', 'INFO'))
        log_format = file_config.get('format',
                                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        max_bytes = file_config.get('max_bytes', 10485760)
        backup_count = file_config.get('backup_count', 5)

        # Create log directory
        log_dir.mkdir(parents=True, exist_ok=True)

        # Use timestamp for log filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_filename = f"{timestamp}_{name}_{filename}"
        log_path = log_dir / log_filename

        # Create file handler (with log rotation)
        handler = RotatingFileHandler(
            log_path,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        handler.setLevel(level)
        handler.setFormatter(logging.Formatter(log_format))

        return handler

    def get_tensorboard_dir(self):
        """
        Get TensorBoard log directory

        Returns:
            TensorBoard log directory path
        """
        tensorboard_config = self.config.get('tensorboard', {})
        if tensorboard_config.get('enabled', True):
            default_tensorboard_dir = Path(__file__).parent.parent / 'logs' / 'tensorboard'
            log_dir = tensorboard_config.get('log_dir', default_tensorboard_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            return str(log_dir)
        return None

    def get_log_dir(self):
        """
        Get log root directory

        Returns:
            log root directory path
        """
        log_dir = Path(self.config.get('log_dir', 'logs'))
        log_dir.mkdir(parents=True, exist_ok=True)
        return str(log_dir)


# Global logger manager instance
logger_manager = LoggerManager()


def get_logger(name='captcha_recognition'):
    """
    Get logger (convenience function)

    Args:
        name: logger name

    Returns:
        logger
    """
    return logger_manager.get_logger(name)


def get_tensorboard_dir():
    """
    Get TensorBoard log directory (convenience function)

    Returns:
        TensorBoard log directory path
    """
    return logger_manager.get_tensorboard_dir()


def get_log_dir():
    """
    Get log root directory (convenience function)

    Returns:
        log root directory path
    """
    return logger_manager.get_log_dir()


if __name__ == '__main__':
    # Test logging system
    logger = get_logger('test')

    logger.debug('This is a DEBUG log')
    logger.info('This is an INFO log')
    logger.warning('This is a WARNING log')
    logger.error('This is an ERROR log')
    logger.critical('This is a CRITICAL log')

    print(f'\nTensorBoard log directory: {get_tensorboard_dir()}')
    print(f'Log root directory: {get_log_dir()}')
