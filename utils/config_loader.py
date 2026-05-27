# -*- coding: utf-8 -*-
"""
Configuration loading utility
Function: unified loading and management of config.yaml
Supports environment variable overrides and absolute path resolution
"""

import yaml
import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

logger = logging.getLogger(__name__)


class ConfigLoader:
    """Config loader supporting environment variable overrides and absolute path resolution"""

    _instance = None
    _config = None
    _project_root = None

    def __new__(cls):
        """Singleton pattern"""
        if cls._instance is None:
            cls._instance = super(ConfigLoader, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize config loader"""
        if self._config is not None:
            return

        self._project_root = self._get_project_root()
        self._config = self._load_config()
        self._chars_config = self._load_chars_config()
        self._image_config = self._load_image_config()

    def _get_project_root(self) -> Path:
        """
        Get project root directory
        Priority: env var PROJECT_ROOT > auto-detection

        Returns:
            Path object for project root
        """
        # Priority: environment variable first
        if 'PROJECT_ROOT' in os.environ:
            project_root = Path(os.environ['PROJECT_ROOT'])
            logger.info("Project root from environment variable: %s", project_root)
            return project_root

        # Auto-detect project root
        current_file = Path(__file__)
        # config_loader.py is in utils/, project root is parent of utils/
        project_root = current_file.parent.parent
        logger.info("Auto-detected project root: %s", project_root)
        return project_root

    def _resolve_path(self, path: str) -> Path:
        """
        Resolve path to absolute path
        Supports relative and absolute paths

        Args:
            path: path string

        Returns:
            absolute Path object
        """
        path_obj = Path(path)

        # If absolute, return directly
        if path_obj.is_absolute():
            return path_obj

        # Relative path, resolve based on project root
        return self._project_root / path_obj

    def _load_config(self) -> Dict[str, Any]:
        """
        Load config file
        Supports environment variable overrides

        Returns:
            config dict
        """
        config_path = self._resolve_path("config.yaml")

        logger.info("Config file path: %s", config_path)
        logger.info("Config file exists: %s", config_path.exists())

        if config_path.exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                    if config is None:
                        logger.warning("Config file is empty: %s", config_path)
                        return self._get_default_config()
                    # Apply environment variable overrides
                    config = self._apply_env_overrides(config)
                    # Convert numeric types
                    config = self._convert_numeric_types(config)
                    return config
            except yaml.YAMLError as e:
                logger.error("Config file parsing failed: %s", str(e))
                logger.info("Using default config")
                return self._get_default_config()
            except Exception as e:
                logger.error("Error loading config file: %s", str(e))
                logger.info("Using default config")
                return self._get_default_config()
        else:
            logger.warning("Config file does not exist: %s", config_path)
            logger.info("Using default config")
            return self._get_default_config()

    def _load_chars_config(self) -> Dict[str, Any]:
        """
        Load charset config file
        Load from chars_config.yaml

        Returns:
            charset config dict
        """
        chars_config_path = self._resolve_path("chars_config.yaml")

        logger.info("Charset config file path: %s", chars_config_path)
        logger.info("Charset config file exists: %s", chars_config_path.exists())

        if chars_config_path.exists():
            try:
                with open(chars_config_path, 'r', encoding='utf-8') as f:
                    chars_config = yaml.safe_load(f)
                    if chars_config is None:
                        logger.warning("Charset config file is empty: %s", chars_config_path)
                        return self._get_default_chars_config()
                    return chars_config
            except yaml.YAMLError as e:
                logger.error("Charset config file parsing failed: %s", str(e))
                logger.info("Using default charset config")
                return self._get_default_chars_config()
            except Exception as e:
                logger.error("Error loading charset config file: %s", str(e))
                logger.info("Using default charset config")
                return self._get_default_chars_config()
        else:
            logger.warning("Charset config file does not exist: %s", chars_config_path)
            logger.info("Using default charset config")
            return self._get_default_chars_config()

    def _load_image_config(self) -> Dict[str, Any]:
        """
        Load image size config file
        Load from image_config.yaml

        Returns:
            image size config dict
        """
        image_config_path = self._resolve_path("image_config.yaml")

        logger.info("Image size config file path: %s", image_config_path)
        logger.info("Image size config file exists: %s", image_config_path.exists())

        if image_config_path.exists():
            try:
                with open(image_config_path, 'r', encoding='utf-8') as f:
                    image_config = yaml.safe_load(f)
                    if image_config is None:
                        logger.warning("Image size config file is empty: %s", image_config_path)
                        return self._get_default_image_config()
                    return image_config
            except yaml.YAMLError as e:
                logger.error("Image size config file parsing failed: %s", str(e))
                logger.info("Using default image size config")
                return self._get_default_image_config()
            except Exception as e:
                logger.error("Error loading image size config file: %s", str(e))
                logger.info("Using default image size config")
                return self._get_default_image_config()
        else:
            logger.warning("Image size config file does not exist: %s", image_config_path)
            logger.info("Using default image size config")
            return self._get_default_image_config()

    def _apply_env_overrides(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply environment variable overrides to config
        Env var format: CONFIG_<section>_<key>
        Example: CONFIG_DATA_RAW_DIR=/path/to/data

        Args:
            config: original config dict

        Returns:
            config dict with overrides applied
        """
        for key, value in os.environ.items():
            if key.startswith('CONFIG_'):
                # Parse env var name
                # CONFIG_DATA_RAW_DIR -> ['data', 'raw_dir']
                parts = key[7:].lower().split('_')

                # Recursively set config value
                current = config
                for i, part in enumerate(parts[:-1]):
                    if part not in current:
                        current[part] = {}
                    current = current[part]

                # Set final value (attempt type conversion)
                final_key = parts[-1]
                current[final_key] = self._parse_env_value(value)

                logger.info("Environment variable override: %s = %s", key, value)

        return config

    def _parse_env_value(self, value: str) -> Any:
        """
        Parse environment variable value to appropriate type

        Args:
            value: env var value string

        Returns:
            parsed value (int, float, bool, str)
        """
        # Try parsing as boolean
        if value.lower() in ('true', 'false'):
            return value.lower() == 'true'

        # Try parsing as integer
        try:
            return int(value)
        except ValueError:
            pass

        # Try parsing as float
        try:
            return float(value)
        except ValueError:
            pass

        # Return as string
        return value

    def _convert_numeric_types(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recursively convert numeric types in config
        Ensures string representations of numbers (e.g. '5e-5') are properly converted

        Args:
            config: config dict

        Returns:
            converted config dict
        """
        for key, value in config.items():
            if isinstance(value, dict):
                config[key] = self._convert_numeric_types(value)
            elif isinstance(value, str):
                # Try converting to numeric type
                try:
                    # Try integer
                    int_val = int(value)
                    config[key] = int_val
                    continue
                except ValueError:
                    pass

                try:
                    # Try float (supports scientific notation)
                    float_val = float(value)
                    config[key] = float_val
                except ValueError:
                    pass
        return config

    def _get_default_config(self) -> Dict[str, Any]:
        """
        Get default config

        Returns:
            default config dict
        """
        return {
            'data': {
                'preprocessed_dir': 'data/preprocessed',
                'train_dir': 'data/train',
                'val_dir': 'data/val',
                'test_dir': 'data/test',
                'train_ratio': 0.8,
                'val_ratio': 0.1,
                'test_ratio': 0.1,
            },
            'model': {
                'chars_config_file': 'chars_config.yaml',
                'type': 'speechbrain',
            },
            'training': {
                'num_epochs': 80,
                'batch_size': 256,
                'learning_rate': 0.00008,
                'weight_decay': 0.0001,
                'optimizer': 'adamw',
                'scheduler': 'onecycle',
            },
            'logging': {
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
            },
            'checkpoint': {
                'checkpoint_dir': 'checkpoints',
                'save_best_only': True,
                'monitor': 'val_accuracy',
                'mode': 'max',
                'save_frequency': 5,
                'filename': 'epoch_{epoch}_acc_{accuracy:.4f}.pth'
            },
            'loss': {
                'type': 'hybrid',
                'ctc_weight': 0.6,
                'ce_weight': 0.4,
                'label_smoothing': 0.1,
            },
            'system': {
                'seed': 42,
                'device': 'auto',
                'num_workers': 0,
                'pin_memory': True
            }
        }

    def _get_default_chars_config(self) -> Dict[str, Any]:
        """
        Get default charset config

        Returns:
            default charset config dict
        """
        return {
            'characters': "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789",
            'num_classes': 62,
            'max_length': 6,
            'blank_index': 62,
            'total_classes': 63
        }

    def _get_default_image_config(self) -> Dict[str, Any]:
        """
        Get default image size config

        Returns:
            default image size config dict
        """
        return {
            'original': {
                'width': 160,
                'height': 60,
                'size': [160, 60]
            },
            'preprocessed': {
                'height': 32,
                'width': 128,
                'size': [32, 128]
            },
            'model_input': {
                'batch_size': 1,
                'channels': 3,
                'height': 32,
                'width': 128,
                'shape': [1, 3, 32, 128]
            },
            'augmentation': {
                'train_size': [32, 128],
                'val_size': [32, 128],
                'test_size': [32, 128]
            },
            'traditional': {
                'char_width': 20,
                'char_height': 20,
                'char_size': [20, 20]
            }
        }

    def get(self, key_path: str, default=None) -> Any:
        """
        Get config value

        Args:
            key_path: config key path (dot-separated, e.g. 'data.raw_dir')
            default: default value

        Returns:
            config value
        """
        keys = key_path.split('.')
        value = self._config

        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default

        return value

    def set(self, key_path: str, value: Any) -> None:
        """
        Set config value (supports nested keys, dot-separated)

        Args:
            key_path: config key path (e.g. 'streaming_dataset.enabled')
            value: value to set
        """
        keys = key_path.split('.')
        target = self._config
        for key in keys[:-1]:
            if key not in target or not isinstance(target[key], dict):
                target[key] = {}
            target = target[key]
        target[keys[-1]] = value

    def get_config(self):
        """
        Canonical method: get the full config dict

        Returns:
            dict: full config dict
        """
        return self._config

    def get_data_config(self) -> Dict[str, Any]:
        """Get data config"""
        return self._config.get('data', {})

    def get_model_config(self) -> Dict[str, Any]:
        """Get model config"""
        return self._config.get('model', {})

    def get_training_config(self) -> Dict[str, Any]:
        """Get training config"""
        return self._config.get('training', {})

    def get_logging_config(self) -> Dict[str, Any]:
        """Get logging config"""
        return self._config.get('logging', {})

    def get_checkpoint_config(self) -> Dict[str, Any]:
        """Get checkpoint config"""
        return self._config.get('checkpoint', {})

    def get_system_config(self) -> Dict[str, Any]:
        """Get system config"""
        return self._config.get('system', {})

    def get_chars_config(self) -> Dict[str, Any]:
        """
        Get charset config
        Loaded from chars_config.yaml

        Returns:
            charset config dict, containing:
            - characters: charset string
            - num_classes: number of character classes (without blank)
            - max_length: max sequence length
            - blank_index: blank token index
            - total_classes: total number of classes (including blank)
        """
        return self._chars_config

    def get_characters(self) -> str:
        """Get charset string"""
        return self._chars_config.get('characters',
                "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789")

    def get_num_classes(self) -> int:
        """Get number of character classes without blank (62)"""
        return self._chars_config.get('num_classes', 62)

    def get_total_classes(self) -> int:
        """Get total number of classes including blank (63)"""
        return self._chars_config.get('total_classes', 63)

    def get_blank_idx(self) -> int:
        """Get blank token index"""
        return self.get_total_classes() - 1

    def get_max_length(self) -> int:
        """Get max sequence length"""
        return self._chars_config.get('max_length', 6)

    def get_blank_index(self) -> int:
        """Get blank token index (deprecated, use get_blank_idx)"""
        return self.get_blank_idx()

    def get_image_config(self) -> Dict[str, Any]:
        """
        Get image size config
        Loaded from image_config.yaml

        Returns:
            image size config dict, containing:
            - original: original image size (generation stage)
            - preprocessed: preprocessed image size (model input)
            - model_input: model input tensor shape
            - augmentation: data augmentation sizes
            - traditional: traditional method char sizes
        """
        return self._image_config

    def get_original_image_size(self) -> tuple:
        """
        Get original image size (width x height)

        Returns:
            original image size tuple (width, height)
        """
        original = self._image_config.get('original', {})
        return (original.get('width', 160), original.get('height', 60))

    def get_preprocessed_image_size(self) -> tuple:
        """
        Get preprocessed image size (height x width)

        Returns:
            preprocessed image size tuple (height, width)
        """
        # Priority: config.yaml data.preprocess.output_size first
        output_size = self.get('data.preprocess.output_size')
        if output_size is not None and isinstance(output_size, list) and len(output_size) == 2:
            return (output_size[0], output_size[1])
        # Fallback to image_config.yaml
        preprocessed = self._image_config.get('preprocessed', {})
        return (preprocessed.get('height', 32), preprocessed.get('width', 128))

    def get_preprocess_config(self) -> Dict[str, Any]:
        """Get preprocess config"""
        return self._config.get('data', {}).get('preprocess', {})

    def get_cutmix_config(self) -> Dict[str, Any]:
        """Get CutMix config"""
        return self._config.get('cutmix', {})

    def get_document_degradation_config(self) -> Dict[str, Any]:
        """Get document degradation config"""
        return self._config.get('augmentation', {}).get('document_degradation', {})

    def get_model_input_shape(self) -> tuple:
        """
        Get model input tensor shape (batch_size, channels, height, width)

        Returns:
            model input shape tuple (batch_size, channels, height, width)
        """
        model_input = self._image_config.get('model_input', {})
        return (
            model_input.get('batch_size', 1),
            model_input.get('channels', 3),
            model_input.get('height', 32),
            model_input.get('width', 128)
        )

    def get_augmentation_size(self, mode: str = 'train') -> tuple:
        """
        Get data augmentation size (height x width)

        Args:
            mode: augmentation mode (train, val, test)

        Returns:
            augmentation size tuple (height, width)
        """
        augmentation = self._image_config.get('augmentation', {})
        size_key = f'{mode}_size'
        size = augmentation.get(size_key, [32, 128])
        return (size[0], size[1])

    def get_traditional_char_size(self) -> tuple:
        """
        Get traditional method char size (width x height)

        Returns:
            traditional char size tuple (width, height)
        """
        traditional = self._image_config.get('traditional', {})
        return (traditional.get('char_width', 20), traditional.get('char_height', 20))

    def get_data_dir(self, dir_type: str) -> str:
        """
        Get data directory (returns absolute path)

        Args:
            dir_type: directory type (preprocessed, train, val, test)

        Returns:
            absolute path string
        """
        dir_mapping = {
            'preprocessed': 'data.preprocessed_dir',
            'train': 'data.train_dir',
            'val': 'data.val_dir',
            'test': 'data.test_dir'
        }

        # Handle train/val/test special paths: output to preprocessed/{train,val,test}
        if dir_type in ['train', 'val', 'test']:
            preprocessed_dir = self.get('data.preprocessed_dir')
            preprocessed_path = self._resolve_path(preprocessed_dir)
            return str(preprocessed_path / dir_type)

        key_path = dir_mapping.get(dir_type)
        if key_path:
            relative_path = self.get(key_path)
            return str(self._resolve_path(relative_path))
        return str(self._resolve_path(Path('data') / dir_type))

    def get_log_dir(self) -> str:
        """Get log directory (returns absolute path)"""
        relative_path = self.get('logging.log_dir', 'logs')
        return str(self._resolve_path(relative_path))

    def get_tensorboard_dir(self) -> str:
        """Get TensorBoard directory (returns absolute path)"""
        relative_path = self.get('logging.tensorboard.tensorboard_dir')
        return str(self._resolve_path(relative_path))

    def get_checkpoint_dir(self) -> str:
        """Get checkpoint directory (returns absolute path)"""
        relative_path = self.get('checkpoint.checkpoint_dir', 'checkpoints')
        return str(self._resolve_path(relative_path))

    def get_results_dir(self) -> str:
        """Get results directory (returns absolute path)"""
        relative_path = self.get('experiments.results_dir', 'results')
        return str(self._resolve_path(relative_path))

    def get_device(self) -> str:
        """Get device config"""
        device = self.get('system.device', 'auto')
        if device == 'auto':
            import torch
            return 'cuda' if torch.cuda.is_available() else 'cpu'
        return device

    def get_seed(self) -> int:
        """Get random seed"""
        return self.get('system.seed', 42)

    def get_project_root(self) -> Path:
        """Get project root directory"""
        return self._project_root

    def reload(self):
        """Reload config files"""
        self._project_root = self._get_project_root()
        self._config = self._load_config()
        self._chars_config = self._load_chars_config()
        self._image_config = self._load_image_config()

    @property
    def config(self):
        """Backward compatibility: get full config dict (recommend using get_config())"""
        return self._config

    def __repr__(self) -> str:
        """String representation"""
        return f"ConfigLoader(project_root={self._project_root}, config={self._config})"

    def validate_config(self) -> Tuple[bool, List[str], List[str]]:
        """
        Validate config completeness and correctness

        Returns:
            tuple: (is_valid, errors, warnings)
        """
        errors = []
        warnings = []

        # Validate data config
        if 'data' not in self._config:
            errors.append("Missing data config section")
        else:
            data_errors, data_warnings = self._validate_data_config()
            errors.extend(data_errors)
            warnings.extend(data_warnings)

        # Validate model config
        if 'model' not in self._config:
            errors.append("Missing model config section")
        else:
            model_errors, model_warnings = self._validate_model_config()
            errors.extend(model_errors)
            warnings.extend(model_warnings)

        # Validate training config
        if 'training' not in self._config:
            errors.append("Missing training config section")
        else:
            training_errors, training_warnings = self._validate_training_config()
            errors.extend(training_errors)
            warnings.extend(training_warnings)

        # Validate system config
        if 'system' not in self._config:
            errors.append("Missing system config section")
        else:
            system_errors, system_warnings = self._validate_system_config()
            errors.extend(system_errors)
            warnings.extend(system_warnings)

        # Validate logging config
        if 'logging' not in self._config:
            errors.append("Missing logging config section")
        else:
            logging_errors, logging_warnings = self._validate_logging_config()
            errors.extend(logging_errors)
            warnings.extend(logging_warnings)

        # Validate checkpoint config
        if 'checkpoint' not in self._config:
            errors.append("Missing checkpoint config section")
        else:
            checkpoint_errors, checkpoint_warnings = self._validate_checkpoint_config()
            errors.extend(checkpoint_errors)
            warnings.extend(checkpoint_warnings)

        return len(errors) == 0, errors, warnings

    def _validate_data_config(self) -> Tuple[List[str], List[str]]:
        """
        Validate data config

        Returns:
            tuple: (errors, warnings)
        """
        errors = []
        warnings = []
        data_config = self._config.get('data', {})

        # Validate dataset paths
        required_dirs = ['preprocessed_dir', 'train_dir', 'val_dir', 'test_dir']
        for dir_key in required_dirs:
            if dir_key not in data_config:
                errors.append(f"Missing data.{dir_key} config")
            else:
                dir_path = data_config[dir_key]
                if not isinstance(dir_path, str):
                    errors.append(f"data.{dir_key} should be a string")

        # Validate dataset split ratios
        if 'train_ratio' in data_config and 'val_ratio' in data_config and 'test_ratio' in data_config:
            train_ratio = data_config['train_ratio']
            val_ratio = data_config['val_ratio']
            test_ratio = data_config['test_ratio']

            total = train_ratio + val_ratio + test_ratio
            if abs(total - 1.0) > 0.01:
                errors.append(f"Dataset split ratios should sum to 1.0, currently {total}")

            if not (0 < train_ratio < 1):
                errors.append(f"train_ratio should be in (0,1), currently {train_ratio}")
            if not (0 < val_ratio < 1):
                errors.append(f"val_ratio should be in (0,1), currently {val_ratio}")
            if not (0 < test_ratio < 1):
                errors.append(f"test_ratio should be in (0,1), currently {test_ratio}")

        return errors, warnings

    def _validate_model_config(self) -> Tuple[List[str], List[str]]:
        """
        Validate model config

        Returns:
            tuple: (errors, warnings)
        """
        errors = []
        warnings = []
        model_config = self._config.get('model', {})

        # Validate charset config
        if 'characters' in model_config:
            characters = model_config['characters']
            if not isinstance(characters, str) or len(characters) == 0:
                errors.append("model.characters should be a non-empty string")

        if 'num_chars' in model_config:
            num_chars = model_config['num_chars']
            if not isinstance(num_chars, int) or num_chars <= 0:
                errors.append(f"model.num_chars should be a positive integer, currently {num_chars}")

        if 'max_length' in model_config:
            max_length = model_config['max_length']
            if not isinstance(max_length, int) or max_length <= 0:
                errors.append(f"model.max_length should be a positive integer, currently {max_length}")

        return errors, warnings

    def _validate_training_config(self) -> Tuple[List[str], List[str]]:
        """
        Validate training config

        Returns:
            tuple: (errors, warnings)
        """
        errors = []
        warnings = []
        training_config = self._config.get('training', {})

        # Validate basic training parameters
        if 'num_epochs' in training_config:
            num_epochs = training_config['num_epochs']
            if not isinstance(num_epochs, int) or num_epochs <= 0:
                errors.append(f"training.num_epochs should be a positive integer, currently {num_epochs}")

        if 'batch_size' in training_config:
            batch_size = training_config['batch_size']
            if not isinstance(batch_size, int) or batch_size <= 0:
                errors.append(f"training.batch_size should be a positive integer, currently {batch_size}")

        if 'learning_rate' in training_config:
            lr = training_config['learning_rate']
            if not isinstance(lr, (int, float)) or lr <= 0:
                errors.append(f"training.learning_rate should be a positive number, currently {lr}")

        return errors, warnings

    def _validate_system_config(self) -> Tuple[List[str], List[str]]:
        """
        Validate system config

        Returns:
            tuple: (errors, warnings)
        """
        errors = []
        warnings = []
        system_config = self._config.get('system', {})

        # Validate device config
        if 'device' in system_config:
            device = system_config['device']
            valid_devices = ['auto', 'cpu', 'cuda']
            if device not in valid_devices:
                warnings.append(f"system.device should be one of {valid_devices}, currently {device}")

        # Validate number of workers
        if 'num_workers' in system_config:
            num_workers = system_config['num_workers']
            if not isinstance(num_workers, int) or num_workers < 0:
                errors.append(f"system.num_workers should be a non-negative integer, currently {num_workers}")

        return errors, warnings

    def _validate_logging_config(self) -> Tuple[List[str], List[str]]:
        """
        Validate logging config

        Returns:
            tuple: (errors, warnings)
        """
        errors = []
        warnings = []
        logging_config = self._config.get('logging', {})

        # Validate log directory
        if 'log_dir' in logging_config:
            log_dir = logging_config['log_dir']
            if not isinstance(log_dir, str):
                errors.append(f"logging.log_dir should be a string type, currently {log_dir}")

        return errors, warnings

    def _validate_checkpoint_config(self) -> Tuple[List[str], List[str]]:
        """
        Validate checkpoint config

        Returns:
            tuple: (errors, warnings)
        """
        errors = []
        warnings = []
        checkpoint_config = self._config.get('checkpoint', {})

        # Validate checkpoint directory
        if 'checkpoint_dir' in checkpoint_config:
            checkpoint_dir = checkpoint_config['checkpoint_dir']
            if not isinstance(checkpoint_dir, str):
                errors.append(f"checkpoint.checkpoint_dir should be a string type, currently {checkpoint_dir}")

        return errors, warnings

    def print_validation_report(self):
        """
        Print config validation report
        """
        is_valid, errors, warnings = self.validate_config()

        print("=" * 80)
        print("Configuration Validation Report")
        print("=" * 80)
        print()

        # Error messages
        if errors:
            print("Errors:")
            for i, error in enumerate(errors, 1):
                print(f"  {i}. {error}")
            print()
        else:
            print("No errors found")
            print()

        # Warning messages
        if warnings:
            print("Warnings:")
            for i, warning in enumerate(warnings, 1):
                print(f"  {i}. {warning}")
            print()
        else:
            print("No warnings found")
            print()

        # Summary
        print("=" * 80)
        if is_valid:
            print("Configuration validation passed")
        else:
            print(f"Configuration validation failed, {len(errors)} errors found")
        print("=" * 80)


# Global config loader instance
config_loader = ConfigLoader()


def get_config() -> ConfigLoader:
    """
    Get config loader (convenience function)

    Returns:
        config loader instance
    """
    return config_loader


def get_config_value(key_path: str, default=None) -> Any:
    """
    Get config value (convenience function)

    Args:
        key_path: config key path
        default: default value

    Returns:
        config value
    """
    return config_loader.get(key_path, default)


def get_project_root() -> Path:
    """
    Get project root directory (convenience function)

    Returns:
        Path object for project root
    """
    return config_loader.get_project_root()


if __name__ == '__main__':
    # Test config loader
    config = get_config()

    print("=" * 60)
    print("Config Loader Test")
    print("=" * 60)

    print(f"\nProject Root: {config.get_project_root()}")

    print(f"\nData Configuration:")
    print(f"  Preprocessed Directory: {config.get_data_dir('preprocessed')}")
    print(f"  Training Directory: {config.get_data_dir('train')}")
    print(f"  Validation Directory: {config.get_data_dir('val')}")
    print(f"  Test Directory: {config.get_data_dir('test')}")

    print(f"\nModel Configuration:")
    print(f"  Character Set: {config.get_characters()}")
    print(f"  Number of Classes: {config.get_num_classes()}")
    print(f"  Max Length: {config.get_max_length()}")

    print(f"\nImage Size Configuration:")
    print(f"  Original Image Size: {config.get_original_image_size()}")
    print(f"  Preprocessed Image Size: {config.get_preprocessed_image_size()}")
    print(f"  Model Input Shape: {config.get_model_input_shape()}")
    print(f"  Augmentation Size (Train): {config.get_augmentation_size('train')}")
    print(f"  Augmentation Size (Val): {config.get_augmentation_size('val')}")
    print(f"  Traditional Char Size: {config.get_traditional_char_size()}")

    print(f"\nTraining Configuration:")
    print(f"  Num Epochs: {config.get('training.num_epochs')}")
    print(f"  Batch Size: {config.get('training.batch_size')}")
    print(f"  Learning Rate: {config.get('training.learning_rate')}")

    print(f"\nLogging Configuration:")
    print(f"  Log Directory: {config.get_log_dir()}")
    print(f"  TensorBoard Directory: {config.get_tensorboard_dir()}")

    print(f"\nSystem Configuration:")
    print(f"  Device: {config.get_device()}")
    print(f"  Random Seed: {config.get_seed()}")

    print(f"\nCheckpoint Configuration:")
    print(f"  Checkpoint Directory: {config.get_checkpoint_dir()}")

    print("\n" + "=" * 60)
    print("Test completed!")
    print("=" * 60)
