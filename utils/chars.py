# -*- coding: utf-8 -*-
"""
Unified character mapping manager (single implementation)

Responsibilities:
- Load charset config from chars_config.yaml
- Provide char_to_idx / idx_to_char dict mapping
- Provide encode() / decode() methods
- Global singleton: all modules get the same instance via CharMapper.get_instance()

Usage:
    from utils.chars import CharMapper
    mapper = CharMapper.get_instance()
    indices = mapper.encode("Ab3")
    text = mapper.decode([0, 27, 52])
"""

import os
import yaml


class CharMapper:
    """Unified character mapping manager (singleton pattern)"""

    _instance = None

    def __init__(self, config_path=None):
        """
        Initialize character mapping

        Args:
            config_path: chars_config.yaml path (default auto-locate)
        """
        if config_path is None:
            config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'chars_config.yaml')

        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            self.characters = str(config.get('characters', 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'))
            self.max_length = int(config.get('max_length', 6))
            self.blank_index = int(config.get('blank_index', len(self.characters)))
        else:
            self.characters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'
            self.max_length = 6
            self.blank_index = len(self.characters)

        self.char_to_idx = {char: idx for idx, char in enumerate(self.characters)}
        self.idx_to_char = {idx: char for idx, char in enumerate(self.characters)}
        self.num_classes = len(self.characters)          # 62 (without blank)
        self.total_classes = self.num_classes + 1         # 63 (with blank)

    @classmethod
    def get_instance(cls):
        """Get global singleton instance"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def encode(self, text):
        """Encode text to index list"""
        return [self.char_to_idx[c] for c in text if c in self.char_to_idx]

    def decode(self, indices):
        """Decode index list to text"""
        valid_indices = [i for i in indices if 0 <= i < len(self.characters)]
        return ''.join(self.idx_to_char[i] for i in valid_indices)

    def decode_ctc(self, indices):
        """CTC decode: remove blanks and consecutive duplicates"""
        result = []
        prev = None
        for idx in indices:
            if idx == self.blank_index:
                continue
            if idx != prev:
                result.append(idx)
                prev = idx
        return self.decode(result)

    def __repr__(self):
        return f"CharMapper(classes={self.total_classes}, max_len={self.max_length}, blank={self.blank_index})"


# ============================================================
# Backward compatibility (for old code)
# ============================================================
_mapper_instance = None


def get_mapper():
    """Get CharMapper singleton (backward compatibility)"""
    global _mapper_instance
    if _mapper_instance is None:
        _mapper_instance = CharMapper.get_instance()
    return _mapper_instance


def get_all_chars():
    """Get all characters list (backward compatibility)"""
    mapper = get_mapper()
    return list(mapper.characters)


def char_to_index(char):
    """Char to index (backward compatibility)"""
    mapper = get_mapper()
    return mapper.char_to_idx.get(char, -1)


def index_to_char(idx):
    """Index to char (backward compatibility)"""
    mapper = get_mapper()
    return mapper.idx_to_char.get(idx, '')


# Backward compatibility constants
NUM_CHARS = len(get_all_chars())
MAX_LENGTH = get_mapper().max_length
BLANK_INDEX = get_mapper().blank_index
TOTAL_CLASSES = NUM_CHARS + 1
