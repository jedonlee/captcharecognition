# -*- coding: utf-8 -*-
"""
General utility functions
Function: path validation and other general utilities
"""

import os
from pathlib import Path


class PathValidator:
    """Path validation utility"""
    
    @staticmethod
    def validate_path(path, path_name="Path", must_exist=True):
        """
        Validate whether a path is valid
        
        Args:
            path: path to validate
            path_name: path name (for error messages)
            must_exist: whether path must exist (default True)
        
        Returns:
            validated path
        
        Raises:
            FileNotFoundError: path does not exist (when must_exist=True)
            ValueError: invalid path
        """
        if not path:
            raise ValueError(f"{path_name} cannot be empty")
        
        path_obj = Path(path)
        
        if must_exist and not path_obj.exists():
            raise FileNotFoundError(f"{path_name} does not exist: {path}")
        
        return str(path_obj)
