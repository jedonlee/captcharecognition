# -*- coding: utf-8 -*-
"""
Directory management module
Function: provides directory creation, permission checking, error handling
Features:
- Supports recursive directory creation
- Automatic permission checking
- Comprehensive error handling
- Supports batch directory creation
- Provides detailed comments
"""

import os
import sys
import stat
import logging
from pathlib import Path
from typing import Union, List, Optional, Dict
from enum import Enum

logger = logging.getLogger(__name__)


class DirectoryError(Exception):
    """Base exception for directory operations"""
    pass


class PermissionError(DirectoryError):
    """Permission error"""
    pass


class CreateDirectoryError(DirectoryError):
    """Directory creation error"""
    pass


class DirectoryStatus(Enum):
    """Directory status enum"""
    EXISTS = "Already exists"
    CREATED = "Created"
    FAILED = "Creation failed"
    NO_PERMISSION = "No permission"


class DirectoryManager:
    """Directory manager

    Functions:
    - Create directories (supports recursive)
    - Check directory permissions
    - Batch create directories
    - Get directory info
    - Clean up directories
    """

    def __init__(self, verbose: bool = True):
        """
        Initialize directory manager

        Args:
            verbose: whether to print detailed info (default True)
        """
        self.verbose = verbose
        self.created_dirs = set()

    def create_directory(self, path: Union[str, Path],
                        parents: bool = True,
                        exist_ok: bool = True,
                        mode: int = 0o755) -> Dict[str, Union[Path, DirectoryStatus, str]]:
        """
        Create a single directory

        Args:
            path: directory path (string or Path object)
            parents: whether to create parent directories (default True, recursive)
            exist_ok: whether to ignore if directory already exists (default True)
            mode: directory permission mode (default 0o755, i.e. rwxr-xr-x)

        Returns:
            dict with directory info:
            {
                'path': Path object,
                'status': DirectoryStatus enum,
                'message': status description
            }

        Raises:
            PermissionError: no permission to create directory
            CreateDirectoryError: directory creation failed
        """
        path_obj = Path(path) if isinstance(path, str) else path

        result = {
            'path': path_obj,
            'status': DirectoryStatus.FAILED,
            'message': ''
        }

        try:
            # Check if directory already exists
            if path_obj.exists():
                if path_obj.is_dir():
                    result['status'] = DirectoryStatus.EXISTS
                    result['message'] = f"Directory already exists: {path_obj.absolute()}"
                    if self.verbose:
                        logger.info(result['message'])
                    return result
                else:
                    raise CreateDirectoryError(f"Path exists but is not a directory: {path_obj.absolute()}")

            # Check parent directory permission
            if parents and path_obj.parent != path_obj:
                if not self._check_parent_permission(path_obj.parent):
                    raise PermissionError(f"No permission to create in parent directory: {path_obj.parent.absolute()}")

            # Create directory
            path_obj.mkdir(parents=parents, exist_ok=exist_ok, mode=mode)

            # Verify directory was created successfully
            if not path_obj.exists() or not path_obj.is_dir():
                raise CreateDirectoryError(f"Directory creation failed: {path_obj.absolute()}")

            result['status'] = DirectoryStatus.CREATED
            result['message'] = f"Directory created successfully: {path_obj.absolute()}"
            self.created_dirs.add(str(path_obj.absolute()))

            if self.verbose:
                logger.info(result['message'])

            return result

        except PermissionError as e:
            result['status'] = DirectoryStatus.NO_PERMISSION
            result['message'] = f"Permission error: {str(e)}"
            if self.verbose:
                logger.error(result['message'])
            raise PermissionError(result['message']) from e

        except Exception as e:
            result['status'] = DirectoryStatus.FAILED
            result['message'] = f"Creation failed: {str(e)}"
            if self.verbose:
                logger.error(result['message'])
            raise CreateDirectoryError(result['message']) from e

    def create_directories(self, paths: List[Union[str, Path]],
                          parents: bool = True,
                          exist_ok: bool = True,
                          mode: int = 0o755) -> Dict[str, Dict]:
        """
        Batch create directories

        Args:
            paths: list of directory paths
            parents: whether to create parent directories (default True)
            exist_ok: whether to ignore if directory already exists (default True)
            mode: directory permission mode (default 0o755)

        Returns:
            dict with all directory creation results:
            {
                'path_string': {
                    'path': Path object,
                    'status': DirectoryStatus enum,
                    'message': status description
                },
                ...
            }
        """
        results = {}

        if self.verbose:
            logger.info(f"\n{'='*60}")
            logger.info(f"Batch creating directories (total {len(paths)})")
            logger.info(f"{'='*60}")

        for path in paths:
            path_str = str(path)
            try:
                results[path_str] = self.create_directory(
                    path, parents=parents, exist_ok=exist_ok, mode=mode
                )
            except Exception as e:
                results[path_str] = {
                    'path': Path(path),
                    'status': DirectoryStatus.FAILED,
                    'message': f"Creation failed: {str(e)}"
                }

        if self.verbose:
            self._print_summary(results)

        return results

    def ensure_directory(self, path: Union[str, Path]) -> Path:
        """
        Ensure directory exists (create if not exists)

        Args:
            path: directory path

        Returns:
            Path object

        Note:
            This is a convenience method, does not raise exceptions
        """
        try:
            result = self.create_directory(path, parents=True, exist_ok=True)
            return result['path']
        except Exception as e:
            if self.verbose:
                logger.warning(f"Failed to ensure directory: {str(e)}")
            return Path(path)

    def _check_parent_permission(self, parent_path: Path) -> bool:
        """
        Check parent directory permission

        Args:
            parent_path: parent directory path

        Returns:
            whether has permission
        """
        try:
            if not parent_path.exists():
                return True

            if not os.access(parent_path, os.W_OK):
                return False

            return True
        except Exception:
            return False

    def _print_summary(self, results: Dict[str, Dict]) -> None:
        """
        Print batch creation summary

        Args:
            results: creation results dict
        """
        logger.info(f"\n{'='*60}")
        logger.info("Creation Summary")
        logger.info(f"{'='*60}")

        status_count = {
            DirectoryStatus.EXISTS: 0,
            DirectoryStatus.CREATED: 0,
            DirectoryStatus.FAILED: 0,
            DirectoryStatus.NO_PERMISSION: 0
        }

        for result in results.values():
            status_count[result['status']] += 1

        logger.info(f"Existing: {status_count[DirectoryStatus.EXISTS]}")
        logger.info(f"Newly created: {status_count[DirectoryStatus.CREATED]}")
        logger.info(f"Failed: {status_count[DirectoryStatus.FAILED]}")
        logger.info(f"No permission: {status_count[DirectoryStatus.NO_PERMISSION]}")
        logger.info(f"{'='*60}\n")

    def get_directory_info(self, path: Union[str, Path]) -> Dict:
        """
        Get directory info

        Args:
            path: directory path

        Returns:
            directory info dict
        """
        path_obj = Path(path) if isinstance(path, str) else path

        info = {
            'path': path_obj.absolute(),
            'exists': path_obj.exists(),
            'is_dir': path_obj.is_dir(),
            'is_writable': False,
            'file_count': 0,
            'size': 0
        }

        if not info['exists']:
            return info

        if not info['is_dir']:
            return info

        try:
            info['is_writable'] = os.access(path_obj, os.W_OK)

            if path_obj.exists():
                files = list(path_obj.iterdir())
                info['file_count'] = len(files)

                for file in files:
                    if file.is_file():
                        info['size'] += file.stat().st_size
        except Exception:
            pass

        return info

    def get_created_directories(self) -> List[str]:
        """
        Get all directories created in this session

        Returns:
            list of directory paths
        """
        return list(self.created_dirs)
    
    def ensure_all_directories(self):
        """
        Ensure all required project directories exist (convenience method)
        """
        from utils.config_loader import get_config
        config = get_config()
        project_root = str(config.get_project_root())
        
        required_dirs = [
            os.path.join(project_root, 'data'),
            os.path.join(project_root, 'data', 'raw'),
            os.path.join(project_root, 'data', 'cleaned'),
            os.path.join(project_root, 'data', 'preprocessed'),
            os.path.join(project_root, 'data', 'augmented'),
            os.path.join(project_root, 'checkpoints'),
            os.path.join(project_root, 'logs'),
            os.path.join(project_root, 'logs', 'training'),
            os.path.join(project_root, 'experiments', 'results'),
            os.path.join(project_root, 'experiments', 'results', 'training_curves'),
            os.path.join(project_root, 'experiments', 'results', 'attention_heatmaps')
        ]
        
        self.create_directories(required_dirs)
    
    @staticmethod
    def ensure_project_directories():
        """
        Static method: ensure all required project directories exist
        """
        dm = DirectoryManager(verbose=True)
        dm.ensure_all_directories()


def ensure_directories(paths: List[Union[str, Path]],
                       verbose: bool = True) -> List[Path]:
    """
    Convenience function: ensure multiple directories exist

    Args:
        paths: list of directory paths
        verbose: whether to print detailed info

    Returns:
        list of Path objects
    """
    manager = DirectoryManager(verbose=verbose)
    results = manager.create_directories(paths)

    return [result['path'] for result in results.values()]


def ensure_directory(path: Union[str, Path],
                    verbose: bool = True) -> Path:
    """
    Convenience function: ensure a single directory exists

    Args:
        path: directory path
        verbose: whether to print detailed info

    Returns:
        Path object
    """
    manager = DirectoryManager(verbose=verbose)
    return manager.ensure_directory(path)


def main():
    """Test directory management module"""
    test_logger = logging.getLogger("directory_manager_test")
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    test_logger.info("=" * 80)
    test_logger.info("🧪 Directory Management Module Test")
    test_logger.info("=" * 80)
    test_logger.info("")

    # Create directory manager
    manager = DirectoryManager(verbose=True)

    # Test 1: Create single directory
    test_logger.info("📋 Test 1: Create single directory")
    test_logger.info("-" * 60)
    try:
            result = manager.create_directory("test_dir_single")
            test_logger.info(f"Status: {result['status'].value}")
            test_logger.info(f"Path: {result['path']}")
        except Exception as e:
            test_logger.error(f"Test failed: {e}")

    test_logger.info("")

    # Test 2: Batch create directories
    test_logger.info("📋 Test 2: Batch create directories")
    test_logger.info("-" * 60)
    try:
        paths = [
            "test_dir_batch/dir1",
            "test_dir_batch/dir2/subdir1",
            "test_dir_batch/dir3/subdir2/subsubdir",
            "test_dir_batch/dir1"  # Duplicate creation
        ]
        results = manager.create_directories(paths)
    except Exception as e:
        test_logger.error(f"Test failed: {e}")

    test_logger.info("")

    # Test 3: Get directory info
    test_logger.info("📋 Test 3: Get directory info")
    test_logger.info("-" * 60)
    try:
        info = manager.get_directory_info("test_dir_single")
        test_logger.info(f"Path: {info['path']}")
        test_logger.info(f"Exists: {info['exists']}")
        test_logger.info(f"Is dir: {info['is_dir']}")
        test_logger.info(f"Writable: {info['is_writable']}")
        test_logger.info(f"File count: {info['file_count']}")
    except Exception as e:
        test_logger.error(f"Test failed: {e}")

    test_logger.info("")

    # Test 4: Convenience functions
    test_logger.info("📋 Test 4: Convenience functions")
    test_logger.info("-" * 60)
    try:
        path = ensure_directory("test_dir_convenient")
        test_logger.info(f"Directory ensured: {path.absolute()}")

        paths = ensure_directories([
            "test_dir_convenient2/dir1",
            "test_dir_convenient2/dir2"
        ])
        test_logger.info(f"Batch directories ensured: {[str(p) for p in paths]}")
    except Exception as e:
        test_logger.error(f"Test failed: {e}")

    test_logger.info("")
    test_logger.info("=" * 80)
    test_logger.info("Test completed")
    test_logger.info("=" * 80)


if __name__ == '__main__':
    main()
