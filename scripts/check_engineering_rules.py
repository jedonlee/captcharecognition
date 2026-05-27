#!/usr/bin/env python3
"""Engineering rules check script: verify compliance with unified config, logging, import, path, etc.

Usage:
    cd /root/captcharecognition
    python scripts/check_engineering_rules.py

Rules:
    1. No manual sys.path manipulation (use standard package imports)
    2. No print() outside __main__ block (use logging)
    3. No hardcoded project directory paths (read from config_loader)
"""

import re
import ast
import sys
import logging
from pathlib import Path

# Project root (script is in scripts/, needs to point one level up)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ----- Rule definitions -----
FORBIDDEN_PATTERNS = [
    # Rule 1: No sys.path manipulation
    (r"sys\.path\.(insert|append|pop)",
     "Manual sys.path manipulation is forbidden, use standard package imports"),

    # Rule 2: No print() outside __main__ block
    (r"\bprint\s*\(",
     "Detected print() call, should be replaced with logging module"),

    # Rule 3: No hardcoded path strings
    (r"""['"]data/""",
     "Hardcoded 'data/' path, should use config_loader"),
    (r"""['"]checkpoints/""",
     "Hardcoded 'checkpoints/' path, should use config_loader"),
    (r"""['"]results/""",
     "Hardcoded 'results/' path, should use config_loader"),
    (r"""['"]logs/""",
     "Hardcoded 'logs/' path, should use config_loader"),
]

# Directories and file patterns to exclude
EXCLUDED_DIRS = {".git", "__pycache__", ".trae", "archive", "venv", "env", ".venv"}
EXCLUDED_FILES = {"check_engineering_rules.py"}  # Exclude self

# Function names where print() is explicitly allowed (for CLI/report output purposes)
PRINT_ALLOWED_FUNCTIONS = {
    'print_validation_report',
    'print_metrics_report',
    'print_summary',
    'print_device_info',
    'print_device_summary',
    '_print_summary',
    '_print_device_info',
}


class PrintLocationFinder(ast.NodeVisitor):
    """
    Use AST analysis to find all print() calls and determine if they are in allowed contexts.
    """

    def __init__(self, source_lines: list):
        self.print_lines = []  # (line_no, is_allowed)
        self._current_function = None
        self._main_block_line_ranges = []  # [(start, end), ...]

    def _find_main_blocks(self, tree: ast.AST):
        # Pre-find all if __name__ == '__main__': block ranges.
        for node in ast.walk(tree):
            if isinstance(node, ast.If):
                test = node.test
                is_main = False
                if isinstance(test, ast.Compare):
                    if (isinstance(test.left, ast.Name) and test.left.id == '__name__'
                            and len(test.ops) == 1 and isinstance(test.ops[0], ast.Eq)
                            and len(test.comparators) == 1
                            and isinstance(test.comparators[0], ast.Constant)
                            and test.comparators[0].value == '__main__'):
                        is_main = True
                if is_main:
                    start_line = node.lineno
                    end_line = node.lineno
                    for child in ast.walk(node):
                        if hasattr(child, 'lineno'):
                            end_line = max(end_line, getattr(child, 'end_lineno', child.lineno))
                    self._main_block_line_ranges.append((start_line, end_line))

    def _is_in_main_block(self, line_no: int) -> bool:
        for start, end in self._main_block_line_ranges:
            if start <= line_no <= end:
                return True
        return False

    def _is_in_allowed_function(self) -> bool:
        return self._current_function in PRINT_ALLOWED_FUNCTIONS

    def _is_run_command_method(self, line_no: int) -> bool:
        return self._current_function in ('run_command', 'run_step', 'run_full_pipeline')

    def visit_FunctionDef(self, node):
        old_func = self._current_function
        self._current_function = node.name
        self.generic_visit(node)
        self._current_function = old_func

    def visit_AsyncFunctionDef(self, node):
        old_func = self._current_function
        self._current_function = node.name
        self.generic_visit(node)
        self._current_function = old_func

    def visit_Call(self, node):
        if isinstance(node.func, ast.Name) and node.func.id == 'print':
            line_no = node.lineno
            is_allowed = (
                self._is_in_main_block(line_no)
                or self._is_in_allowed_function()
                or self._is_run_command_method(line_no)
            )
            self.print_lines.append((line_no, is_allowed))
        else:
            self.generic_visit(node)


def find_print_lines_with_ast(content: str) -> list:
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return []

    finder = PrintLocationFinder(content.split('\n'))
    finder._find_main_blocks(tree)
    finder.visit(tree)
    return finder.print_lines


def _is_config_default_value(content: str, match_pos: int) -> bool:
    lines = content[:match_pos].split('\n')
    current_line_idx = len(lines) - 1

    for i in range(current_line_idx, max(0, current_line_idx - 500), -1):
        line = lines[i].rstrip()
        if not line:
            continue
        if '_get_default_config' in line and 'def' in line:
            return True
        if '_load_config' in line and 'def' in line:
            return True
        if re.match(r'\s*def\s+\w+\s*\(', line):
            return False

    return False


def _is_config_get_fallback(content: str, match_pos: int) -> bool:
    start = max(0, match_pos - 200)
    end = min(len(content), match_pos + 20)
    context = content[start:end]

    if re.search(r'(config|self)\.get\s*\([^)]*[,，]\s*[^)]*$', context, re.DOTALL):
        return True
    return False


def check_file(file_path: Path) -> list:
    violations = []
    try:
        content = file_path.read_text(encoding="utf-8")
    except Exception:
        return [f"Cannot read file {file_path}"]

    is_yaml = file_path.suffix in ('.yaml', '.yml')

    ast_print_lines = set()
    ast_print_allowed = set()
    if not is_yaml:
        for line_no, is_allowed in find_print_lines_with_ast(content):
            ast_print_lines.add(line_no)
            if is_allowed:
                ast_print_allowed.add(line_no)

    for pattern, message in FORBIDDEN_PATTERNS:
        matches = list(re.finditer(pattern, content, re.MULTILINE))
        for m in matches:
            line_no = content[:m.start()].count("\n") + 1
            snippet = m.group(0)[:60]

            if "sys.path" in message:
                violations.append(
                    f"{file_path.relative_to(PROJECT_ROOT)}:{line_no} - {message} (found: {snippet})"
                )
                continue

            if "print()" in message:
                if is_yaml:
                    continue
                if line_no in ast_print_allowed:
                    continue
                if line_no in ast_print_lines:
                    violations.append(
                        f"{file_path.relative_to(PROJECT_ROOT)}:{line_no} - {message} (found: {snippet})"
                    )
                continue

            if "Hardcoded" in message:
                if is_yaml:
                    continue
                if _is_config_default_value(content, m.start()):
                    continue
                if _is_config_get_fallback(content, m.start()):
                    continue
                violations.append(
                    f"{file_path.relative_to(PROJECT_ROOT)}:{line_no} - {message} (found: {snippet})"
                )
                continue

    return violations


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logger = logging.getLogger("check_rules")

    logger.info(f"Project root: {PROJECT_ROOT}")
    logger.info("=" * 60)

    violations_all = []
    py_files = sorted(PROJECT_ROOT.rglob("*.py"))
    logger.info(f"Starting check of {len(py_files)} Python files...")

    for fp in py_files:
        if any(part in EXCLUDED_DIRS for part in fp.parts):
            continue
        if fp.name in EXCLUDED_FILES:
            continue

        violations = check_file(fp)
        violations_all.extend(violations)

    logger.info("=" * 60)

    if violations_all:
        logger.error(f"Found {len(violations_all)} violations:")
        for v in violations_all:
            logger.error("  " + v)
        logger.error("")
        logger.error("Please fix before committing/training!")
        sys.exit(1)
    else:
        logger.info("All files passed engineering rules check, project is clean.")
        sys.exit(0)


if __name__ == "__main__":
    main()
