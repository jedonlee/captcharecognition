# Contributing to CAPTCHA Recognition System

Thank you for your interest in contributing to this project! This document provides guidelines and information for contributors.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How to Contribute](#how-to-contribute)
- [Development Setup](#development-setup)
- [Code Style](#code-style)
- [Commit Messages](#commit-messages)
- [Pull Requests](#pull-requests)
- [Issues](#issues)

## Code of Conduct

Please be respectful and inclusive in all interactions. We are committed to providing a welcoming and inspiring community for everyone.

## How to Contribute

1. **Fork the repository**
2. **Create a feature branch** from `main`
3. **Make your changes**
4. **Write or update tests** if applicable
5. **Update documentation** if needed
6. **Submit a pull request**

## Development Setup

```bash
# Clone your fork
git clone https://github.com/your-username/captcharecognition.git
cd captcharecognition

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Run the project
python main.py --mode full
```

## Code Style

- Follow PEP 8 guidelines
- Use `pathlib.Path` for file paths (no string concatenation)
- Use `logging` module instead of `print()` (except in `__main__` blocks)
- Read all configuration through `config_loader.py` (no hardcoding)
- Write docstrings for public functions and classes
- Use type hints where appropriate

### Project Rules

Please follow the rules defined in `.trae/rules/` directory:
- Rule 1: Change impact analysis
- Rule 2: Configuration management
- Rule 3: Logging standards
- Rule 4: Parameter validation
- Rule 5: Model checkpoint management
- Rule 6: Resource management
- Rule 7: Process management
- Rule 8: Error handling
- Rule 9: API standards
- Rule 10: Security
- Rule 11: Testing
- Rule 12: Performance monitoring
- Rule 13: Docker deployment

## Commit Messages

Follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

### Types

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

### Examples

```
feat(model): add Transformer encoder support
fix(decoder): correct beam search decoding issue
docs(readme): update API documentation
```

## Pull Requests

1. **Keep PRs focused**: One feature or fix per PR
2. **Write descriptive titles**: Summarize the change clearly
3. **Add description**: Explain what and why, not how
4. **Link issues**: Reference related issues with `#issue-number`
5. **Ensure CI passes**: All checks must be green
6. **Request review**: Ask for review from maintainers

### PR Template

```markdown
## Description

Brief description of the changes.

## Type of Change

- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Refactoring
- [ ] Performance improvement

## Testing

Describe the tests you ran to verify your changes.

## Checklist

- [ ] Code follows project style guidelines
- [ ] Self-reviewed the code
- [ ] Added comments for complex logic
- [ ] Updated documentation if needed
- [ ] All tests pass
```

## Issues

### Bug Reports

When reporting bugs, please include:

1. **Environment**: OS, Python version, CUDA version, GPU model
2. **Steps to reproduce**: Clear, minimal steps
3. **Expected behavior**: What should happen
4. **Actual behavior**: What actually happened
5. **Error logs**: Full traceback if applicable
6. **Screenshots**: If applicable

### Feature Requests

When requesting features, please include:

1. **Problem description**: What problem does this solve?
2. **Proposed solution**: How should it work?
3. **Alternatives considered**: Other approaches you've thought about
4. **Additional context**: Any other relevant information

## Questions?

If you have questions about contributing, feel free to:
- Open an issue with the `question` label
- Reach out to the maintainers

Thank you for contributing!
