# Contributing to MLX Distributed Training

## Getting Started

Thank you for considering contributing to MLX Distributed Training! This framework aims to make distributed training on Apple Silicon accessible and efficient.

### Development Setup

1. Fork and clone the repository
2. Install development dependencies:
```bash
pip install -e ".[dev]"
```

### Code Style

We follow:
- Black for Python formatting
- isort for import sorting
- MyPy for type checking

Run pre-commit checks:
```bash
./scripts/lint.sh
```

### Testing

Run the test suite:
```bash
pytest tests/ -v
```

## Contributing Guidelines

1. **Pull Requests**
   - Create feature branches from `main`
   - Include tests for new features
   - Update documentation
   - Add type hints
   - Ensure CI passes

2. **Issues**
   - Use issue templates
   - Include system information
   - Provide minimal reproduction steps

3. **Documentation**
   - Update relevant docs
   - Include docstrings
   - Add code examples

## Community

- Join our Discord server
- Check the discussions tab
- Read our Code of Conduct 