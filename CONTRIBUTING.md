# Contributing to LangRS

Thank you for your interest in contributing to LangRS! This guide will help you get started.

---

## Development Setup

### 1. Clone the Repository

```bash
git clone https://github.com/MohanadDiab/langrs.git
cd langrs
```

### 2. Create Development Environment

```bash
conda create -n langrs-dev python=3.10
conda activate langrs-dev
pip install -r requirements.txt
pip install -e .
```

### 3. Install Development Dependencies

```bash
pip install pytest pytest-cov black isort mypy
```

---

## Code Style

### Type Hints

All code must include type hints:

```python
def my_function(param: str) -> int:
    """Function with type hints."""
    return len(param)
```

### Formatting

We use `black` for code formatting:

```bash
black langrs/ tests/
```

### Import Sorting

We use `isort` for import sorting:

```bash
isort langrs/ tests/
```

### Linting

We use `mypy` for type checking:

```bash
mypy langrs/
```

---

## Adding New Features

### Adding a New Detection Model

1. Create model class in `langrs/models/detection/`
2. Inherit from `DetectionModel` ABC
3. Implement all required methods
4. Register with `@ModelRegistry.register_detection("model_name")`
5. Write tests in `tests/test_<model_name>.py`

Example:
```python
from langrs.models.base import DetectionModel
from langrs.models.registry import ModelRegistry

@ModelRegistry.register_detection("my_model")
class MyDetectionModel(DetectionModel):
    # Implement required methods
    pass
```

**Reference Implementation**: See `langrs/models/detection/rex_omni.py` for a complete example of integrating a detection model (Rex-Omni) that:
- Downloads models from Hugging Face
- Supports multiple backends (transformers, vllm)
- Handles optional dependencies gracefully
- Converts output formats to LangRS standard

### Adding a New Segmentation Model

1. Create model class in `langrs/models/segmentation/`
2. Inherit from `SegmentationModel` ABC
3. Implement all required methods
4. Register with `@ModelRegistry.register_segmentation("model_name")`
5. Write tests

### Adding a New Outlier Detection Method

1. Create detector class in `langrs/processing/outlier_detection.py`
2. Inherit from `OutlierDetector` ABC
3. Implement `detect()` method
4. Add to builder in `langrs/core/builder.py`
5. Write tests

### Adding a New Visualization Backend

1. Create visualizer class in `langrs/visualization/`
2. Inherit from `Visualizer` ABC
3. Implement all required methods
4. Write tests

---

## Testing

### Running Tests

```bash
# All tests
pytest tests/ -v

# Without slow tests
pytest tests/ -v -k "not slow"

# Specific test file
pytest tests/test_pipeline.py -v

# With coverage
pytest tests/ --cov=langrs --cov-report=html
```

### Writing Tests

- Place tests in `tests/` directory
- Use descriptive test names
- Follow pytest conventions
- Mock external dependencies
- Aim for 100% coverage for new code

Example:
```python
def test_my_feature():
    """Test description."""
    # Arrange
    component = MyComponent()
    
    # Act
    result = component.do_something()
    
    # Assert
    assert result is not None
```

---

## Documentation

### Docstrings

Use Google-style docstrings:

```python
def my_function(param: str) -> int:
    """
    Brief description.
    
    Args:
        param: Parameter description
        
    Returns:
        Return value description
        
    Raises:
        ValueError: When param is invalid
    """
    pass
```

### Updating README

When adding features:
1. Update README.md with examples
2. Add to appropriate section
3. Include code examples
4. Update feature list

---

## Pull Request Process

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Make** your changes
4. **Write** tests for new features
5. **Ensure** all tests pass (`pytest tests/`)
6. **Format** code (`black langrs/ tests/`)
7. **Sort** imports (`isort langrs/ tests/`)
8. **Check** types (`mypy langrs/`)
9. **Commit** changes (`git commit -m 'Add amazing feature'`)
10. **Push** to branch (`git push origin feature/amazing-feature`)
11. **Open** a Pull Request

### PR Checklist

- [ ] Code follows style guidelines
- [ ] Tests added/updated
- [ ] All tests pass
- [ ] Documentation updated
- [ ] Type hints added
- [ ] No linter errors

---

## Architecture Guidelines

### SOLID Principles

- **Single Responsibility**: Each class has one purpose
- **Open/Closed**: Open for extension, closed for modification
- **Liskov Substitution**: Subtypes must be substitutable
- **Interface Segregation**: Many specific interfaces
- **Dependency Inversion**: Depend on abstractions

### Design Patterns

- **Abstract Base Classes**: For interfaces
- **Registry Pattern**: For model registration
- **Factory Pattern**: For object creation
- **Builder Pattern**: For complex object construction
- **Dependency Injection**: For testability

---

## Questions?

- Open an issue for questions
- Check existing issues
- Review the codebase
- Read the architecture docs

---

Thank you for contributing! 🎉
