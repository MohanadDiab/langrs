# Running Examples

## Installation

Before running the examples, you need to install the LangRS package in development mode:

```bash
# From the repository root directory
pip install -e .
```

Or if you want to install it normally:

```bash
pip install .
```

## Running Examples

After installation, you can run any example from the repository root:

```bash
python examples/basic_usage.py
python examples/advanced_usage.py
python examples/step_by_step.py
# etc.
```

## Troubleshooting

### ModuleNotFoundError: No module named 'langrs'

This error means the package is not installed. Run:
```bash
pip install -e .
```

### Import Errors

Make sure you're running the examples from the repository root directory, not from within the `examples/` folder.

### Missing Dependencies

If you get import errors for specific models (e.g., GroundingDINO, Rex-Omni), make sure all dependencies are installed:

```bash
pip install -r requirements.txt
```

For Rex-Omni specifically, you may need additional dependencies:
```bash
pip install transformers qwen-vl-utils accelerate huggingface_hub
```
