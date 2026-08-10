# Installation

## Requirements

- Python 3.10 or later
- NumPy 1.23 or later
- SciPy 1.10 or later

Matplotlib, pandas, and openpyxl are optional and are used only by plotting
and data examples.

## Install from source

```bash
python -m pip install .
```

For all examples and development checks:

```bash
python -m pip install -e ".[dev]"
python -m pytest
```

## Build distribution archives

```bash
python -m build
```

This creates a source archive and a platform-independent wheel in `dist/`.
