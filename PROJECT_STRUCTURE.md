# Project Structure

## Overview

This is a research-grade, modular codebase for the WBC-Bench-2026 Kaggle competition.

```
wbc-bench-2026/
├── README.md                 # Main project documentation
├── requirements.txt          # Python dependencies
├── setup.py                  # Package installation script
├── .gitignore               # Git ignore rules
│
├── docs/                    # Documentation
│   └── guides/             # User guides and tutorials
│       ├── START_HERE_FIRST.md
│       ├── STEP_BY_STEP_START_HERE.md
│       ├── COMPLETE_GUIDE.md
│       ├── QUICK_REFERENCE.md
│       └── ANSWERS_TO_YOUR_QUESTIONS.md
│
├── src/                     # Source code (main package)
│   ├── __init__.py
│   ├── config.py            # Configuration management
│   │
│   ├── data/                # Data handling modules
│   │   ├── __init__.py
│   │   ├── dataset.py       # Dataset class and transforms
│   │   └── preprocessing.py # Data preprocessing utilities
│   │
│   ├── models/               # Model definitions
│   │   ├── __init__.py
│   │   ├── architectures.py # Model architectures (EfficientNet, etc.)
│   │   └── losses.py        # Loss functions (Focal Loss, etc.)
│   │
│   ├── training/            # Training utilities (reserved for future)
│   │   └── __init__.py
│   │
│   └── inference/           # Inference utilities (reserved for future)
│       └── __init__.py
│
├── scripts/                 # Executable scripts
│   ├── train.py            # Main training script
│   ├── inference.py        # Inference/prediction script
│   ├── explore_data.py     # Data exploration
│   ├── visualize_samples.py # Visualization utilities
│   ├── check_setup.py      # Setup verification
│   └── train_ensemble.py   # Ensemble training helper
│
├── utils/                   # Utility modules
│   ├── __init__.py
│   └── experiment_tracker.py # Experiment tracking
│
├── tests/                   # Unit tests (reserved for future)
│
└── outputs/                 # Generated outputs (gitignored)
    ├── models/              # Trained model checkpoints
    ├── predictions/         # Prediction CSV files
    └── logs/                # Training logs
```

## Module Organization

### `src/` - Core Package
- **config.py**: Centralized configuration management
- **data/**: Data loading, preprocessing, and augmentation
- **models/**: Model architectures and loss functions
- **training/**: Training utilities (extensible)
- **inference/**: Inference utilities (extensible)

### `scripts/` - Executable Scripts
- Standalone scripts that can be run from command line
- Each script handles its own imports and path setup

### `utils/` - Utilities
- Reusable utility functions
- Experiment tracking
- Helper functions

### `docs/` - Documentation
- User guides
- Tutorials
- Reference materials

## Import Structure

### From scripts:
```python
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import Config
from src.data import WBCDataset, get_train_transforms
from src.models import get_model, get_loss_fn
```

### As installed package:
```python
from src.config import Config
from src.data import WBCDataset
from src.models import get_model
```

## Key Design Principles

1. **Modularity**: Each module has a single responsibility
2. **Extensibility**: Easy to add new models, losses, or utilities
3. **Research-Grade**: Professional structure suitable for publication
4. **Separation of Concerns**: Data, models, training, and inference are separate
5. **Configuration-Driven**: All settings in one place (`config.py`)

## Adding New Components

### New Model Architecture:
1. Add to `src/models/architectures.py` or create new file
2. Update `src/models/__init__.py` to export

### New Loss Function:
1. Add to `src/models/losses.py`
2. Update `get_loss_fn()` function

### New Data Augmentation:
1. Add to `src/data/dataset.py`
2. Update transform functions

### New Script:
1. Create in `scripts/` directory
2. Follow import pattern from existing scripts

## Testing

Run from project root:
```bash
# Training
python scripts/train.py

# Inference
python scripts/inference.py

# Data exploration
python scripts/explore_data.py
```

## Installation

```bash
pip install -e .
```

This installs the package in editable mode, allowing imports from `src/`.

