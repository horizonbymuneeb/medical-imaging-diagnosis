"""Utility functions for production ML."""
import numpy as np
import torch
import random
import json
from pathlib import Path
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info(f"Random seed set to {seed}")

def save_metrics(metrics: Dict[str, float], path: str) -> None:
    """Save evaluation metrics to JSON."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved to {path}")

def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    import yaml
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def get_device() -> torch.device:
    """Get the best available device."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def format_number(n: int) -> str:
    """Format large numbers with K/M/B suffixes."""
    for unit in ['', 'K', 'M', 'B']:
        if abs(n) < 1000:
            return f"{n:.1f}{unit}"
        n /= 1000
    return f"{n:.1f}T"

# Add DICOM loader with metadata extraction [2025-06-12T13:17:06]

# Update inference pipeline for batch DICOM [2025-06-19T15:44:18]

# Implement Grad-CAM explainability heatmaps [2025-06-29T14:45:30]

# Add DICOM loader with metadata extraction [2025-07-01T12:45:42]

# WIP: tuning loss for imbalanced tumor labels [2025-07-03T20:21:39]

# WIP: tuning loss for imbalanced tumor labels [2025-07-14T18:46:26]
