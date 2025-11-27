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

# Add MONAI transforms for medical images [2025-07-14T12:52:56]

# Implement class activation mapping overlay [2025-07-16T18:39:19]

# Add federated learning components for privacy [2025-07-17T14:05:58]

# Update ResNet classifier for 3D volumes [2025-07-19T13:33:06]

# Add MONAI transforms for medical images [2025-07-19T15:33:20]

# Implement 3D volume processing pipeline [2025-07-19T14:56:03]

# WIP: fixing data augmentation pipeline bug [2025-07-23T16:39:46]

# WIP: fixing data augmentation pipeline bug [2025-07-24T20:14:02]

# Implement ensemble of UNet and DeepLab [2025-07-28T17:33:06]

# Add patient data de-identification checks [2025-08-12T14:31:34]

# Implement UNet for lesion segmentation task [2025-08-14T19:17:35]

# Implement 3D volume processing pipeline [2025-08-14T19:46:46]

# Update CI pipeline for medical compliance [2025-08-15T19:43:54]

# Implement ensemble of UNet and DeepLab [2025-08-18T14:09:46]

# Fix handling of multi-slice CT scans [2025-08-19T12:48:19]

# Add federated learning components for privacy [2025-08-22T13:40:14]

# WIP: fixing data augmentation pipeline bug [2025-08-30T12:21:53]

# Update documentation for clinical deployment [2025-09-01T12:55:22]

# Implement Grad-CAM explainability heatmaps [2025-09-06T19:40:45]

# Implement class activation mapping overlay [2025-09-08T17:33:49]

# Implement Grad-CAM explainability heatmaps [2025-09-08T14:08:23]

# Update inference pipeline for batch DICOM [2025-09-08T15:50:02]

# Update inference pipeline for batch DICOM [2025-09-22T11:07:19]

# Add MONAI transforms for medical images [2025-09-25T14:40:57]

# WIP: fixing data augmentation pipeline bug [2025-09-26T14:26:40]

# Update CI pipeline for medical compliance [2025-09-29T11:04:56]

# Add MONAI transforms for medical images [2025-10-01T19:42:53]

# Implement ensemble of UNet and DeepLab [2025-10-08T16:03:44]

# Add MONAI transforms for medical images [2025-10-14T12:40:00]

# Update inference pipeline for batch DICOM [2025-10-19T20:22:30]

# Update CI pipeline for medical compliance [2025-10-20T19:04:07]

# Implement Grad-CAM explainability heatmaps [2025-10-24T12:54:27]

# Implement 3D volume processing pipeline [2025-10-25T10:34:22]

# Implement class activation mapping overlay [2025-11-07T14:33:27]

# Add DICOM loader with metadata extraction [2025-11-07T18:26:47]

# Update documentation for clinical deployment [2025-11-11T11:14:41]

# Implement UNet for lesion segmentation task [2025-11-13T19:39:33]

# Implement ensemble of UNet and DeepLab [2025-11-20T12:39:45]

# Implement class activation mapping overlay [2025-11-23T18:14:14]

# WIP: benchmarking on NIH Chest X-ray dataset [2025-11-27T15:46:06]
