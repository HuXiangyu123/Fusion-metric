"""
Fusion-metric: Basic benchmark metrics for image fusion evaluation
"""

from .metrics import (
    mse,
    mae,
    psnr,
    ssim,
    entropy,
    mutual_information,
)

__version__ = "0.1.0"
__all__ = [
    "mse",
    "mae",
    "psnr",
    "ssim",
    "entropy",
    "mutual_information",
]
