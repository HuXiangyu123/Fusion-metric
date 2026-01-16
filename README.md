# Fusion-metric

Self-build metric for image fusion evaluation.

## Overview

Fusion-metric is a Python library that provides basic benchmark metrics for evaluating image fusion quality. It includes commonly used metrics such as PSNR, SSIM, MSE, MAE, Entropy, and Mutual Information.

## Installation

### From source

```bash
git clone https://github.com/HuXiangyu123/Fusion-metric.git
cd Fusion-metric
pip install -r requirements.txt
pip install -e .
```

## Requirements

- Python >= 3.6
- NumPy >= 1.19.0
- scikit-image >= 0.18.0
- scipy >= 1.5.0

## Available Metrics

### Basic Metrics

- **MSE (Mean Squared Error)**: Measures the average squared difference between pixels
- **MAE (Mean Absolute Error)**: Measures the average absolute difference between pixels
- **PSNR (Peak Signal-to-Noise Ratio)**: Ratio between maximum possible power and corrupting noise power
- **SSIM (Structural Similarity Index)**: Measures structural similarity between images

### Information Theory Metrics

- **Entropy**: Measures the amount of information in an image
- **Mutual Information**: Measures the mutual dependence between two images

## Usage

### Basic Example

```python
import numpy as np
from fusion_metric import mse, mae, psnr, ssim, entropy, mutual_information

# Create sample images (or load your own)
img1 = np.random.rand(256, 256) * 255
img2 = np.random.rand(256, 256) * 255

# Calculate metrics
mse_value = mse(img1, img2)
mae_value = mae(img1, img2)
psnr_value = psnr(img1, img2)
ssim_value = ssim(img1, img2)

print(f"MSE: {mse_value:.4f}")
print(f"MAE: {mae_value:.4f}")
print(f"PSNR: {psnr_value:.4f} dB")
print(f"SSIM: {ssim_value:.4f}")

# Calculate entropy
entropy_value = entropy(img1)
print(f"Entropy: {entropy_value:.4f}")

# Calculate mutual information
mi_value = mutual_information(img1, img2)
print(f"Mutual Information: {mi_value:.4f}")
```

### With Real Images

```python
import numpy as np
from PIL import Image
from fusion_metric import psnr, ssim

# Load images
img1 = np.array(Image.open('image1.png'))
img2 = np.array(Image.open('image2.png'))

# For grayscale images
if len(img1.shape) == 2:
    psnr_value = psnr(img1, img2, data_range=255)
    ssim_value = ssim(img1, img2, data_range=255, channel_axis=None)
# For color images
else:
    psnr_value = psnr(img1, img2, data_range=255)
    ssim_value = ssim(img1, img2, data_range=255, channel_axis=-1)

print(f"PSNR: {psnr_value:.4f} dB")
print(f"SSIM: {ssim_value:.4f}")
```

## API Reference

### mse(img1, img2)

Calculate Mean Squared Error between two images.

**Parameters:**
- `img1` (numpy.ndarray): First image
- `img2` (numpy.ndarray): Second image

**Returns:**
- `float`: Mean Squared Error value

### mae(img1, img2)

Calculate Mean Absolute Error between two images.

**Parameters:**
- `img1` (numpy.ndarray): First image
- `img2` (numpy.ndarray): Second image

**Returns:**
- `float`: Mean Absolute Error value

### psnr(img1, img2, data_range=255.0)

Calculate Peak Signal-to-Noise Ratio between two images.

**Parameters:**
- `img1` (numpy.ndarray): First image
- `img2` (numpy.ndarray): Second image
- `data_range` (float): The data range of the input image (default: 255 for uint8)

**Returns:**
- `float`: PSNR value in dB

### ssim(img1, img2, data_range=255.0, channel_axis=None)

Calculate Structural Similarity Index between two images.

**Parameters:**
- `img1` (numpy.ndarray): First image
- `img2` (numpy.ndarray): Second image
- `data_range` (float): The data range of the input image (default: 255 for uint8)
- `channel_axis` (int, optional): If None, the image is assumed to be grayscale. If not None, specifies the axis of the array that represents channels (typically -1 for the last axis)

**Returns:**
- `float`: SSIM value between -1 and 1 (1 means identical)

### entropy(img)

Calculate the entropy of an image.

**Parameters:**
- `img` (numpy.ndarray): Input image

**Returns:**
- `float`: Entropy value

### mutual_information(img1, img2, bins=256)

Calculate Mutual Information between two images.

**Parameters:**
- `img1` (numpy.ndarray): First image
- `img2` (numpy.ndarray): Second image
- `bins` (int): Number of bins for histogram calculation (default: 256)

**Returns:**
- `float`: Mutual Information value

## Examples

Check the `examples/` directory for more usage examples.

## License

This project is licensed under the MIT License.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 
