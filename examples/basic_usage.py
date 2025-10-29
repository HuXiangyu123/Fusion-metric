"""
Example script demonstrating the usage of fusion_metric package
"""

import numpy as np
from fusion_metric import mse, mae, psnr, ssim, entropy, mutual_information


def main():
    print("=" * 60)
    print("Fusion-metric Example: Basic Image Metrics")
    print("=" * 60)
    
    # Create two sample grayscale images
    print("\n1. Creating sample images...")
    np.random.seed(42)
    img1 = np.random.rand(256, 256) * 255
    img2 = img1 + np.random.randn(256, 256) * 10  # Add some noise
    
    print(f"   Image 1 shape: {img1.shape}")
    print(f"   Image 2 shape: {img2.shape}")
    
    # Calculate basic metrics
    print("\n2. Calculating basic metrics...")
    mse_value = mse(img1, img2)
    mae_value = mae(img1, img2)
    psnr_value = psnr(img1, img2, data_range=255)
    ssim_value = ssim(img1, img2, data_range=255, multichannel=False)
    
    print(f"   MSE: {mse_value:.4f}")
    print(f"   MAE: {mae_value:.4f}")
    print(f"   PSNR: {psnr_value:.4f} dB")
    print(f"   SSIM: {ssim_value:.4f}")
    
    # Calculate information theory metrics
    print("\n3. Calculating information theory metrics...")
    entropy1 = entropy(img1)
    entropy2 = entropy(img2)
    mi_value = mutual_information(img1, img2)
    
    print(f"   Entropy (Image 1): {entropy1:.4f}")
    print(f"   Entropy (Image 2): {entropy2:.4f}")
    print(f"   Mutual Information: {mi_value:.4f}")
    
    # Test with identical images
    print("\n4. Testing with identical images...")
    mse_identical = mse(img1, img1)
    psnr_identical = psnr(img1, img1)
    ssim_identical = ssim(img1, img1, data_range=255, multichannel=False)
    
    print(f"   MSE (identical): {mse_identical:.4f}")
    print(f"   PSNR (identical): {psnr_identical} dB")
    print(f"   SSIM (identical): {ssim_identical:.4f}")
    
    # Test with color images
    print("\n5. Testing with color images...")
    img1_color = np.random.rand(128, 128, 3) * 255
    img2_color = img1_color + np.random.randn(128, 128, 3) * 5
    
    psnr_color = psnr(img1_color, img2_color, data_range=255)
    ssim_color = ssim(img1_color, img2_color, data_range=255, multichannel=True)
    
    print(f"   Color image shape: {img1_color.shape}")
    print(f"   PSNR (color): {psnr_color:.4f} dB")
    print(f"   SSIM (color): {ssim_color:.4f}")
    
    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
