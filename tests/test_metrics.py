"""
Unit tests for fusion_metric package
"""

import unittest
import numpy as np
from fusion_metric import mse, mae, psnr, ssim, entropy, mutual_information


class TestBasicMetrics(unittest.TestCase):
    """Test cases for basic image metrics"""
    
    def setUp(self):
        """Set up test fixtures"""
        np.random.seed(42)
        self.img1 = np.random.rand(100, 100) * 255
        self.img2 = self.img1 + np.random.randn(100, 100) * 10
    
    def test_mse_identical_images(self):
        """Test MSE with identical images"""
        result = mse(self.img1, self.img1)
        self.assertAlmostEqual(result, 0.0, places=10)
    
    def test_mse_different_images(self):
        """Test MSE with different images"""
        result = mse(self.img1, self.img2)
        self.assertGreater(result, 0)
    
    def test_mse_shape_mismatch(self):
        """Test MSE raises error for mismatched shapes"""
        img_small = np.random.rand(50, 50) * 255
        with self.assertRaises(ValueError):
            mse(self.img1, img_small)
    
    def test_mae_identical_images(self):
        """Test MAE with identical images"""
        result = mae(self.img1, self.img1)
        self.assertAlmostEqual(result, 0.0, places=10)
    
    def test_mae_different_images(self):
        """Test MAE with different images"""
        result = mae(self.img1, self.img2)
        self.assertGreater(result, 0)
    
    def test_mae_shape_mismatch(self):
        """Test MAE raises error for mismatched shapes"""
        img_small = np.random.rand(50, 50) * 255
        with self.assertRaises(ValueError):
            mae(self.img1, img_small)
    
    def test_psnr_identical_images(self):
        """Test PSNR with identical images"""
        result = psnr(self.img1, self.img1)
        self.assertEqual(result, float('inf'))
    
    def test_psnr_different_images(self):
        """Test PSNR with different images"""
        result = psnr(self.img1, self.img2, data_range=255)
        self.assertGreater(result, 0)
        self.assertLess(result, 100)  # Typical PSNR range
    
    def test_psnr_shape_mismatch(self):
        """Test PSNR raises error for mismatched shapes"""
        img_small = np.random.rand(50, 50) * 255
        with self.assertRaises(ValueError):
            psnr(self.img1, img_small)
    
    def test_ssim_identical_images(self):
        """Test SSIM with identical images"""
        result = ssim(self.img1, self.img1, data_range=255, channel_axis=None)
        self.assertAlmostEqual(result, 1.0, places=5)
    
    def test_ssim_different_images(self):
        """Test SSIM with different images"""
        result = ssim(self.img1, self.img2, data_range=255, channel_axis=None)
        self.assertGreater(result, -1)
        self.assertLess(result, 1)
    
    def test_ssim_shape_mismatch(self):
        """Test SSIM raises error for mismatched shapes"""
        img_small = np.random.rand(50, 50) * 255
        with self.assertRaises(ValueError):
            ssim(self.img1, img_small)
    
    def test_ssim_multichannel(self):
        """Test SSIM with color images"""
        img1_color = np.random.rand(100, 100, 3) * 255
        img2_color = img1_color + np.random.randn(100, 100, 3) * 5
        result = ssim(img1_color, img2_color, data_range=255, channel_axis=-1)
        self.assertGreater(result, 0)
        self.assertLessEqual(result, 1)


class TestInformationTheoryMetrics(unittest.TestCase):
    """Test cases for information theory metrics"""
    
    def setUp(self):
        """Set up test fixtures"""
        np.random.seed(42)
        self.img1 = np.random.rand(100, 100) * 255
        self.img2 = np.random.rand(100, 100) * 255
    
    def test_entropy_positive(self):
        """Test that entropy is positive"""
        result = entropy(self.img1)
        self.assertGreater(result, 0)
    
    def test_entropy_uniform_image(self):
        """Test entropy for uniform image"""
        uniform_img = np.ones((100, 100)) * 128
        result = entropy(uniform_img)
        self.assertAlmostEqual(result, 0.0, places=5)
    
    def test_entropy_random_image(self):
        """Test entropy for random image"""
        result = entropy(self.img1)
        # Random image should have high entropy
        self.assertGreater(result, 5)
    
    def test_mutual_information_identical_images(self):
        """Test mutual information with identical images"""
        result = mutual_information(self.img1, self.img1)
        # MI should be equal to entropy for identical images
        entropy_val = entropy(self.img1)
        self.assertGreater(result, 0)
        self.assertLess(abs(result - entropy_val), entropy_val * 0.5)
    
    def test_mutual_information_different_images(self):
        """Test mutual information with different images"""
        result = mutual_information(self.img1, self.img2)
        self.assertGreater(result, 0)
    
    def test_mutual_information_shape_mismatch(self):
        """Test MI raises error for mismatched shapes"""
        img_small = np.random.rand(50, 50) * 255
        with self.assertRaises(ValueError):
            mutual_information(self.img1, img_small)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and special scenarios"""
    
    def test_all_zeros_images(self):
        """Test metrics with all-zero images"""
        img_zeros = np.zeros((50, 50))
        
        # MSE and MAE should be 0
        self.assertEqual(mse(img_zeros, img_zeros), 0.0)
        self.assertEqual(mae(img_zeros, img_zeros), 0.0)
        
        # PSNR should be inf
        self.assertEqual(psnr(img_zeros, img_zeros), float('inf'))
        
        # SSIM should be 1
        result_ssim = ssim(img_zeros, img_zeros, data_range=1.0, channel_axis=None)
        self.assertAlmostEqual(result_ssim, 1.0, places=5)
    
    def test_different_dtypes(self):
        """Test metrics with different data types"""
        img1_uint8 = np.random.randint(0, 256, (50, 50), dtype=np.uint8)
        img2_uint8 = np.random.randint(0, 256, (50, 50), dtype=np.uint8)
        
        # Should work with uint8
        result = mse(img1_uint8, img2_uint8)
        self.assertIsInstance(result, (float, np.floating))
    
    def test_single_pixel_difference(self):
        """Test metrics with single pixel difference"""
        img1 = np.ones((50, 50)) * 100
        img2 = img1.copy()
        img2[0, 0] = 101
        
        # MSE should be small but non-zero
        result_mse = mse(img1, img2)
        self.assertGreater(result_mse, 0)
        self.assertLess(result_mse, 1)


if __name__ == '__main__':
    unittest.main()
