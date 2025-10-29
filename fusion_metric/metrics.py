"""
Core image metrics for fusion evaluation
"""

import numpy as np
from skimage.metrics import structural_similarity as compare_ssim
from scipy.stats import entropy as scipy_entropy


def mse(img1, img2):
    """
    Calculate Mean Squared Error between two images.
    
    Parameters:
    -----------
    img1 : numpy.ndarray
        First image
    img2 : numpy.ndarray
        Second image
        
    Returns:
    --------
    float
        Mean Squared Error
    """
    img1 = np.asarray(img1, dtype=np.float64)
    img2 = np.asarray(img2, dtype=np.float64)
    
    if img1.shape != img2.shape:
        raise ValueError("Images must have the same shape")
    
    return np.mean((img1 - img2) ** 2)


def mae(img1, img2):
    """
    Calculate Mean Absolute Error between two images.
    
    Parameters:
    -----------
    img1 : numpy.ndarray
        First image
    img2 : numpy.ndarray
        Second image
        
    Returns:
    --------
    float
        Mean Absolute Error
    """
    img1 = np.asarray(img1, dtype=np.float64)
    img2 = np.asarray(img2, dtype=np.float64)
    
    if img1.shape != img2.shape:
        raise ValueError("Images must have the same shape")
    
    return np.mean(np.abs(img1 - img2))


def psnr(img1, img2, data_range=255.0):
    """
    Calculate Peak Signal-to-Noise Ratio between two images.
    
    Parameters:
    -----------
    img1 : numpy.ndarray
        First image
    img2 : numpy.ndarray
        Second image
    data_range : float
        The data range of the input image (default: 255 for uint8)
        
    Returns:
    --------
    float
        PSNR value in dB
    """
    img1 = np.asarray(img1, dtype=np.float64)
    img2 = np.asarray(img2, dtype=np.float64)
    
    if img1.shape != img2.shape:
        raise ValueError("Images must have the same shape")
    
    mse_val = mse(img1, img2)
    
    if mse_val == 0:
        return float('inf')
    
    return 10 * np.log10((data_range ** 2) / mse_val)


def ssim(img1, img2, data_range=255.0, multichannel=None):
    """
    Calculate Structural Similarity Index between two images.
    
    Parameters:
    -----------
    img1 : numpy.ndarray
        First image
    img2 : numpy.ndarray
        Second image
    data_range : float
        The data range of the input image (default: 255 for uint8)
    multichannel : bool, optional
        Whether to treat the last dimension as channels
        
    Returns:
    --------
    float
        SSIM value between -1 and 1 (1 means identical)
    """
    img1 = np.asarray(img1, dtype=np.float64)
    img2 = np.asarray(img2, dtype=np.float64)
    
    if img1.shape != img2.shape:
        raise ValueError("Images must have the same shape")
    
    # Determine if image is multichannel
    if multichannel is None:
        multichannel = img1.ndim == 3 and img1.shape[-1] in [3, 4]
    
    return compare_ssim(img1, img2, data_range=data_range, channel_axis=-1 if multichannel else None)


def entropy(img):
    """
    Calculate the entropy of an image.
    
    Parameters:
    -----------
    img : numpy.ndarray
        Input image
        
    Returns:
    --------
    float
        Entropy value
    """
    img = np.asarray(img)
    
    # Flatten the image and calculate histogram
    hist, _ = np.histogram(img.flatten(), bins=256, range=(0, 256))
    
    # Normalize histogram to get probability distribution
    prob = hist / hist.sum()
    
    # Remove zero probabilities
    prob = prob[prob > 0]
    
    # Calculate entropy
    return -np.sum(prob * np.log2(prob))


def mutual_information(img1, img2, bins=256):
    """
    Calculate Mutual Information between two images.
    
    Parameters:
    -----------
    img1 : numpy.ndarray
        First image
    img2 : numpy.ndarray
        Second image
    bins : int
        Number of bins for histogram calculation
        
    Returns:
    --------
    float
        Mutual Information value
    """
    img1 = np.asarray(img1).flatten()
    img2 = np.asarray(img2).flatten()
    
    if img1.shape != img2.shape:
        raise ValueError("Images must have the same shape")
    
    # Calculate 2D histogram
    hist_2d, _, _ = np.histogram2d(img1, img2, bins=bins, range=[[0, 256], [0, 256]])
    
    # Calculate marginal distributions
    p_x = np.sum(hist_2d, axis=1)
    p_y = np.sum(hist_2d, axis=0)
    
    # Normalize to get probabilities
    p_xy = hist_2d / np.sum(hist_2d)
    p_x = p_x / np.sum(p_x)
    p_y = p_y / np.sum(p_y)
    
    # Calculate mutual information
    mi = 0.0
    for i in range(bins):
        for j in range(bins):
            if p_xy[i, j] > 0:
                mi += p_xy[i, j] * np.log2(p_xy[i, j] / (p_x[i] * p_y[j] + 1e-10))
    
    return mi
