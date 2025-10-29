"""
Example demonstrating evaluation of image fusion results using fusion_metric
"""

import numpy as np
from fusion_metric import mse, mae, psnr, ssim, entropy, mutual_information


def evaluate_fusion_quality(source1, source2, fused):
    """
    Evaluate the quality of a fused image compared to its source images.
    
    Parameters:
    -----------
    source1 : numpy.ndarray
        First source image
    source2 : numpy.ndarray
        Second source image
    fused : numpy.ndarray
        Fused image
        
    Returns:
    --------
    dict
        Dictionary containing all evaluation metrics
    """
    results = {
        'source1_vs_fused': {},
        'source2_vs_fused': {},
        'source_mutual_info': {},
        'fused_properties': {}
    }
    
    # Metrics between source1 and fused image
    results['source1_vs_fused']['mse'] = mse(source1, fused)
    results['source1_vs_fused']['mae'] = mae(source1, fused)
    results['source1_vs_fused']['psnr'] = psnr(source1, fused)
    results['source1_vs_fused']['ssim'] = ssim(source1, fused)
    
    # Metrics between source2 and fused image
    results['source2_vs_fused']['mse'] = mse(source2, fused)
    results['source2_vs_fused']['mae'] = mae(source2, fused)
    results['source2_vs_fused']['psnr'] = psnr(source2, fused)
    results['source2_vs_fused']['ssim'] = ssim(source2, fused)
    
    # Mutual information between sources and fused image
    results['source_mutual_info']['mi_source1_fused'] = mutual_information(source1, fused)
    results['source_mutual_info']['mi_source2_fused'] = mutual_information(source2, fused)
    results['source_mutual_info']['mi_source1_source2'] = mutual_information(source1, source2)
    
    # Properties of fused image
    results['fused_properties']['entropy'] = entropy(fused)
    results['fused_properties']['source1_entropy'] = entropy(source1)
    results['fused_properties']['source2_entropy'] = entropy(source2)
    
    return results


def print_evaluation_results(results):
    """Print evaluation results in a formatted way"""
    print("\n" + "=" * 70)
    print("IMAGE FUSION QUALITY EVALUATION")
    print("=" * 70)
    
    print("\n1. Source 1 vs Fused Image:")
    print(f"   MSE:  {results['source1_vs_fused']['mse']:.4f}")
    print(f"   MAE:  {results['source1_vs_fused']['mae']:.4f}")
    print(f"   PSNR: {results['source1_vs_fused']['psnr']:.4f} dB")
    print(f"   SSIM: {results['source1_vs_fused']['ssim']:.4f}")
    
    print("\n2. Source 2 vs Fused Image:")
    print(f"   MSE:  {results['source2_vs_fused']['mse']:.4f}")
    print(f"   MAE:  {results['source2_vs_fused']['mae']:.4f}")
    print(f"   PSNR: {results['source2_vs_fused']['psnr']:.4f} dB")
    print(f"   SSIM: {results['source2_vs_fused']['ssim']:.4f}")
    
    print("\n3. Mutual Information Analysis:")
    print(f"   MI (Source 1 ↔ Fused):  {results['source_mutual_info']['mi_source1_fused']:.4f}")
    print(f"   MI (Source 2 ↔ Fused):  {results['source_mutual_info']['mi_source2_fused']:.4f}")
    print(f"   MI (Source 1 ↔ Source 2): {results['source_mutual_info']['mi_source1_source2']:.4f}")
    
    print("\n4. Entropy Analysis:")
    print(f"   Source 1 Entropy: {results['fused_properties']['source1_entropy']:.4f}")
    print(f"   Source 2 Entropy: {results['fused_properties']['source2_entropy']:.4f}")
    print(f"   Fused Entropy:    {results['fused_properties']['entropy']:.4f}")
    
    print("\n" + "=" * 70)


def main():
    """Main function demonstrating fusion evaluation"""
    print("Simulating image fusion scenario...")
    
    # Create simulated source images
    # In practice, these would be loaded from disk
    np.random.seed(42)
    
    # Simulate a focused image (source 1) and an out-of-focus image (source 2)
    source1 = np.random.rand(256, 256) * 200 + 55  # High contrast
    source2 = np.random.rand(256, 256) * 150 + 50  # Lower contrast
    
    # Simulate a fused image (weighted average)
    fused = 0.6 * source1 + 0.4 * source2
    
    print(f"Source 1 shape: {source1.shape}")
    print(f"Source 2 shape: {source2.shape}")
    print(f"Fused image shape: {fused.shape}")
    
    # Evaluate fusion quality
    results = evaluate_fusion_quality(source1, source2, fused)
    
    # Print results
    print_evaluation_results(results)
    
    # Interpretation
    print("\nInterpretation:")
    print("- Higher PSNR and SSIM values indicate better similarity")
    print("- Higher mutual information indicates more shared information")
    print("- Fused image entropy should capture information from both sources")
    print("\nNote: These are simulated results. Real fusion evaluation depends on")
    print("      the actual images and fusion algorithm used.")


if __name__ == "__main__":
    main()
