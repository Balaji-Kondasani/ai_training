import numpy as np

def gaussian_probability(x, mean=0.0, std=1.0):
    """
    Computes Gaussian Probability Density Function (PDF) value at x.
    """
    variance = std ** 2
    coefficient = 1 / np.sqrt(2 * np.pi * variance)
    exponent = np.exp(-((x - mean) ** 2) / (2 * variance))
    return coefficient * exponent

def main():
    print("=== Day 2: Probability & Statistics for Deep Learning ===")
    
    # Calculate probability density at different points of a standard normal distribution
    test_points = [-1.0, 0.0, 1.0]
    print("Standard Normal Distribution (mean=0.0, std=1.0):")
    for pt in test_points:
        pdf_val = gaussian_probability(pt)
        print(f"  Probability Density at x={pt:4.1f}: {pdf_val:.6f}")
        
    print("\nStatistical Concepts in DL:")
    print("  * Probability Distributions: Output layers of classifiers output a probability distribution (via Softmax).")
    print("  * Maximum Likelihood Estimation (MLE): Principle used to derive loss functions (like Cross-Entropy) to align model predictions with targets.")
    print("  * Eigenvalues / Eigenvectors: Principal Component Analysis (PCA) and spectral methods rely on computing eigenvectors of covariance matrices.")

if __name__ == "__main__":
    main()
