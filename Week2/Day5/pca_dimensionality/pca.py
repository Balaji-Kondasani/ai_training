import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def main():
    print("=== Day 9: Principal Component Analysis (PCA) ===")
    
    # 1. Load High-Dimensional Dataset
    # Breast cancer dataset has 30 numerical features
    data = load_breast_cancer()
    X = data.data
    y = data.target
    feature_names = data.feature_names
    target_names = data.target_names
    
    print(f"Loaded dataset: Breast Cancer")
    print(f"Original shape: {X.shape} (30 features)")
    
    # 2. Preprocessing: Standardize Features
    # PCA is highly sensitive to the scale of the features because it projects data onto directions of maximum variance.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 3. Apply PCA to project to 2 dimensions
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    
    print(f"New projected shape after PCA: {X_pca.shape} (2 principal components)")
    
    # 4. Analyze Explained Variance
    # Ratio of the dataset's variance that lies along each principal component
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.sum(explained_variance)
    
    print(f"\nExplained Variance Ratio:")
    print(f"  PC1: {explained_variance[0]:.4f} ({explained_variance[0]*100:.2f}%)")
    print(f"  PC2: {explained_variance[1]:.4f} ({explained_variance[1]*100:.2f}%)")
    print(f"Total variance retained by 2 Components: {cumulative_variance:.4f} ({cumulative_variance*100:.2f}%)")
    
    # Show component loadings (weights for each feature in PC1 and PC2)
    # This helps understand what features contribute most to the principal components
    print("\nTop 3 features contributing to Principal Component 1 (absolute weights):")
    pc1_loadings = np.abs(pca.components_[0])
    top_indices = np.argsort(pc1_loadings)[::-1][:3]
    for idx in top_indices:
        print(f"  {feature_names[idx]}: loading={pca.components_[0][idx]:.4f}")
        
    # 5. Save Visualization (2D Projection of 30D Data)
    plt.figure(figsize=(10, 8))
    
    # Plot classes
    for target_idx, target_name in enumerate(target_names):
        plt.scatter(
            X_pca[y == target_idx, 0],
            X_pca[y == target_idx, 1],
            alpha=0.8,
            label=target_name,
            edgecolors='k'
        )
        
    plt.title("PCA Projection: 30 Features Reduced to 2D")
    plt.xlabel(f"Principal Component 1 ({explained_variance[0]*100:.1f}% Variance)")
    plt.ylabel(f"Principal Component 2 ({explained_variance[1]*100:.1f}% Variance)")
    plt.legend()
    plt.grid(True)
    
    import os
    os.makedirs("plots", exist_ok=True)
    plot_path = os.path.join("plots", "day_09_pca.png")
    plt.savefig(plot_path)
    print(f"\nSaved PCA scatter plot to: {plot_path}")
    plt.close()

if __name__ == "__main__":
    main()
