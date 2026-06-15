import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

def main():
    print("=== Day 8: K-Means Clustering ===")
    
    # 1. Generate Synthetic Clustered Dataset (unlabeled)
    # 4 distinct centers in 2D space
    X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=42)
    
    print(f"Generated data shape: {X.shape}")
    print("This is unsupervised learning, so we train our model WITHOUT using any true labels.")
    
    # 2. Train KMeans Clustering model
    k = 4
    kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=300, random_state=42)
    kmeans.fit(X)
    
    # Get predictions and centroids
    y_pred = kmeans.predict(X)
    centroids = kmeans.cluster_centers_
    inertia = kmeans.inertia_
    
    print(f"\nTraining KMeans with k={k}:")
    print(f"Sum of squared distances of samples to their closest cluster center (Inertia): {inertia:.4f}")
    print(f"Cluster centroids coordinates:\n{centroids}")
    
    # 3. The Elbow Method (Concept)
    # Running KMeans with different k values to find the "elbow"
    inertias = []
    k_range = range(1, 9)
    for i in k_range:
        km = KMeans(n_clusters=i, random_state=42)
        km.fit(X)
        inertias.append(km.inertia_)
        
    print("\nInertia values for K=1 to K=8 (used to draw the Elbow Curve):")
    for i, val in zip(k_range, inertias):
        print(f"  K={i}: Inertia={val:.2f}")
        
    # 4. Save Visualizations (Cluster Assignment and Elbow Curve)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Scanned Clusters
    ax1.scatter(X[:, 0], X[:, 1], c=y_pred, s=50, cmap='viridis', alpha=0.7, edgecolors='k', label='Data Points')
    ax1.scatter(centroids[:, 0], centroids[:, 1], c='red', s=200, marker='X', edgecolors='black', label='Centroids')
    ax1.set_title("K-Means Clustering Predictions (K=4)")
    ax1.set_xlabel("Feature 1")
    ax1.set_ylabel("Feature 2")
    ax1.legend()
    ax1.grid(True)
    
    # Plot 2: Elbow Curve
    ax2.plot(k_range, inertias, marker='o', color='purple', linestyle='-', linewidth=2)
    ax2.set_title("Elbow Method to Find Optimal K")
    ax2.set_xlabel("Number of Clusters (K)")
    ax2.set_ylabel("Inertia")
    ax2.grid(True)
    
    import os
    os.makedirs("plots", exist_ok=True)
    plot_path = os.path.join("plots", "day_08_kmeans.png")
    plt.savefig(plot_path)
    print(f"\nSaved cluster and elbow visualizations to: {plot_path}")
    plt.close()

if __name__ == "__main__":
    main()
