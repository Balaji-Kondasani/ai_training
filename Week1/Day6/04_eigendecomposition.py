"""
=============================================================================
PART 4: EIGENDECOMPOSITION (Core of PCA)
=============================================================================

Eigendecomposition is the foundation of:
  - PCA (Principal Component Analysis) -- dimensionality reduction
  - Spectral clustering
  - Google's PageRank algorithm
  - Understanding how linear transformations work

Topics covered:
  4.1 Eigenvalues and Eigenvectors
  4.2 Eigendecomposition in Practice: PCA from Scratch
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for saving figures
import matplotlib.pyplot as plt

print("=" * 70)
print("PART 4: EIGENDECOMPOSITION (Core of PCA)")
print("=" * 70)

# ──────────────────────────────────────────────────────────────────────
# 4.1 EIGENVALUES AND EIGENVECTORS
# ──────────────────────────────────────────────────────────────────────
print("\n" + "─" * 70)
print("4.1 EIGENVALUES AND EIGENVECTORS")
print("─" * 70)

# The eigenvalue equation: A @ v = λ * v
#
# In plain English:
# When you multiply matrix A by eigenvector v, the result is just v
# scaled by a number λ (the eigenvalue). The direction doesn't change!
#
# Think of it like this: most vectors change direction when multiplied
# by a matrix. Eigenvectors are special -- they only get stretched or
# compressed, never rotated.

A = np.array([[4, 2],
              [1, 3]])

print(f"Matrix A:\n{A}\n")

# Compute eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(A)

print(f"Eigenvalues:  {eigenvalues}")
print(f"Eigenvectors (as columns):\n{eigenvectors}\n")

# Verify the eigenvalue equation: A @ v = λ * v
for i in range(len(eigenvalues)):
    v = eigenvectors[:, i]  # eigenvectors are stored as COLUMNS
    lam = eigenvalues[i]
    left_side = A @ v
    right_side = lam * v
    print(f"Eigenpair {i + 1}:")
    print(f"  λ = {lam:.4f}")
    print(f"  v = {v.round(4)}")
    print(f"  A @ v     = {left_side.round(4)}")
    print(f"  λ * v     = {right_side.round(4)}")
    print(f"  Match: {np.allclose(left_side, right_side)}")

# Properties of eigenvalues (connecting to trace and determinant)
print(f"\n--- Properties ---")
print(f"Sum of eigenvalues = trace:       {eigenvalues.sum():.4f} == {np.trace(A)}")
print(f"Product of eigenvalues = det:     {eigenvalues.prod():.4f} == {np.linalg.det(A):.4f}")

# Eigendecomposition: A = P @ D @ P⁻¹
# where P = matrix of eigenvectors, D = diagonal matrix of eigenvalues
P = eigenvectors
D = np.diag(eigenvalues)
P_inv = np.linalg.inv(P)

A_reconstructed = P @ D @ P_inv
print(f"\nEigendecomposition: A = P @ D @ P⁻¹")
print(f"Reconstructed A:\n{A_reconstructed.round(4)}")
print(f"Matches original: {np.allclose(A, A_reconstructed)}")

# ─── Symmetric Matrices Have Special Properties ─────────────────────

print(f"\n--- Symmetric Matrices (Important for ML!) ---")

# Covariance matrices are always symmetric.
# Symmetric matrices have:
# 1. All REAL eigenvalues (no imaginary parts)
# 2. Orthogonal eigenvectors (perpendicular to each other)

S = np.array([[5, 2],
              [2, 3]])  # symmetric: S = S^T

print(f"Symmetric matrix S:\n{S}")
print(f"S == S^T: {np.allclose(S, S.T)}")

# Use eigh (not eig) for symmetric matrices -- faster and more accurate
eigenvalues_s, eigenvectors_s = np.linalg.eigh(S)

print(f"\nEigenvalues: {eigenvalues_s}")
print(f"Eigenvectors:\n{eigenvectors_s}")

# Verify orthogonality: v1 · v2 = 0
v1 = eigenvectors_s[:, 0]
v2 = eigenvectors_s[:, 1]
print(f"\nOrthogonality check: v1 · v2 = {np.dot(v1, v2):.10f} (≈ 0)")
print(f"Both unit vectors: ||v1|| = {np.linalg.norm(v1):.4f}, ||v2|| = {np.linalg.norm(v2):.4f}")

# ──────────────────────────────────────────────────────────────────────
# 4.2 EIGENDECOMPOSITION IN PRACTICE: PCA FROM SCRATCH
# ──────────────────────────────────────────────────────────────────────
print("\n" + "─" * 70)
print("4.2 PCA FROM SCRATCH (using eigendecomposition)")
print("─" * 70)

# PCA (Principal Component Analysis) is THE most common dimensionality
# reduction technique. It finds the directions (principal components)
# along which data varies the most.
#
# Algorithm:
#   1. Center the data (subtract mean)
#   2. Compute covariance matrix
#   3. Eigendecomposition of covariance matrix
#   4. Sort eigenvectors by eigenvalue (largest first)
#   5. Project data onto top k eigenvectors

from sklearn.datasets import load_iris

# Load the Iris dataset: 150 flowers, 4 measurements each
iris = load_iris()
X = iris.data       # (150, 4)
y = iris.target     # species labels (0, 1, 2)
feature_names = iris.feature_names

print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
print(f"Features: {feature_names}")
print(f"First 3 samples:\n{X[:3]}\n")

# ─── Step 1: Center the data ────────────────────────────────────────
mean = X.mean(axis=0)
X_centered = X - mean  # broadcasting!

print("Step 1: Center the data (subtract mean of each feature)")
print(f"  Feature means: {mean.round(4)}")
print(f"  After centering, means: {X_centered.mean(axis=0).round(10)}")

# ─── Step 2: Compute the covariance matrix ──────────────────────────
# Cov = (1/(n-1)) * X_centered^T @ X_centered
n = X_centered.shape[0]
cov_matrix = (X_centered.T @ X_centered) / (n - 1)

# Equivalent to:
cov_matrix_np = np.cov(X_centered, rowvar=False)

print(f"\nStep 2: Covariance matrix ({cov_matrix.shape[0]}×{cov_matrix.shape[1]})")
print(f"  Cov matrix:\n{cov_matrix.round(4)}")
print(f"  Is symmetric: {np.allclose(cov_matrix, cov_matrix.T)}")
print(f"  Matches np.cov: {np.allclose(cov_matrix, cov_matrix_np)}")

# ─── Step 3: Eigendecomposition ─────────────────────────────────────
# Using eigh because covariance matrix is symmetric
eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

print(f"\nStep 3: Eigendecomposition")
print(f"  Eigenvalues (unsorted): {eigenvalues.round(4)}")

# ─── Step 4: Sort by eigenvalue (largest first) ─────────────────────
# eigh returns eigenvalues in ascending order; we want descending
idx = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

print(f"\nStep 4: Sort eigenvalues (largest first)")
print(f"  Sorted eigenvalues: {eigenvalues.round(4)}")

# Explained variance ratio: how much of the total variance does
# each component capture?
explained_variance_ratio = eigenvalues / eigenvalues.sum()
cumulative_variance = np.cumsum(explained_variance_ratio)

print(f"\n  Explained variance ratio: {explained_variance_ratio.round(4)}")
print(f"  Cumulative variance:     {cumulative_variance.round(4)}")
print(f"\n  PC1 explains {explained_variance_ratio[0]:.1%} of variance")
print(f"  PC1+PC2 explain {cumulative_variance[1]:.1%} of variance")
print(f"  → We can reduce 4 features to 2 and keep {cumulative_variance[1]:.1%} of information!")

# ─── Step 5: Project data onto top k components ─────────────────────
n_components = 2
W = eigenvectors[:, :n_components]  # (4, 2) projection matrix
X_pca = X_centered @ W             # (150, 4) @ (4, 2) = (150, 2)

print(f"\nStep 5: Project onto top {n_components} components")
print(f"  Projection matrix W shape: {W.shape}")
print(f"  Original data shape:  {X.shape}")
print(f"  Reduced data shape:   {X_pca.shape}")
print(f"  First 3 transformed samples:\n{X_pca[:3].round(4)}")

# ─── Verify against sklearn's PCA ───────────────────────────────────
from sklearn.decomposition import PCA

pca_sklearn = PCA(n_components=2)
X_pca_sklearn = pca_sklearn.fit_transform(X)

# Note: signs of components may be flipped (both are valid)
print(f"\n--- Verification against sklearn PCA ---")
print(f"Our explained variance ratio:     {explained_variance_ratio[:2].round(6)}")
print(f"sklearn explained variance ratio:  {pca_sklearn.explained_variance_ratio_.round(6)}")
print(f"Ratios match: {np.allclose(explained_variance_ratio[:2], pca_sklearn.explained_variance_ratio_)}")

# ─── Visualize PCA Results ───────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Our PCA
species_names = iris.target_names
colors = ['#e41a1c', '#377eb8', '#4daf4a']
for i, name in enumerate(species_names):
    mask = y == i
    axes[0].scatter(X_pca[mask, 0], X_pca[mask, 1],
                    c=colors[i], label=name, alpha=0.7, edgecolors='k', linewidth=0.5)
axes[0].set_xlabel(f'PC1 ({explained_variance_ratio[0]:.1%} variance)')
axes[0].set_ylabel(f'PC2 ({explained_variance_ratio[1]:.1%} variance)')
axes[0].set_title('Our PCA from Scratch')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# sklearn PCA
for i, name in enumerate(species_names):
    mask = y == i
    axes[1].scatter(X_pca_sklearn[mask, 0], X_pca_sklearn[mask, 1],
                    c=colors[i], label=name, alpha=0.7, edgecolors='k', linewidth=0.5)
axes[1].set_xlabel(f'PC1 ({pca_sklearn.explained_variance_ratio_[0]:.1%} variance)')
axes[1].set_ylabel(f'PC2 ({pca_sklearn.explained_variance_ratio_[1]:.1%} variance)')
axes[1].set_title('sklearn PCA (verification)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('linear_algebra_for_ml/04_pca_visualization.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"\nPCA visualization saved to: 04_pca_visualization.png")

# ─── Scree Plot (Explained Variance) ────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
components = range(1, len(eigenvalues) + 1)
ax.bar(components, explained_variance_ratio, alpha=0.7, label='Individual')
ax.plot(components, cumulative_variance, 'ro-', label='Cumulative')
ax.axhline(y=0.95, color='gray', linestyle='--', label='95% threshold')
ax.set_xlabel('Principal Component')
ax.set_ylabel('Explained Variance Ratio')
ax.set_title('Scree Plot -- How Many Components Do We Need?')
ax.set_xticks(list(components))
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('linear_algebra_for_ml/04_scree_plot.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Scree plot saved to: 04_scree_plot.png")

print("\n" + "=" * 70)
print("PART 4 COMPLETE -- You now understand eigendecomposition and PCA!")
print("Key takeaways:")
print("  - Eigenvectors = directions that don't rotate under transformation")
print("  - Eigenvalues = how much stretching in that direction")
print("  - PCA = eigendecomposition of the covariance matrix")
print("  - PCA steps: center → covariance → eigen → sort → project")
print("  - Sum of eigenvalues = trace, Product = determinant")
print("=" * 70)
