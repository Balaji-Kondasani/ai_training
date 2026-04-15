"""
=============================================================================
PART 5: SINGULAR VALUE DECOMPOSITION (SVD)
=============================================================================

SVD is the "Swiss army knife" of linear algebra. It works on ANY matrix
(not just square ones). It's the foundation of:
  - PCA (another way to compute it)
  - Image compression
  - Latent Semantic Analysis (LSA) for NLP
  - Recommender systems (Netflix Prize!)
  - Noise reduction

Topics covered:
  5.1 SVD Theory and Implementation
  5.2 SVD Application: Image Compression
  5.3 SVD for Dimensionality Reduction (Truncated SVD / LSA)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

print("=" * 70)
print("PART 5: SINGULAR VALUE DECOMPOSITION (SVD)")
print("=" * 70)

# ──────────────────────────────────────────────────────────────────────
# 5.1 SVD THEORY AND IMPLEMENTATION
# ──────────────────────────────────────────────────────────────────────
print("\n" + "─" * 70)
print("5.1 SVD THEORY AND IMPLEMENTATION")
print("─" * 70)

# Any matrix A (m×n) can be decomposed as: A = U @ Σ @ V^T
#
# U  (m×m): Left singular vectors  -- orthogonal matrix
# Σ  (m×n): Singular values        -- diagonal (non-negative, sorted)
# V^T(n×n): Right singular vectors -- orthogonal matrix
#
# The singular values σ₁ ≥ σ₂ ≥ ... ≥ 0 tell you how "important"
# each component is (like eigenvalues, but always non-negative).

A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9],
              [10, 11, 12]])  # 4×3 matrix (not square!)

print(f"Matrix A ({A.shape[0]}×{A.shape[1]}):\n{A}\n")

# Compute SVD
U, s, Vt = np.linalg.svd(A)

print(f"U shape:  {U.shape}")   # (4, 4) -- left singular vectors
print(f"s shape:  {s.shape}")   # (3,)   -- singular values (as 1D array)
print(f"Vt shape: {Vt.shape}")  # (3, 3) -- right singular vectors (transposed)

print(f"\nSingular values: {s.round(4)}")
print(f"U (left singular vectors):\n{U.round(4)}\n")
print(f"Vt (right singular vectors):\n{Vt.round(4)}")

# Reconstruct the original matrix: A = U @ Σ @ V^T
# NumPy returns s as a 1D array, so we need to build the Σ matrix
Sigma = np.zeros_like(A, dtype=float)
np.fill_diagonal(Sigma, s)

A_reconstructed = U @ Sigma @ Vt
print(f"\nReconstructed A:\n{A_reconstructed.round(4)}")
print(f"Matches original: {np.allclose(A, A_reconstructed)}")

# Properties of SVD
print(f"\n--- SVD Properties ---")
print(f"U is orthogonal: U^T @ U = I?  {np.allclose(U.T @ U, np.eye(U.shape[1]))}")
print(f"V is orthogonal: Vt @ Vt^T = I? {np.allclose(Vt @ Vt.T, np.eye(Vt.shape[0]))}")

# Relationship to eigenvalues:
# Singular values of A = sqrt(eigenvalues of A^T @ A)
ATA = A.T @ A
eigvals_ATA = np.linalg.eigvalsh(ATA)
singular_from_eigen = np.sqrt(np.sort(eigvals_ATA)[::-1])
print(f"\nSingular values:          {s.round(4)}")
print(f"sqrt(eigenvalues of ATA): {singular_from_eigen.round(4)}")
print(f"Match: {np.allclose(s, singular_from_eigen)}")

# Compact (economy) SVD: only compute the "useful" parts
U_compact, s_compact, Vt_compact = np.linalg.svd(A, full_matrices=False)
print(f"\n--- Compact SVD ---")
print(f"Full: U={U.shape}, s={s.shape}, Vt={Vt.shape}")
print(f"Compact: U={U_compact.shape}, s={s_compact.shape}, Vt={Vt_compact.shape}")

# ──────────────────────────────────────────────────────────────────────
# 5.2 SVD APPLICATION: IMAGE COMPRESSION
# ──────────────────────────────────────────────────────────────────────
print("\n" + "─" * 70)
print("5.2 SVD APPLICATION: IMAGE COMPRESSION")
print("─" * 70)

# Create a synthetic grayscale "image" with clear structure
# (In practice you'd load a real image with PIL or matplotlib)
np.random.seed(42)
size = 100

# Create an image with horizontal and vertical gradients + some noise
x = np.linspace(0, 1, size)
y = np.linspace(0, 1, size)
X_grid, Y_grid = np.meshgrid(x, y)
image = (np.sin(2 * np.pi * X_grid) * np.cos(3 * np.pi * Y_grid)
         + 0.3 * np.random.randn(size, size))

print(f"Image shape: {image.shape}")
print(f"Total elements: {image.size}")

# Compute SVD of the image
U, s, Vt = np.linalg.svd(image)

def compress_image(U, s, Vt, k):
    """Reconstruct image using only the top k singular values."""
    return U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]

def storage_ratio(original_shape, k):
    """How much storage does rank-k approximation need vs original?"""
    m, n = original_shape
    original = m * n
    compressed = k * (m + n + 1)  # U columns + Vt rows + singular values
    return compressed / original

# Compress at different ranks
ranks = [1, 5, 10, 20, 50]
print(f"\n--- Compression Results ---")
print(f"{'Rank k':<10} {'Storage %':<15} {'Error (Frobenius)':<20}")
print("-" * 45)

for k in ranks:
    compressed = compress_image(U, s, Vt, k)
    error = np.linalg.norm(image - compressed, 'fro') / np.linalg.norm(image, 'fro')
    ratio = storage_ratio(image.shape, k)
    print(f"k={k:<7} {ratio:>8.1%}         {error:.4f}")

# Visualize compression at different ranks
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

# Original
axes[0].imshow(image, cmap='viridis')
axes[0].set_title(f'Original ({size}×{size} = {size**2} values)')
axes[0].axis('off')

# Compressed versions
for idx, k in enumerate([1, 5, 10, 20, 50]):
    compressed = compress_image(U, s, Vt, k)
    ratio = storage_ratio(image.shape, k)
    axes[idx + 1].imshow(compressed, cmap='viridis')
    axes[idx + 1].set_title(f'Rank {k} ({ratio:.0%} storage)')
    axes[idx + 1].axis('off')

plt.suptitle('Image Compression using SVD', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('linear_algebra_for_ml/05_svd_image_compression.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"\nImage compression visualization saved to: 05_svd_image_compression.png")

# Singular value decay plot
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(range(1, len(s) + 1), s, 'b.-')
ax.set_xlabel('Singular Value Index')
ax.set_ylabel('Singular Value Magnitude')
ax.set_title('Singular Value Decay -- How Many Components Matter?')
ax.grid(True, alpha=0.3)

# Cumulative energy
energy = np.cumsum(s ** 2) / np.sum(s ** 2)
ax2 = ax.twinx()
ax2.plot(range(1, len(s) + 1), energy, 'r.-', alpha=0.7)
ax2.set_ylabel('Cumulative Energy (red)', color='red')
ax2.axhline(y=0.95, color='gray', linestyle='--', alpha=0.5)
ax2.tick_params(axis='y', labelcolor='red')

plt.tight_layout()
plt.savefig('linear_algebra_for_ml/05_singular_value_decay.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Singular value decay plot saved to: 05_singular_value_decay.png")

# ──────────────────────────────────────────────────────────────────────
# 5.3 SVD FOR DIMENSIONALITY REDUCTION (Truncated SVD / LSA)
# ──────────────────────────────────────────────────────────────────────
print("\n" + "─" * 70)
print("5.3 SVD FOR NLP: LATENT SEMANTIC ANALYSIS (LSA)")
print("─" * 70)

# LSA uses SVD on a term-document matrix (like TF-IDF) to discover
# hidden "topics" in a collection of documents.

from sklearn.feature_extraction.text import TfidfVectorizer

# Sample documents about different topics
documents = [
    "Machine learning uses algorithms to learn from data",
    "Deep learning is a subset of machine learning using neural networks",
    "Natural language processing handles text and speech data",
    "NLP uses tokenization and word embeddings for text analysis",
    "Python is popular for data science and machine learning",
    "TensorFlow and PyTorch are deep learning frameworks",
    "Football is a popular sport played worldwide",
    "The World Cup is the biggest football tournament",
    "Basketball players score points by shooting the ball",
    "NBA is the top professional basketball league",
]

print("Documents:")
for i, doc in enumerate(documents):
    print(f"  [{i}] {doc}")

# Create TF-IDF matrix
tfidf = TfidfVectorizer(stop_words='english')
X_tfidf = tfidf.fit_transform(documents).toarray()

feature_names = tfidf.get_feature_names_out()
print(f"\nTF-IDF matrix shape: {X_tfidf.shape}")
print(f"  {X_tfidf.shape[0]} documents × {X_tfidf.shape[1]} unique terms")
print(f"  Terms: {list(feature_names)}")

# Apply SVD (Truncated)
U, s, Vt = np.linalg.svd(X_tfidf, full_matrices=False)

print(f"\nSVD shapes: U={U.shape}, s={s.shape}, Vt={Vt.shape}")
print(f"Singular values: {s.round(3)}")

# Reduce to 2 "topics" (latent dimensions)
n_topics = 2
X_lsa = U[:, :n_topics] @ np.diag(s[:n_topics])  # documents in topic space

print(f"\nDocuments projected into {n_topics}-dimensional topic space:")
for i, doc in enumerate(documents):
    topic_desc = "ML/NLP" if X_lsa[i, 0] > 0.1 else "Sports"
    print(f"  [{i}] Topic coords: [{X_lsa[i, 0]:+.3f}, {X_lsa[i, 1]:+.3f}]  "
          f"→ likely {topic_desc}")

# Visualize document similarity in topic space
fig, ax = plt.subplots(figsize=(10, 7))

# Color by topic: docs 0-5 are ML/NLP, docs 6-9 are sports
colors = ['#2196F3'] * 6 + ['#FF5722'] * 4
labels = ['ML/NLP'] * 6 + ['Sports'] * 4

for i in range(len(documents)):
    ax.scatter(X_lsa[i, 0], X_lsa[i, 1], c=colors[i], s=100,
               edgecolors='black', linewidth=0.5, zorder=3)
    # Shorten document text for labeling
    short = documents[i][:40] + "..." if len(documents[i]) > 40 else documents[i]
    ax.annotate(f"[{i}] {short}", (X_lsa[i, 0], X_lsa[i, 1]),
                textcoords="offset points", xytext=(5, 5), fontsize=7)

# Custom legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='#2196F3', label='ML/NLP'),
                   Patch(facecolor='#FF5722', label='Sports')]
ax.legend(handles=legend_elements, loc='best')

ax.set_xlabel('Topic 1')
ax.set_ylabel('Topic 2')
ax.set_title('Latent Semantic Analysis (LSA) -- Documents in Topic Space')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('linear_algebra_for_ml/05_lsa_visualization.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"\nLSA visualization saved to: 05_lsa_visualization.png")

# Document similarity using LSA
print(f"\n--- Document Similarity (Cosine) in LSA Space ---")
from numpy.linalg import norm

def cosine_sim(a, b):
    return np.dot(a, b) / (norm(a) * norm(b))

# Compare a few pairs
pairs = [(0, 1), (0, 6), (6, 7), (2, 3), (0, 8)]
for i, j in pairs:
    sim = cosine_sim(X_lsa[i], X_lsa[j])
    print(f"  doc[{i}] vs doc[{j}]: similarity = {sim:.3f}")
    print(f"    '{documents[i][:50]}...'")
    print(f"    '{documents[j][:50]}...'")
    print()

# Compare with sklearn TruncatedSVD
from sklearn.decomposition import TruncatedSVD

svd_sklearn = TruncatedSVD(n_components=2, random_state=42)
X_lsa_sklearn = svd_sklearn.fit_transform(X_tfidf)

print(f"sklearn explained variance ratio: {svd_sklearn.explained_variance_ratio_.round(4)}")
print(f"Total variance explained: {svd_sklearn.explained_variance_ratio_.sum():.1%}")

print("\n" + "=" * 70)
print("PART 5 COMPLETE -- You now understand SVD!")
print("Key takeaways:")
print("  - SVD: A = U @ Σ @ V^T (works on ANY matrix)")
print("  - Singular values = importance of each component")
print("  - Low-rank approximation → image compression")
print("  - LSA = SVD on TF-IDF matrix → discovers hidden topics")
print("  - SVD is another way to compute PCA")
print("=" * 70)
