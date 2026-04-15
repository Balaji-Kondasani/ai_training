"""
=============================================================================
EXERCISES: 7 Practice Problems Combining All Concepts
=============================================================================

These exercises test everything from Parts 1-6.
Each exercise is self-contained and includes verification.

  Exercise 1: Linear Regression using the Normal Equation (no sklearn)
  Exercise 2: PCA from Scratch vs sklearn PCA
  Exercise 3: Image Compression using SVD
  Exercise 4: Cosine Similarity between TF-IDF vectors
  Exercise 5: Gradient Descent for Linear Regression
  Exercise 6: KNN Classifier using Distance Matrices
  Exercise 7: Softmax and Cross-Entropy Loss from Scratch
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

print("=" * 70)
print("EXERCISES: 7 Practice Problems Combining All Concepts")
print("=" * 70)

# ╔════════════════════════════════════════════════════════════════════╗
# ║  EXERCISE 1: Linear Regression using the Normal Equation         ║
# ╚════════════════════════════════════════════════════════════════════╝
print("\n" + "═" * 70)
print("EXERCISE 1: Linear Regression -- Normal Equation (no sklearn)")
print("═" * 70)

# Task: Predict house prices given area, bedrooms, and age.
# Implement ONLY using NumPy.

np.random.seed(42)
n = 500

# Synthetic data: price = 150*area + 50000*bedrooms - 2000*age + 100000
area = np.random.uniform(50, 300, n)        # square meters
bedrooms = np.random.randint(1, 6, n).astype(float)
age = np.random.uniform(0, 50, n)           # years
noise = np.random.randn(n) * 10000

true_coeffs = np.array([100000, 150, 50000, -2000])  # [bias, area, beds, age]
X_raw = np.column_stack([area, bedrooms, age])
y = 100000 + 150 * area + 50000 * bedrooms - 2000 * age + noise

# Step 1: Add bias column
X = np.column_stack([np.ones(n), X_raw])

# Step 2: Split into train/test (80/20) manually
split = int(0.8 * n)
indices = np.random.permutation(n)
train_idx, test_idx = indices[:split], indices[split:]

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

# Step 3: Normal Equation: θ = (X^T X)^(-1) X^T y
theta = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ y_train

print(f"True coefficients:   {true_coeffs}")
print(f"Learned coefficients: {theta.round(2)}")

# Step 4: Evaluate on test set
y_pred = X_test @ theta
mse = np.mean((y_test - y_pred) ** 2)
rmse = np.sqrt(mse)
ss_res = np.sum((y_test - y_pred) ** 2)
ss_tot = np.sum((y_test - y_test.mean()) ** 2)
r2 = 1 - ss_res / ss_tot

print(f"\nTest Set Metrics:")
print(f"  RMSE: {rmse:.2f}")
print(f"  R²:   {r2:.4f}")

# Step 5: Also solve using lstsq (more stable)
theta_lstsq = np.linalg.lstsq(X_train, y_train, rcond=None)[0]
print(f"\nlstsq coefficients:  {theta_lstsq.round(2)}")
print(f"Match Normal Eq: {np.allclose(theta, theta_lstsq, atol=0.01)}")

print("\n✓ Exercise 1 Complete!")

# ╔════════════════════════════════════════════════════════════════════╗
# ║  EXERCISE 2: PCA from Scratch vs sklearn PCA                     ║
# ╚════════════════════════════════════════════════════════════════════╝
print("\n" + "═" * 70)
print("EXERCISE 2: PCA from Scratch vs sklearn")
print("═" * 70)

from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load Wine dataset: 178 samples, 13 features
wine = load_wine()
X_wine = wine.data
y_wine = wine.target

print(f"Wine dataset: {X_wine.shape[0]} samples, {X_wine.shape[1]} features")

# Step 1: Standardize (important for PCA when features have different scales!)
X_std = (X_wine - X_wine.mean(axis=0)) / X_wine.std(axis=0)

# Step 2: Covariance matrix
cov_mat = np.cov(X_std, rowvar=False)
print(f"Covariance matrix shape: {cov_mat.shape}")

# Step 3: Eigendecomposition
eigenvalues, eigenvectors = np.linalg.eigh(cov_mat)

# Step 4: Sort descending
idx = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

# Step 5: Project onto top 2 components
n_comp = 2
W = eigenvectors[:, :n_comp]
X_pca_manual = X_std @ W

# Compare with sklearn
pca = PCA(n_components=2)
X_pca_sklearn = pca.fit_transform(X_std)

# Explained variance
exp_var = eigenvalues / eigenvalues.sum()
print(f"\nOur explained variance (top 5): {exp_var[:5].round(4)}")
print(f"sklearn explained variance:      {pca.explained_variance_ratio_.round(4)}")
print(f"Total (2 components): ours={exp_var[:2].sum():.4f}, "
      f"sklearn={pca.explained_variance_ratio_.sum():.4f}")

# Visualize both
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
colors = ['#e41a1c', '#377eb8', '#4daf4a']
class_names = wine.target_names

for i, name in enumerate(class_names):
    mask = y_wine == i
    axes[0].scatter(X_pca_manual[mask, 0], X_pca_manual[mask, 1],
                    c=colors[i], label=name, alpha=0.7, edgecolors='k', linewidth=0.5)
    axes[1].scatter(X_pca_sklearn[mask, 0], X_pca_sklearn[mask, 1],
                    c=colors[i], label=name, alpha=0.7, edgecolors='k', linewidth=0.5)

axes[0].set_title('Our PCA from Scratch')
axes[0].set_xlabel(f'PC1 ({exp_var[0]:.1%})')
axes[0].set_ylabel(f'PC2 ({exp_var[1]:.1%})')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].set_title('sklearn PCA')
axes[1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
axes[1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('linear_algebra_for_ml/07_ex2_pca_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"\nPCA comparison saved to: 07_ex2_pca_comparison.png")

print("\n✓ Exercise 2 Complete!")

# ╔════════════════════════════════════════════════════════════════════╗
# ║  EXERCISE 3: Image Compression using SVD                         ║
# ╚════════════════════════════════════════════════════════════════════╝
print("\n" + "═" * 70)
print("EXERCISE 3: Image Compression using SVD")
print("═" * 70)

# Create a more interesting synthetic image
np.random.seed(42)
size = 200
x = np.linspace(-3, 3, size)
y_grid = np.linspace(-3, 3, size)
X_mesh, Y_mesh = np.meshgrid(x, y_grid)

# A pattern with both smooth and sharp features
image = (np.sin(X_mesh ** 2 + Y_mesh ** 2) +
         np.exp(-((X_mesh - 1) ** 2 + (Y_mesh - 1) ** 2)) * 3 +
         0.1 * np.random.randn(size, size))

print(f"Image shape: {image.shape}")
print(f"Original storage: {image.size} values")

# SVD
U, s, Vt = np.linalg.svd(image)

# Compress at different ranks
ranks = [1, 5, 10, 25, 50, 100]
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.flatten()

axes[0].imshow(image, cmap='inferno')
axes[0].set_title(f'Original ({size}×{size})')
axes[0].axis('off')

for idx, k in enumerate(ranks):
    compressed = U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]
    storage = k * (size + size + 1)
    ratio = storage / image.size
    error = np.linalg.norm(image - compressed, 'fro') / np.linalg.norm(image, 'fro')

    axes[idx + 1].imshow(compressed, cmap='inferno')
    axes[idx + 1].set_title(f'Rank {k}\n{ratio:.0%} storage, {error:.2%} error')
    axes[idx + 1].axis('off')

# Hide the last subplot if empty
if len(ranks) + 1 < len(axes):
    axes[-1].axis('off')

plt.suptitle('Image Compression using SVD at Different Ranks', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('linear_algebra_for_ml/07_ex3_svd_compression.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Compression comparison saved to: 07_ex3_svd_compression.png")

# Find the optimal rank for 95% energy
energy = np.cumsum(s ** 2) / np.sum(s ** 2)
k_95 = np.searchsorted(energy, 0.95) + 1
print(f"\nSingular values needed for 95% energy: {k_95} out of {len(s)}")
print(f"Compression ratio: {k_95 * (size + size + 1) / image.size:.1%}")

print("\n✓ Exercise 3 Complete!")

# ╔════════════════════════════════════════════════════════════════════╗
# ║  EXERCISE 4: Cosine Similarity between TF-IDF Vectors            ║
# ╚════════════════════════════════════════════════════════════════════╝
print("\n" + "═" * 70)
print("EXERCISE 4: Cosine Similarity between TF-IDF Vectors")
print("═" * 70)

from sklearn.feature_extraction.text import TfidfVectorizer

sentences = [
    "I love machine learning and artificial intelligence",
    "Deep learning is a subset of machine learning",
    "Natural language processing uses neural networks",
    "The cat sat on the mat near the door",
    "Dogs are loyal and friendly animals",
    "Python is great for data science projects",
    "Java and C++ are compiled programming languages",
    "The weather is sunny and warm today",
    "Football and basketball are popular sports",
    "I enjoy cooking Italian food at home",
]

print("Sentences:")
for i, s in enumerate(sentences):
    print(f"  [{i}] {s}")

# Create TF-IDF vectors
tfidf = TfidfVectorizer()
X_tfidf = tfidf.fit_transform(sentences).toarray()
print(f"\nTF-IDF matrix shape: {X_tfidf.shape}")

# Compute cosine similarity matrix (all pairs)
norms = np.linalg.norm(X_tfidf, axis=1, keepdims=True)
norms_safe = np.where(norms == 0, 1, norms)  # avoid division by zero
X_normalized = X_tfidf / norms_safe
cos_sim_matrix = X_normalized @ X_normalized.T

print(f"\nCosine Similarity Matrix (10×10):")
np.set_printoptions(precision=2, suppress=True, linewidth=120)
print(cos_sim_matrix)

# Find the most similar pairs
n_sentences = len(sentences)
pairs = []
for i in range(n_sentences):
    for j in range(i + 1, n_sentences):
        pairs.append((i, j, cos_sim_matrix[i, j]))

pairs.sort(key=lambda x: x[2], reverse=True)

print(f"\nTop 5 most similar pairs:")
for i, j, sim in pairs[:5]:
    print(f"  [{i}]-[{j}] sim={sim:.3f}")
    print(f"    '{sentences[i]}'")
    print(f"    '{sentences[j]}'")

print(f"\nLeast similar pair:")
i, j, sim = pairs[-1]
print(f"  [{i}]-[{j}] sim={sim:.3f}")
print(f"    '{sentences[i]}'")
print(f"    '{sentences[j]}'")

# Visualize as heatmap
fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(cos_sim_matrix, cmap='YlOrRd', vmin=0, vmax=1)
ax.set_xticks(range(n_sentences))
ax.set_yticks(range(n_sentences))
ax.set_xticklabels([f"[{i}]" for i in range(n_sentences)])
ax.set_yticklabels([f"[{i}] {s[:30]}..." for i, s in enumerate(sentences)], fontsize=8)
plt.colorbar(im, label='Cosine Similarity')
ax.set_title('Cosine Similarity Heatmap (TF-IDF Vectors)')
plt.tight_layout()
plt.savefig('linear_algebra_for_ml/07_ex4_cosine_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"\nHeatmap saved to: 07_ex4_cosine_heatmap.png")

print("\n✓ Exercise 4 Complete!")

# ╔════════════════════════════════════════════════════════════════════╗
# ║  EXERCISE 5: Gradient Descent for Linear Regression              ║
# ╚════════════════════════════════════════════════════════════════════╝
print("\n" + "═" * 70)
print("EXERCISE 5: Gradient Descent for Linear Regression")
print("═" * 70)

np.random.seed(42)

# Generate 2D data for visualization: y = 3x + 7 + noise
n = 100
X_1d = np.random.uniform(-5, 5, n)
y_gd = 3 * X_1d + 7 + np.random.randn(n) * 1.5

# Build feature matrix with bias
X_gd = np.column_stack([np.ones(n), X_1d])  # (100, 2)

# Gradient descent implementation with tracking
w = np.array([0.0, 0.0])  # [bias, slope] -- start at zero
lr = 0.01
epochs = 200
history = {'loss': [], 'w0': [], 'w1': []}

for epoch in range(epochs):
    predictions = X_gd @ w
    errors = predictions - y_gd
    loss = np.mean(errors ** 2)
    gradient = (2 / n) * X_gd.T @ errors
    w = w - lr * gradient

    history['loss'].append(loss)
    history['w0'].append(w[0])
    history['w1'].append(w[1])

print(f"True parameters: bias=7.0, slope=3.0")
print(f"Learned:         bias={w[0]:.4f}, slope={w[1]:.4f}")
print(f"Final loss:      {history['loss'][-1]:.4f}")

# Visualize: data + fit line + loss curve + parameter path
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Data and fit
axes[0].scatter(X_1d, y_gd, alpha=0.5, s=20, c='steelblue')
x_line = np.linspace(-5, 5, 100)
axes[0].plot(x_line, w[0] + w[1] * x_line, 'r-', linewidth=2, label=f'y = {w[0]:.1f} + {w[1]:.1f}x')
axes[0].set_xlabel('x')
axes[0].set_ylabel('y')
axes[0].set_title('Data and Learned Line')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Loss curve
axes[1].plot(history['loss'])
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('MSE Loss')
axes[1].set_title('Loss Over Training')
axes[1].set_yscale('log')
axes[1].grid(True, alpha=0.3)

# Parameter trajectory
axes[2].plot(history['w0'], history['w1'], 'b.-', alpha=0.5, markersize=2)
axes[2].plot(history['w0'][0], history['w1'][0], 'go', markersize=10, label='Start')
axes[2].plot(history['w0'][-1], history['w1'][-1], 'r*', markersize=15, label='End')
axes[2].plot(7.0, 3.0, 'kx', markersize=12, markeredgewidth=3, label='True')
axes[2].set_xlabel('Bias (w0)')
axes[2].set_ylabel('Slope (w1)')
axes[2].set_title('Parameter Trajectory')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.suptitle('Gradient Descent Visualization', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('linear_algebra_for_ml/07_ex5_gradient_descent.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"\nVisualization saved to: 07_ex5_gradient_descent.png")

print("\n✓ Exercise 5 Complete!")

# ╔════════════════════════════════════════════════════════════════════╗
# ║  EXERCISE 6: KNN Classifier using Distance Matrices              ║
# ╚════════════════════════════════════════════════════════════════════╝
print("\n" + "═" * 70)
print("EXERCISE 6: KNN Classifier from Scratch (NumPy Only)")
print("═" * 70)

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Load data
iris = load_iris()
X_iris = iris.data
y_iris = iris.target

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_iris, y_iris, test_size=0.3, random_state=42
)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set:     {X_test.shape[0]} samples")

def knn_predict(X_train, y_train, X_test, k=5):
    """KNN classifier using only NumPy distance computations."""
    # Compute all pairwise distances: test vs train
    # Using expansion trick: ||a-b||² = ||a||² + ||b||² - 2*a·b
    train_sq = np.sum(X_train ** 2, axis=1)    # (n_train,)
    test_sq = np.sum(X_test ** 2, axis=1)      # (n_test,)
    cross = X_test @ X_train.T                  # (n_test, n_train)

    sq_dists = test_sq[:, np.newaxis] + train_sq[np.newaxis, :] - 2 * cross
    distances = np.sqrt(np.clip(sq_dists, 0, None))

    predictions = np.zeros(X_test.shape[0], dtype=int)

    for i in range(X_test.shape[0]):
        # Find k nearest neighbors
        k_nearest_idx = np.argsort(distances[i])[:k]
        k_nearest_labels = y_train[k_nearest_idx]

        # Majority vote
        counts = np.bincount(k_nearest_labels)
        predictions[i] = np.argmax(counts)

    return predictions

# Test our implementation
for k in [1, 3, 5, 7]:
    y_pred_ours = knn_predict(X_train, y_train, X_test, k=k)
    accuracy_ours = np.mean(y_pred_ours == y_test)

    # Compare with sklearn
    knn_sklearn = KNeighborsClassifier(n_neighbors=k)
    knn_sklearn.fit(X_train, y_train)
    accuracy_sklearn = knn_sklearn.score(X_test, y_test)

    print(f"k={k}: Our accuracy = {accuracy_ours:.4f}, "
          f"sklearn accuracy = {accuracy_sklearn:.4f}, "
          f"Match = {np.isclose(accuracy_ours, accuracy_sklearn)}")

print("\n✓ Exercise 6 Complete!")

# ╔════════════════════════════════════════════════════════════════════╗
# ║  EXERCISE 7: Softmax and Cross-Entropy Loss from Scratch         ║
# ╚════════════════════════════════════════════════════════════════════╝
print("\n" + "═" * 70)
print("EXERCISE 7: Softmax + Cross-Entropy + Simple Classifier")
print("═" * 70)

# Build a simple softmax classifier (multinomial logistic regression)
# trained with gradient descent -- ALL from scratch.

from sklearn.datasets import load_iris

iris = load_iris()
X = iris.data       # (150, 4)
y = iris.target     # (150,) with values 0, 1, 2

# Standardize features
X = (X - X.mean(axis=0)) / X.std(axis=0)

# One-hot encode targets
n_classes = 3
n_samples = X.shape[0]
n_features = X.shape[1]
Y_onehot = np.zeros((n_samples, n_classes))
Y_onehot[np.arange(n_samples), y] = 1

# Train/test split
np.random.seed(42)
perm = np.random.permutation(n_samples)
split = int(0.8 * n_samples)
train_idx, test_idx = perm[:split], perm[split:]

X_train, X_test = X[train_idx], X[test_idx]
Y_train, Y_test = Y_onehot[train_idx], Y_onehot[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

print(f"Training: {X_train.shape[0]} samples, Test: {X_test.shape[0]} samples")
print(f"Features: {n_features}, Classes: {n_classes}")

def softmax(Z):
    """Numerically stable softmax for batches (each row is a sample)."""
    Z_shifted = Z - np.max(Z, axis=1, keepdims=True)
    exp_Z = np.exp(Z_shifted)
    return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)

def cross_entropy(Y_true, Y_pred):
    """Average cross-entropy loss over a batch."""
    Y_pred_clipped = np.clip(Y_pred, 1e-15, 1 - 1e-15)
    return -np.mean(np.sum(Y_true * np.log(Y_pred_clipped), axis=1))

# Initialize weights and bias
W = np.random.randn(n_features, n_classes) * 0.01  # (4, 3)
b = np.zeros((1, n_classes))                         # (1, 3)

lr = 0.1
epochs = 500
loss_history = []

print(f"\nTraining softmax classifier...")
print(f"Learning rate: {lr}, Epochs: {epochs}\n")

for epoch in range(epochs):
    # Forward pass
    logits = X_train @ W + b           # (120, 3)
    probs = softmax(logits)            # (120, 3)

    # Compute loss
    loss = cross_entropy(Y_train, probs)
    loss_history.append(loss)

    # Backward pass (gradient computation)
    # For softmax + cross-entropy, the gradient is beautifully simple:
    # dL/dZ = (probs - Y_true) / n_samples
    dZ = (probs - Y_train) / X_train.shape[0]

    # Gradients for W and b
    dW = X_train.T @ dZ     # (4, 120) @ (120, 3) = (4, 3)
    db = np.sum(dZ, axis=0, keepdims=True)  # (1, 3)

    # Update parameters
    W -= lr * dW
    b -= lr * db

    if epoch % 100 == 0 or epoch == epochs - 1:
        train_pred = np.argmax(probs, axis=1)
        train_acc = np.mean(train_pred == y_train)
        print(f"  Epoch {epoch:>3d}: loss = {loss:.4f}, train_acc = {train_acc:.2%}")

# Evaluate on test set
test_logits = X_test @ W + b
test_probs = softmax(test_logits)
test_pred = np.argmax(test_probs, axis=1)
test_acc = np.mean(test_pred == y_test)
test_loss = cross_entropy(Y_test, test_probs)

print(f"\n--- Test Results ---")
print(f"Test loss:     {test_loss:.4f}")
print(f"Test accuracy: {test_acc:.2%}")

# Confusion matrix (from scratch)
confusion = np.zeros((n_classes, n_classes), dtype=int)
for true, pred in zip(y_test, test_pred):
    confusion[true, pred] += 1

print(f"\nConfusion Matrix:")
print(f"             Predicted")
print(f"             {iris.target_names}")
print(f"Actual:")
for i, name in enumerate(iris.target_names):
    print(f"  {name:>10}: {confusion[i]}")

# Compare with sklearn
from sklearn.linear_model import LogisticRegression

lr_sklearn = LogisticRegression(max_iter=1000, random_state=42)
lr_sklearn.fit(X_train, y_train)
sklearn_acc = lr_sklearn.score(X_test, y_test)
print(f"\nsklearn LogisticRegression accuracy: {sklearn_acc:.2%}")
print(f"Our softmax classifier accuracy:     {test_acc:.2%}")

# Plot training loss
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(loss_history)
ax.set_xlabel('Epoch')
ax.set_ylabel('Cross-Entropy Loss')
ax.set_title(f'Softmax Classifier Training (Final Test Acc: {test_acc:.1%})')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('linear_algebra_for_ml/07_ex7_softmax_training.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"\nTraining plot saved to: 07_ex7_softmax_training.png")

print("\n✓ Exercise 7 Complete!")

# ══════════════════════════════════════════════════════════════════════
print("\n" + "═" * 70)
print("ALL 7 EXERCISES COMPLETE!")
print("═" * 70)
print("""
Summary of what you built from scratch using ONLY NumPy:

  1. Linear Regression (Normal Equation) -- closed-form solution
  2. PCA -- eigendecomposition of covariance matrix
  3. Image Compression -- SVD low-rank approximation
  4. Document Similarity -- TF-IDF + cosine similarity
  5. Gradient Descent -- iterative optimization
  6. KNN Classifier -- pairwise distance computation
  7. Softmax Classifier -- forward pass, cross-entropy, backprop

Each of these demonstrates a core linear algebra concept
applied to a real ML problem. You're now ready to understand
what sklearn and PyTorch do under the hood!
""")
