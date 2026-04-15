"""
=============================================================================
PART 6: PRACTICAL PATTERNS YOU MUST KNOW
=============================================================================

These are the linear algebra patterns you'll use every day in ML:
  - Distance matrices (for KNN, clustering)
  - Softmax function (neural network output layer)
  - Gradient descent (how every ML model learns)
  - Cross-entropy loss (the standard classification loss)

Topics covered:
  6.1 Distance Matrices (Euclidean, Manhattan, Cosine)
  6.2 Softmax Function
  6.3 Gradient Descent for Linear Regression
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

print("=" * 70)
print("PART 6: PRACTICAL PATTERNS YOU MUST KNOW")
print("=" * 70)

# ──────────────────────────────────────────────────────────────────────
# 6.1 DISTANCE MATRICES (for KNN, Clustering)
# ──────────────────────────────────────────────────────────────────────
print("\n" + "─" * 70)
print("6.1 DISTANCE MATRICES")
print("─" * 70)

# In KNN and clustering, you need to compute distances between
# ALL pairs of points. This creates a distance matrix.

np.random.seed(42)
X = np.array([[1, 2],
              [3, 4],
              [5, 6],
              [7, 8],
              [2, 8]])

print(f"Data points ({X.shape[0]} points in {X.shape[1]}D):\n{X}\n")

# Method 1: Using scipy (most convenient)
from scipy.spatial.distance import cdist

dist_euclidean = cdist(X, X, metric='euclidean')
dist_manhattan = cdist(X, X, metric='cityblock')
dist_cosine = cdist(X, X, metric='cosine')

print("Euclidean distance matrix:")
print(np.array2string(dist_euclidean, precision=2, suppress_small=True))

print(f"\nManhattan distance matrix:")
print(np.array2string(dist_manhattan, precision=2, suppress_small=True))

print(f"\nCosine distance matrix (1 - cosine_similarity):")
print(np.array2string(dist_cosine, precision=3, suppress_small=True))

# Method 2: Pure NumPy vectorized (no loops!)
# Uses broadcasting: X[:, None, :] - X[None, :, :]
# Shape: (5,1,2) - (1,5,2) → broadcasts to (5,5,2)
diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
dist_numpy = np.sqrt(np.sum(diff ** 2, axis=2))

print(f"\nNumPy vectorized Euclidean (same result):")
print(np.array2string(dist_numpy, precision=2, suppress_small=True))
print(f"Matches scipy: {np.allclose(dist_euclidean, dist_numpy)}")

# Method 3: Using the expansion trick (fastest for large datasets)
# ||a - b||² = ||a||² + ||b||² - 2 * a · b
# This avoids creating the huge (n, n, d) intermediate array
sq_norms = np.sum(X ** 2, axis=1)  # ||x_i||² for each point
dot_products = X @ X.T              # x_i · x_j for all pairs
sq_dists = sq_norms[:, np.newaxis] + sq_norms[np.newaxis, :] - 2 * dot_products
# Clip negative values (floating point errors) before sqrt
dist_fast = np.sqrt(np.clip(sq_dists, 0, None))

print(f"\nExpansion trick (fastest for large data):")
print(np.array2string(dist_fast, precision=2, suppress_small=True))
print(f"Matches scipy: {np.allclose(dist_euclidean, dist_fast)}")

# KNN example: find the 2 nearest neighbors for point 0
k = 2
distances_from_0 = dist_euclidean[0]
# argsort gives indices that would sort the array; skip index 0 (self)
nearest = np.argsort(distances_from_0)[1:k + 1]
print(f"\n--- KNN Example ---")
print(f"Point 0: {X[0]}")
print(f"Distances from point 0: {distances_from_0.round(2)}")
print(f"Nearest {k} neighbors: indices {nearest} = {X[nearest]}")

# ──────────────────────────────────────────────────────────────────────
# 6.2 SOFTMAX FUNCTION
# ──────────────────────────────────────────────────────────────────────
print("\n" + "─" * 70)
print("6.2 SOFTMAX FUNCTION")
print("─" * 70)

# Softmax converts raw scores ("logits") into probabilities.
# It's the output layer of every classification neural network.
#
# softmax(z_i) = exp(z_i) / sum(exp(z_j))
#
# Properties:
# - All outputs are in (0, 1)
# - All outputs sum to 1
# - Larger inputs get larger probabilities

def softmax(z):
    """Numerically stable softmax."""
    # Subtract max for numerical stability (prevents overflow in exp)
    z_shifted = z - np.max(z)
    exp_z = np.exp(z_shifted)
    return exp_z / exp_z.sum()

# Example: 3-class classification
logits = np.array([2.0, 1.0, 0.1])
probs = softmax(logits)

print(f"Logits (raw scores): {logits}")
print(f"Softmax (probabilities): {probs.round(4)}")
print(f"Sum of probabilities: {probs.sum():.6f} (should be 1.0)")
print(f"Predicted class: {np.argmax(probs)} (highest probability)")

# Temperature scaling: controls "confidence" of predictions
print(f"\n--- Temperature Scaling ---")
print(f"Lower temperature → more confident, Higher → more uniform")

def softmax_with_temperature(z, temperature=1.0):
    z_scaled = z / temperature
    z_shifted = z_scaled - np.max(z_scaled)
    exp_z = np.exp(z_shifted)
    return exp_z / exp_z.sum()

logits = np.array([2.0, 1.0, 0.5])
for temp in [0.1, 0.5, 1.0, 2.0, 10.0]:
    probs = softmax_with_temperature(logits, temp)
    print(f"  T={temp:<4} → {probs.round(4)}")

# Softmax for a batch of samples (2D -- each row is a sample)
def softmax_batch(Z):
    """Softmax applied to each row of a 2D array."""
    Z_shifted = Z - np.max(Z, axis=1, keepdims=True)
    exp_Z = np.exp(Z_shifted)
    return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)

batch_logits = np.array([[2.0, 1.0, 0.1],
                          [0.5, 2.5, 1.0],
                          [1.0, 1.0, 1.0]])

batch_probs = softmax_batch(batch_logits)
print(f"\nBatch softmax:")
print(f"Logits:\n{batch_logits}")
print(f"\nProbabilities:\n{batch_probs.round(4)}")
print(f"Row sums: {batch_probs.sum(axis=1).round(6)} (all should be 1.0)")

# ─── Cross-Entropy Loss ─────────────────────────────────────────────

print(f"\n--- Cross-Entropy Loss ---")
print(f"The standard loss function for classification.")
print(f"Loss = -sum(y_true * log(y_pred))")

def cross_entropy_loss(y_true_onehot, y_pred_probs):
    """Cross-entropy loss for one sample."""
    # Clip to prevent log(0)
    y_pred_clipped = np.clip(y_pred_probs, 1e-15, 1 - 1e-15)
    return -np.sum(y_true_onehot * np.log(y_pred_clipped))

# True label is class 0, model predictions:
y_true = np.array([1, 0, 0])  # one-hot: class 0

# Good prediction (confident and correct)
y_pred_good = np.array([0.9, 0.05, 0.05])
loss_good = cross_entropy_loss(y_true, y_pred_good)

# Bad prediction (confident but wrong)
y_pred_bad = np.array([0.1, 0.8, 0.1])
loss_bad = cross_entropy_loss(y_true, y_pred_bad)

# Uncertain prediction
y_pred_unsure = np.array([0.4, 0.3, 0.3])
loss_unsure = cross_entropy_loss(y_true, y_pred_unsure)

print(f"\nTrue label: class 0 (one-hot: {y_true})")
print(f"Good prediction {y_pred_good} → loss = {loss_good:.4f} (low = good)")
print(f"Uncertain pred  {y_pred_unsure} → loss = {loss_unsure:.4f}")
print(f"Bad prediction  {y_pred_bad} → loss = {loss_bad:.4f} (high = bad)")

# ──────────────────────────────────────────────────────────────────────
# 6.3 GRADIENT DESCENT (How Every ML Model Learns)
# ──────────────────────────────────────────────────────────────────────
print("\n" + "─" * 70)
print("6.3 GRADIENT DESCENT FOR LINEAR REGRESSION")
print("─" * 70)

# Gradient descent is an iterative optimization algorithm.
# It finds the weights that minimize the loss function by
# repeatedly taking small steps in the direction of steepest descent.
#
# For linear regression with MSE loss:
#   Loss = (1/n) * ||Xw - y||²
#   Gradient = (2/n) * X^T @ (Xw - y)   ← pure linear algebra!
#   Update:  w = w - learning_rate * gradient

np.random.seed(42)

# Generate synthetic data: y = 3x₁ - 2x₂ + 0.5x₃ + 7 (bias)
n_samples = 300
n_features = 3
true_weights = np.array([3.0, -2.0, 0.5])
true_bias = 7.0

X = np.random.randn(n_samples, n_features)
noise = np.random.randn(n_samples) * 0.2
y = X @ true_weights + true_bias + noise

# Add bias column (column of 1s prepended)
X_b = np.column_stack([np.ones(n_samples), X])  # (300, 4)

print(f"Data: {n_samples} samples, {n_features} features")
print(f"True parameters: bias={true_bias}, weights={true_weights}")
print(f"X_b shape (with bias column): {X_b.shape}")

# ─── Full Batch Gradient Descent ────────────────────────────────────

# Initialize weights randomly
w = np.zeros(n_features + 1)  # +1 for bias
learning_rate = 0.01
n_epochs = 1000
loss_history = []

print(f"\n--- Full Batch Gradient Descent ---")
print(f"Learning rate: {learning_rate}")
print(f"Epochs: {n_epochs}")
print(f"Initial weights: {w.round(4)}\n")

for epoch in range(n_epochs):
    # Forward pass: predictions
    predictions = X_b @ w                         # (300,4) @ (4,) = (300,)

    # Compute error
    error = predictions - y                        # (300,)

    # Compute loss (MSE)
    loss = np.mean(error ** 2)
    loss_history.append(loss)

    # Compute gradient: (2/n) * X^T @ error
    gradient = (2 / n_samples) * X_b.T @ error    # (4,300) @ (300,) = (4,)

    # Update weights
    w = w - learning_rate * gradient

    if epoch % 200 == 0 or epoch == n_epochs - 1:
        print(f"  Epoch {epoch:>4d}: loss = {loss:.6f}, "
              f"weights = {w.round(4)}")

print(f"\nFinal weights: {w.round(4)}")
print(f"True params:   [{true_bias}, {', '.join(str(tw) for tw in true_weights)}]")
print(f"Difference:    {np.abs(w - np.array([true_bias, *true_weights])).round(4)}")

# ─── Stochastic Gradient Descent (SGD) ──────────────────────────────

print(f"\n--- Stochastic Gradient Descent (SGD) ---")
print(f"Uses ONE random sample per update (faster, noisier)")

w_sgd = np.zeros(n_features + 1)
learning_rate_sgd = 0.01
n_epochs_sgd = 50  # fewer epochs needed
loss_history_sgd = []

for epoch in range(n_epochs_sgd):
    # Shuffle data each epoch
    indices = np.random.permutation(n_samples)

    epoch_loss = 0
    for i in indices:
        xi = X_b[i]        # single sample (4,)
        yi = y[i]           # single target

        pred = xi @ w_sgd
        error = pred - yi

        # Gradient for single sample
        gradient = 2 * error * xi

        w_sgd = w_sgd - learning_rate_sgd * gradient
        epoch_loss += error ** 2

    avg_loss = epoch_loss / n_samples
    loss_history_sgd.append(avg_loss)

    if epoch % 10 == 0 or epoch == n_epochs_sgd - 1:
        print(f"  Epoch {epoch:>3d}: loss = {avg_loss:.6f}, "
              f"weights = {w_sgd.round(4)}")

# ─── Mini-Batch Gradient Descent ────────────────────────────────────

print(f"\n--- Mini-Batch Gradient Descent ---")
print(f"Best of both worlds: uses small batches (e.g., 32 samples)")

w_mini = np.zeros(n_features + 1)
learning_rate_mini = 0.01
n_epochs_mini = 100
batch_size = 32
loss_history_mini = []

for epoch in range(n_epochs_mini):
    indices = np.random.permutation(n_samples)

    epoch_loss = 0
    n_batches = 0

    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        batch_idx = indices[start:end]
        X_batch = X_b[batch_idx]
        y_batch = y[batch_idx]

        preds = X_batch @ w_mini
        errors = preds - y_batch

        gradient = (2 / len(batch_idx)) * X_batch.T @ errors
        w_mini = w_mini - learning_rate_mini * gradient

        epoch_loss += np.mean(errors ** 2)
        n_batches += 1

    avg_loss = epoch_loss / n_batches
    loss_history_mini.append(avg_loss)

    if epoch % 20 == 0 or epoch == n_epochs_mini - 1:
        print(f"  Epoch {epoch:>3d}: loss = {avg_loss:.6f}, "
              f"weights = {w_mini.round(4)}")

# ─── Compare with Normal Equation ───────────────────────────────────

print(f"\n--- Comparison: All Methods ---")
w_normal = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y

print(f"True parameters:      [{true_bias:.1f}, {true_weights[0]:.1f}, "
      f"{true_weights[1]:.1f}, {true_weights[2]:.1f}]")
print(f"Normal Equation:      {w_normal.round(4)}")
print(f"Batch GD:             {w.round(4)}")
print(f"SGD:                  {w_sgd.round(4)}")
print(f"Mini-Batch GD:        {w_mini.round(4)}")

# ─── Visualize Training Curves ───────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].plot(loss_history)
axes[0].set_title('Batch Gradient Descent')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('MSE Loss')
axes[0].set_yscale('log')
axes[0].grid(True, alpha=0.3)

axes[1].plot(loss_history_sgd, 'orange')
axes[1].set_title('Stochastic Gradient Descent')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('MSE Loss')
axes[1].grid(True, alpha=0.3)

axes[2].plot(loss_history_mini, 'green')
axes[2].set_title('Mini-Batch Gradient Descent')
axes[2].set_xlabel('Epoch')
axes[2].set_ylabel('MSE Loss')
axes[2].grid(True, alpha=0.3)

plt.suptitle('Training Loss Curves -- Three Flavors of Gradient Descent',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('linear_algebra_for_ml/06_gradient_descent_curves.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"\nTraining curves saved to: 06_gradient_descent_curves.png")

print("\n" + "=" * 70)
print("PART 6 COMPLETE -- You now know the practical ML patterns!")
print("Key takeaways:")
print("  - Distance matrix: expansion trick is O(n²d), avoid O(n²d) memory")
print("  - Softmax: z - max(z) for stability, probabilities sum to 1")
print("  - Cross-entropy: the standard classification loss")
print("  - Gradient = (2/n) * X^T @ error → pure linear algebra")
print("  - Three GD flavors: Batch (stable), SGD (fast), Mini-Batch (best)")
print("=" * 70)
