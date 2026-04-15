"""
=============================================================================
PART 5: INFORMATION THEORY
=============================================================================

Information theory provides the mathematical framework for:
- Loss functions (cross-entropy is THE standard classification loss)
- Decision trees (splitting criteria: information gain = entropy reduction)
- Compression and encoding
- Measuring how different two distributions are (KL divergence)

Topics covered:
  5.1 Entropy (measuring uncertainty/information)
  5.2 Cross-Entropy (the loss function of classification)
  5.3 KL Divergence (measuring distribution difference)
  5.4 Mutual Information (feature dependence)
  5.5 Information Gain (how decision trees split)
"""

import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

print("=" * 70)
print("PART 5: INFORMATION THEORY")
print("=" * 70)

# ──────────────────────────────────────────────────────────────────────
# 5.1 ENTROPY
# ──────────────────────────────────────────────────────────────────────
print("\n" + "─" * 70)
print("5.1 ENTROPY (Measuring Uncertainty)")
print("─" * 70)

# Entropy: H(X) = -Σ p(x) * log₂(p(x))
#
# Measures the average "surprise" or "uncertainty" of a distribution.
# - Low entropy: outcomes are predictable (one outcome dominates)
# - High entropy: outcomes are unpredictable (all outcomes equally likely)
#
# For binary: H = -p*log(p) - (1-p)*log(1-p)

def entropy(probs):
    """Shannon entropy of a discrete probability distribution."""
    probs = np.array(probs, dtype=float)
    probs = probs[probs > 0]  # remove zeros (0*log(0) = 0 by convention)
    return -np.sum(probs * np.log2(probs))

# Binary entropy at different probabilities
print("Binary entropy H(p):")
for p in [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]:
    if p == 0 or p == 1:
        h = 0
    else:
        h = entropy([p, 1-p])
    print(f"  p={p:.2f}: H = {h:.4f} bits")

print(f"\nMaximum entropy at p=0.5 (most uncertain)")
print(f"Minimum entropy at p=0 or p=1 (completely certain)")

# Multi-class entropy
print(f"\n--- Multi-class Entropy ---")
distributions = {
    'Uniform (4 classes)': [0.25, 0.25, 0.25, 0.25],
    'Peaked':              [0.7, 0.1, 0.1, 0.1],
    'Very peaked':         [0.97, 0.01, 0.01, 0.01],
    'Deterministic':       [1.0, 0.0, 0.0, 0.0],
}

for name, probs in distributions.items():
    h = entropy(probs)
    print(f"  {name:>25s}: H = {h:.4f} bits  probs = {probs}")

print(f"\n  Maximum entropy for 4 classes = log₂(4) = {np.log2(4):.4f} bits (uniform)")

# Visualize binary entropy
p_range = np.linspace(0.001, 0.999, 200)
h_values = [-p * np.log2(p) - (1-p) * np.log2(1-p) for p in p_range]

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(p_range, h_values, 'b-', linewidth=2)
ax.set_xlabel('p (probability of class 1)')
ax.set_ylabel('Entropy H(p) in bits')
ax.set_title('Binary Entropy Function')
ax.axvline(x=0.5, color='red', linestyle='--', alpha=0.5, label='Max entropy at p=0.5')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('probability_for_ml/05_binary_entropy.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"\nBinary entropy plot saved to: 05_binary_entropy.png")

# ──────────────────────────────────────────────────────────────────────
# 5.2 CROSS-ENTROPY (The Loss Function of Classification)
# ──────────────────────────────────────────────────────────────────────
print("\n" + "─" * 70)
print("5.2 CROSS-ENTROPY")
print("─" * 70)

# Cross-entropy: H(p, q) = -Σ p(x) * log(q(x))
#
# Measures how well distribution q approximates the true distribution p.
# In ML:
#   p = true labels (one-hot)
#   q = model's predicted probabilities
#   H(p, q) = the cross-entropy LOSS

def cross_entropy(p_true, q_pred):
    """Cross-entropy between true distribution p and predicted distribution q."""
    q_pred = np.clip(q_pred, 1e-15, 1 - 1e-15)
    return -np.sum(p_true * np.log(q_pred))

def binary_cross_entropy(y_true, y_pred):
    """Binary cross-entropy for a single sample."""
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    return -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

# Example: 3-class classification
y_true = np.array([1, 0, 0])  # true class is 0

# Different model predictions
predictions = {
    'Perfect':    np.array([1.0, 0.0, 0.0]),
    'Good':       np.array([0.9, 0.05, 0.05]),
    'Okay':       np.array([0.6, 0.2, 0.2]),
    'Uncertain':  np.array([0.34, 0.33, 0.33]),
    'Wrong':      np.array([0.1, 0.8, 0.1]),
    'Very wrong': np.array([0.01, 0.98, 0.01]),
}

print(f"True label: class 0, one-hot = {y_true}")
print(f"\n{'Prediction':<15} {'Probabilities':<25} {'Cross-Entropy':<15} {'Quality'}")
print("-" * 70)
for name, pred in predictions.items():
    ce = cross_entropy(y_true, pred)
    print(f"{name:<15} {str(pred):<25} {ce:<15.4f} {'← lower is better' if name == 'Perfect' else ''}")

print(f"\nRelationship: Cross-entropy = Entropy + KL Divergence")
print(f"  H(p, q) = H(p) + D_KL(p || q)")
print(f"  When q = p: H(p, p) = H(p) (minimum possible)")

# Binary cross-entropy visualization
print(f"\n--- Binary Cross-Entropy ---")
print(f"True label = 1:")
pred_range = np.linspace(0.01, 0.99, 100)
bce_values = [-np.log(p) for p in pred_range]

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(pred_range, bce_values, 'r-', linewidth=2, label='True y=1')
ax.plot(pred_range, [-np.log(1-p) for p in pred_range], 'b-', linewidth=2, label='True y=0')
ax.set_xlabel('Predicted probability')
ax.set_ylabel('Binary Cross-Entropy Loss')
ax.set_title('Binary Cross-Entropy Loss')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('probability_for_ml/05_cross_entropy.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Cross-entropy plot saved to: 05_cross_entropy.png")

# ──────────────────────────────────────────────────────────────────────
# 5.3 KL DIVERGENCE (Measuring Distribution Difference)
# ──────────────────────────────────────────────────────────────────────
print("\n" + "─" * 70)
print("5.3 KL DIVERGENCE")
print("─" * 70)

# KL Divergence: D_KL(P || Q) = Σ p(x) * log(p(x) / q(x))
#
# Measures how different Q is from P.
# - D_KL >= 0 (always non-negative)
# - D_KL = 0 only when P = Q
# - NOT symmetric: D_KL(P||Q) ≠ D_KL(Q||P)
#
# ML uses:
# - VAE loss (Variational Autoencoders)
# - Knowledge distillation
# - Comparing learned distribution to prior

def kl_divergence(p, q):
    """KL divergence D_KL(P || Q) for discrete distributions."""
    p = np.array(p, dtype=float)
    q = np.array(q, dtype=float)
    # Only compute where p > 0
    mask = p > 0
    q_safe = np.clip(q[mask], 1e-15, None)
    return np.sum(p[mask] * np.log(p[mask] / q_safe))

# Example: comparing distributions
p = np.array([0.4, 0.3, 0.2, 0.1])  # true distribution
q1 = np.array([0.4, 0.3, 0.2, 0.1]) # same as p
q2 = np.array([0.25, 0.25, 0.25, 0.25])  # uniform
q3 = np.array([0.1, 0.1, 0.4, 0.4]) # very different

print(f"P (true):       {p}")
print(f"Q1 (same as P): {q1}")
print(f"Q2 (uniform):   {q2}")
print(f"Q3 (different): {q3}")

print(f"\nD_KL(P || Q1) = {kl_divergence(p, q1):.6f}  (same → 0)")
print(f"D_KL(P || Q2) = {kl_divergence(p, q2):.6f}")
print(f"D_KL(P || Q3) = {kl_divergence(p, q3):.6f}")

# Asymmetry of KL divergence
print(f"\n--- KL Divergence is NOT symmetric ---")
print(f"D_KL(P || Q2) = {kl_divergence(p, q2):.6f}")
print(f"D_KL(Q2 || P) = {kl_divergence(q2, p):.6f}")
print(f"These are different!")

# KL divergence between two Gaussians (closed form)
def kl_gaussian(mu1, sigma1, mu2, sigma2):
    """KL divergence between two univariate Gaussians."""
    return (np.log(sigma2/sigma1) +
            (sigma1**2 + (mu1 - mu2)**2) / (2 * sigma2**2) - 0.5)

print(f"\n--- KL Divergence between Gaussians ---")
print(f"D_KL(N(0,1) || N(0,1)) = {kl_gaussian(0, 1, 0, 1):.6f}")
print(f"D_KL(N(0,1) || N(1,1)) = {kl_gaussian(0, 1, 1, 1):.6f}")
print(f"D_KL(N(0,1) || N(0,2)) = {kl_gaussian(0, 1, 0, 2):.6f}")
print(f"D_KL(N(0,1) || N(3,2)) = {kl_gaussian(0, 1, 3, 2):.6f}")

# ──────────────────────────────────────────────────────────────────────
# 5.4 MUTUAL INFORMATION
# ──────────────────────────────────────────────────────────────────────
print("\n" + "─" * 70)
print("5.4 MUTUAL INFORMATION")
print("─" * 70)

# Mutual Information: I(X; Y) = H(X) + H(Y) - H(X, Y)
# Measures how much knowing X reduces uncertainty about Y.
# - I(X; Y) = 0 → X and Y are independent
# - I(X; Y) = H(X) → X fully determines Y
#
# Unlike correlation, MI captures nonlinear relationships!

from sklearn.feature_selection import mutual_info_classif
from sklearn.datasets import load_iris

iris = load_iris()
X, y = iris.data, iris.target

mi_scores = mutual_info_classif(X, y, random_state=42)

print(f"Mutual Information of each Iris feature with target:")
for name, score in zip(iris.feature_names, mi_scores):
    bar = "█" * int(score * 30)
    print(f"  {name:>20s}: MI = {score:.4f}  {bar}")

print(f"\nHighest MI = most informative feature for classification")

# ──────────────────────────────────────────────────────────────────────
# 5.5 INFORMATION GAIN (How Decision Trees Split)
# ──────────────────────────────────────────────────────────────────────
print("\n" + "─" * 70)
print("5.5 INFORMATION GAIN (Decision Trees)")
print("─" * 70)

# Information Gain = Entropy(parent) - Weighted_Entropy(children)
# Decision trees choose the split that maximizes information gain.

def entropy_from_counts(counts):
    """Compute entropy from class counts."""
    total = sum(counts)
    if total == 0:
        return 0
    probs = [c / total for c in counts if c > 0]
    return -sum(p * np.log2(p) for p in probs)

def information_gain(parent_counts, left_counts, right_counts):
    """Compute information gain from a binary split."""
    total = sum(parent_counts)
    left_total = sum(left_counts)
    right_total = sum(right_counts)

    h_parent = entropy_from_counts(parent_counts)
    h_left = entropy_from_counts(left_counts)
    h_right = entropy_from_counts(right_counts)

    h_children = (left_total / total) * h_left + (right_total / total) * h_right

    return h_parent - h_children

# Example: email spam classification
# Parent node: 100 emails (60 spam, 40 not spam)
parent = [60, 40]
print(f"Parent node: {parent[0]} spam, {parent[1]} ham")
print(f"Parent entropy: {entropy_from_counts(parent):.4f} bits")

# Split option 1: by "contains 'free'"
left_1 = [50, 5]    # contains "free": 50 spam, 5 ham
right_1 = [10, 35]  # doesn't contain "free": 10 spam, 35 ham
ig_1 = information_gain(parent, left_1, right_1)

# Split option 2: by "sent at night"
left_2 = [35, 20]   # sent at night: 35 spam, 20 ham
right_2 = [25, 20]  # sent during day: 25 spam, 20 ham
ig_2 = information_gain(parent, left_2, right_2)

print(f"\nSplit 1 - 'contains free':")
print(f"  Left (yes):  {left_1} → H = {entropy_from_counts(left_1):.4f}")
print(f"  Right (no):  {right_1} → H = {entropy_from_counts(right_1):.4f}")
print(f"  Information Gain = {ig_1:.4f} bits")

print(f"\nSplit 2 - 'sent at night':")
print(f"  Left (yes):  {left_2} → H = {entropy_from_counts(left_2):.4f}")
print(f"  Right (no):  {right_2} → H = {entropy_from_counts(right_2):.4f}")
print(f"  Information Gain = {ig_2:.4f} bits")

print(f"\nBest split: {'contains free' if ig_1 > ig_2 else 'sent at night'} "
      f"(higher IG = {max(ig_1, ig_2):.4f})")

# Gini impurity (alternative to entropy, used by sklearn's DecisionTree)
def gini_impurity(counts):
    total = sum(counts)
    if total == 0:
        return 0
    probs = [c / total for c in counts]
    return 1 - sum(p ** 2 for p in probs)

print(f"\n--- Gini Impurity (alternative to entropy) ---")
print(f"Parent Gini:    {gini_impurity(parent):.4f}")
print(f"Split 1 - Left: {gini_impurity(left_1):.4f}, Right: {gini_impurity(right_1):.4f}")
print(f"Split 2 - Left: {gini_impurity(left_2):.4f}, Right: {gini_impurity(right_2):.4f}")

print("\n" + "=" * 70)
print("PART 5 COMPLETE -- You now understand information theory for ML!")
print("Key takeaways:")
print("  - Entropy: measures uncertainty (high = unpredictable)")
print("  - Cross-entropy: THE classification loss function")
print("  - KL divergence: measures how different two distributions are")
print("  - Mutual information: detects nonlinear feature-target relationships")
print("  - Information gain: how decision trees choose splits")
print("=" * 70)
