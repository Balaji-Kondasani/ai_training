"""
=============================================================================
PART 3: EXPECTATION, VARIANCE, COVARIANCE & CORRELATION
=============================================================================

These statistical measures are the building blocks of every ML algorithm:
- Mean (expectation) → predictions, centroids in clustering
- Variance → uncertainty, spread, regularization
- Covariance → feature relationships, PCA
- Correlation → feature selection, multicollinearity detection

Topics covered:
  3.1 Expectation (Mean)
  3.2 Variance and Standard Deviation
  3.3 Covariance and Correlation
  3.4 Law of Large Numbers
  3.5 Central Limit Theorem
"""

import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

print("=" * 70)
print("PART 3: EXPECTATION, VARIANCE, COVARIANCE & CORRELATION")
print("=" * 70)

# ──────────────────────────────────────────────────────────────────────
# 3.1 EXPECTATION (MEAN)
# ──────────────────────────────────────────────────────────────────────
print("\n" + "─" * 70)
print("3.1 EXPECTATION (MEAN)")
print("─" * 70)

# E[X] = sum of (value * probability) for all values
# For data: E[X] ≈ (1/n) * sum(x_i)

# Discrete example: expected value of a loaded die
faces = np.array([1, 2, 3, 4, 5, 6])
# Loaded: 6 appears twice as often
probs = np.array([1, 1, 1, 1, 1, 2]) / 7

expected_value = np.sum(faces * probs)
print(f"Loaded die probabilities: {probs.round(4)}")
print(f"E[X] = {expected_value:.4f}")
print(f"Fair die E[X] = {faces.mean():.4f}")

# Simulate to verify
np.random.seed(42)
rolls = np.random.choice(faces, size=100000, p=probs)
print(f"Simulated mean: {rolls.mean():.4f}")

# Properties of expectation
# E[aX + b] = a*E[X] + b  (linearity)
X = np.random.randn(10000)
a, b = 3, 5
Y = a * X + b
print(f"\nLinearity: E[{a}X + {b}] = {a}*E[X] + {b}")
print(f"  E[X] = {X.mean():.4f}")
print(f"  E[{a}X + {b}] = {Y.mean():.4f}")
print(f"  {a}*E[X] + {b} = {a * X.mean() + b:.4f}")

# E[X + Y] = E[X] + E[Y]  (always true, even if dependent!)
X = np.random.randn(10000)
Y = X ** 2  # Y depends on X!
print(f"\nE[X + Y] = E[X] + E[Y] (even for dependent variables!):")
print(f"  E[X+Y] = {(X + Y).mean():.4f}")
print(f"  E[X] + E[Y] = {X.mean() + Y.mean():.4f}")

# ──────────────────────────────────────────────────────────────────────
# 3.2 VARIANCE AND STANDARD DEVIATION
# ──────────────────────────────────────────────────────────────────────
print("\n" + "─" * 70)
print("3.2 VARIANCE AND STANDARD DEVIATION")
print("─" * 70)

# Var(X) = E[(X - E[X])²] = E[X²] - (E[X])²
# Std(X) = sqrt(Var(X))
#
# Variance measures SPREAD. High variance = data is spread out.

data = np.array([2, 4, 4, 4, 5, 5, 7, 9])

# Manual variance calculation
mean = data.mean()
variance_manual = np.mean((data - mean) ** 2)  # population variance
variance_numpy = np.var(data)                    # same thing

# Sample variance (ddof=1): used when estimating from a sample
# Divides by (n-1) instead of n to correct for bias (Bessel's correction)
variance_sample = np.var(data, ddof=1)

print(f"Data: {data}")
print(f"Mean: {mean}")
print(f"Population variance: {variance_manual:.4f}")
print(f"Sample variance (ddof=1): {variance_sample:.4f}")
print(f"Standard deviation: {np.std(data):.4f}")
print(f"Sample std (ddof=1): {np.std(data, ddof=1):.4f}")

# Properties of variance
# Var(aX + b) = a² * Var(X)  (adding constant doesn't change spread!)
X = np.random.randn(10000)
a, b = 3, 100
Y = a * X + b
print(f"\nVar({a}X + {b}) = {a}² * Var(X)")
print(f"  Var(X) = {np.var(X):.4f}")
print(f"  Var({a}X + {b}) = {np.var(Y):.4f}")
print(f"  {a}² * Var(X) = {a**2 * np.var(X):.4f}")
print(f"  Adding {b} doesn't change the variance!")

# Comparing variance of different distributions
print(f"\n--- Variance Comparison ---")
distributions = {
    'Uniform[0,1]': np.random.uniform(0, 1, 10000),
    'Normal(0,1)': np.random.randn(10000),
    'Normal(0,3)': np.random.randn(10000) * 3,
    'Exponential(1)': np.random.exponential(1, 10000),
}

for name, samples in distributions.items():
    print(f"  {name:>20s}: mean={samples.mean():>7.3f}, var={samples.var():>7.3f}, "
          f"std={samples.std():>7.3f}")

# ──────────────────────────────────────────────────────────────────────
# 3.3 COVARIANCE AND CORRELATION
# ──────────────────────────────────────────────────────────────────────
print("\n" + "─" * 70)
print("3.3 COVARIANCE AND CORRELATION")
print("─" * 70)

# Covariance: measures how two variables move together
# Cov(X, Y) = E[(X - E[X])(Y - E[Y])]
#   > 0: X and Y tend to increase together
#   < 0: when X increases, Y tends to decrease
#   = 0: no linear relationship

# Correlation: normalized covariance (scale-independent)
# Corr(X, Y) = Cov(X, Y) / (Std(X) * Std(Y))
# Range: [-1, 1]

np.random.seed(42)
n = 1000

# Create correlated variables
x = np.random.randn(n)
y_positive = 2 * x + np.random.randn(n) * 0.5   # strong positive
y_negative = -1.5 * x + np.random.randn(n) * 0.5 # strong negative
y_none = np.random.randn(n)                        # no correlation
y_nonlinear = x ** 2 + np.random.randn(n) * 0.3   # nonlinear (corr ≈ 0!)

# Covariance (manual)
def covariance(x, y):
    return np.mean((x - x.mean()) * (y - y.mean()))

def correlation(x, y):
    return covariance(x, y) / (x.std() * y.std())

print(f"--- Covariance ---")
print(f"Cov(X, Y_positive)  = {covariance(x, y_positive):.4f}")
print(f"Cov(X, Y_negative)  = {covariance(x, y_negative):.4f}")
print(f"Cov(X, Y_none)      = {covariance(x, y_none):.4f}")
print(f"Cov(X, Y_nonlinear) = {covariance(x, y_nonlinear):.4f}")

print(f"\n--- Correlation (Pearson) ---")
print(f"Corr(X, Y_positive)  = {correlation(x, y_positive):.4f}")
print(f"Corr(X, Y_negative)  = {correlation(x, y_negative):.4f}")
print(f"Corr(X, Y_none)      = {correlation(x, y_none):.4f}")
print(f"Corr(X, Y_nonlinear) = {correlation(x, y_nonlinear):.4f}")
print(f"  (Nonlinear has ~0 correlation but clear relationship!)")

# NumPy covariance matrix
data_matrix = np.column_stack([x, y_positive, y_negative, y_none])
cov_matrix = np.cov(data_matrix, rowvar=False)
corr_matrix = np.corrcoef(data_matrix, rowvar=False)

labels = ['X', 'Y_pos', 'Y_neg', 'Y_none']
print(f"\n--- Correlation Matrix ---")
print(f"{'':>10}", end='')
for l in labels:
    print(f"{l:>10}", end='')
print()
for i, row_label in enumerate(labels):
    print(f"{row_label:>10}", end='')
    for j in range(len(labels)):
        print(f"{corr_matrix[i, j]:>10.4f}", end='')
    print()

# Visualize correlations
fig, axes = plt.subplots(1, 4, figsize=(20, 5))
pairs = [
    (y_positive, f'Positive (r={correlation(x, y_positive):.2f})'),
    (y_negative, f'Negative (r={correlation(x, y_negative):.2f})'),
    (y_none, f'None (r={correlation(x, y_none):.2f})'),
    (y_nonlinear, f'Nonlinear (r={correlation(x, y_nonlinear):.2f})'),
]

for ax, (y_data, title) in zip(axes, pairs):
    ax.scatter(x, y_data, alpha=0.3, s=10)
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.grid(True, alpha=0.3)

axes[0].set_ylabel('Y')
plt.suptitle('Correlation Types', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('probability_for_ml/03_correlation_types.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"\nCorrelation types plot saved to: 03_correlation_types.png")

# ML Application: Feature Correlation Analysis
print(f"\n--- ML Application: Feature Correlation for Feature Selection ---")
from sklearn.datasets import load_wine

wine = load_wine()
X_wine = wine.data
feature_names = wine.feature_names

corr_wine = np.corrcoef(X_wine, rowvar=False)

# Find highly correlated feature pairs (potential redundancy)
threshold = 0.8
print(f"Feature pairs with |correlation| > {threshold}:")
for i in range(len(feature_names)):
    for j in range(i + 1, len(feature_names)):
        if abs(corr_wine[i, j]) > threshold:
            print(f"  {feature_names[i]:>25s} <-> {feature_names[j]:<25s}: "
                  f"r = {corr_wine[i, j]:+.3f}")

# ──────────────────────────────────────────────────────────────────────
# 3.4 LAW OF LARGE NUMBERS
# ──────────────────────────────────────────────────────────────────────
print("\n" + "─" * 70)
print("3.4 LAW OF LARGE NUMBERS")
print("─" * 70)

# As sample size n → ∞, the sample mean → population mean.
# This is WHY ML works: more data → better estimates.

np.random.seed(42)
true_mean = 5.0
true_std = 2.0

sample_sizes = [10, 50, 100, 500, 1000, 5000, 10000, 50000]
print(f"True population mean: {true_mean}")
print(f"\nSample means converging to true mean:")

running_means = []
max_n = 50000
all_samples = np.random.normal(true_mean, true_std, max_n)

for n in sample_sizes:
    sample_mean = all_samples[:n].mean()
    error = abs(sample_mean - true_mean)
    running_means.append(sample_mean)
    print(f"  n = {n:>6d}: mean = {sample_mean:.4f}, |error| = {error:.4f}")

# Visualize convergence
fig, ax = plt.subplots(figsize=(10, 5))
cumulative_means = np.cumsum(all_samples) / np.arange(1, max_n + 1)
ax.plot(cumulative_means, alpha=0.7)
ax.axhline(y=true_mean, color='red', linestyle='--', label=f'True mean = {true_mean}')
ax.set_xlabel('Number of Samples')
ax.set_ylabel('Running Mean')
ax.set_title('Law of Large Numbers: Running Mean Converges to True Mean')
ax.set_xscale('log')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('probability_for_ml/03_law_of_large_numbers.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"\nLLN plot saved to: 03_law_of_large_numbers.png")

# ──────────────────────────────────────────────────────────────────────
# 3.5 CENTRAL LIMIT THEOREM (CLT)
# ──────────────────────────────────────────────────────────────────────
print("\n" + "─" * 70)
print("3.5 CENTRAL LIMIT THEOREM (CLT)")
print("─" * 70)

# THE most important theorem in statistics:
# The mean of n independent samples from ANY distribution
# converges to a Normal distribution as n grows.
#
# This is why:
# - Gaussian assumption works so often in ML
# - Confidence intervals work
# - Statistical tests are valid

# Demonstrate with a VERY non-normal distribution: Exponential
np.random.seed(42)
n_experiments = 10000

fig, axes = plt.subplots(2, 4, figsize=(20, 10))

print("CLT with Exponential(λ=1) -- very skewed distribution:")
print(f"True mean = 1.0, True variance = 1.0\n")

for idx, sample_size in enumerate([1, 2, 5, 10, 30, 50, 100, 500]):
    row, col = idx // 4, idx % 4
    ax = axes[row, col]

    # Draw n_experiments sets of sample_size samples, take their means
    sample_means = np.array([
        np.random.exponential(1, sample_size).mean()
        for _ in range(n_experiments)
    ])

    ax.hist(sample_means, bins=50, density=True, alpha=0.7, edgecolor='black', linewidth=0.5)

    # Overlay theoretical normal (by CLT)
    x = np.linspace(sample_means.min(), sample_means.max(), 100)
    clt_std = 1.0 / np.sqrt(sample_size)  # std of mean = σ/√n
    ax.plot(x, stats.norm.pdf(x, 1.0, clt_std), 'r-', linewidth=2)

    ax.set_title(f'n = {sample_size}')
    ax.set_xlim(0, 3 if sample_size < 10 else 2)

    if idx < 4:
        skewness = stats.skew(sample_means)
        print(f"  n={sample_size:>3d}: mean of means={sample_means.mean():.4f}, "
              f"std={sample_means.std():.4f} (expected {clt_std:.4f}), "
              f"skewness={skewness:.3f}")
    else:
        skewness = stats.skew(sample_means)
        print(f"  n={sample_size:>3d}: mean of means={sample_means.mean():.4f}, "
              f"std={sample_means.std():.4f} (expected {clt_std:.4f}), "
              f"skewness={skewness:.3f}")

plt.suptitle('Central Limit Theorem: Sample Means → Normal\n'
             '(Red = theoretical Normal; Blue = actual distribution of means)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('probability_for_ml/03_central_limit_theorem.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"\nCLT demonstration saved to: 03_central_limit_theorem.png")
print(f"Notice: as n grows, skewness → 0 (becomes symmetric/Gaussian)")

print("\n" + "=" * 70)
print("PART 3 COMPLETE -- You now understand statistical measures!")
print("Key takeaways:")
print("  - E[X]: average outcome, central to all predictions")
print("  - Var(X): measures uncertainty/spread")
print("  - Correlation: detects linear relationships between features")
print("  - Correlation ≈ 0 does NOT mean independence (nonlinear!)")
print("  - LLN: more data → better estimates")
print("  - CLT: sample means are always ~Normal → Gaussian assumption works")
print("=" * 70)
