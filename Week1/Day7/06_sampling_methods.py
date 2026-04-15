"""
=============================================================================
PART 6: SAMPLING, MONTE CARLO & BOOTSTRAP METHODS
=============================================================================

Sampling methods are essential for:
- Estimating quantities that can't be computed analytically
- Computing confidence intervals
- Hypothesis testing
- Training ML models (SGD uses random sampling!)
- Bayesian inference (MCMC)

Topics covered:
  6.1 Random Sampling & Simulation
  6.2 Monte Carlo Estimation
  6.3 Bootstrap Method (Confidence Intervals)
  6.4 Permutation Tests (Hypothesis Testing)
  6.5 Markov Chain Monte Carlo (MCMC) -- Intuition
"""

import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

print("=" * 70)
print("PART 6: SAMPLING, MONTE CARLO & BOOTSTRAP METHODS")
print("=" * 70)

# ──────────────────────────────────────────────────────────────────────
# 6.1 RANDOM SAMPLING & SIMULATION
# ──────────────────────────────────────────────────────────────────────
print("\n" + "─" * 70)
print("6.1 RANDOM SAMPLING & SIMULATION")
print("─" * 70)

np.random.seed(42)

# Basic sampling functions
print("--- NumPy Sampling Functions ---")

# Sampling from a distribution
uniform_samples = np.random.uniform(0, 10, 5)
normal_samples = np.random.normal(5, 2, 5)
choice_samples = np.random.choice(['A', 'B', 'C'], size=5, p=[0.5, 0.3, 0.2])

print(f"Uniform[0,10]:  {uniform_samples.round(2)}")
print(f"Normal(5,2):    {normal_samples.round(2)}")
print(f"Choice(A/B/C):  {choice_samples}")

# Sampling with and without replacement
data = np.array([10, 20, 30, 40, 50])
with_replacement = np.random.choice(data, size=8, replace=True)
without_replacement = np.random.choice(data, size=3, replace=False)

print(f"\nData: {data}")
print(f"With replacement (n=8):    {with_replacement}")
print(f"Without replacement (n=3): {without_replacement}")

# Shuffle and permutation
shuffled = np.random.permutation(data)
print(f"Shuffled: {shuffled}")

# Reproducibility with seeds
print(f"\n--- Reproducibility ---")
np.random.seed(123)
a = np.random.randn(3)
np.random.seed(123)
b = np.random.randn(3)
print(f"Same seed → same numbers: {np.array_equal(a, b)}")

# Train/test split from scratch using sampling
print(f"\n--- Train/Test Split from Scratch ---")
n = 100
X = np.random.randn(n, 3)
y = np.random.randint(0, 2, n)

indices = np.random.permutation(n)
split = int(0.8 * n)
train_idx, test_idx = indices[:split], indices[split:]

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]
print(f"Total: {n}, Train: {len(train_idx)}, Test: {len(test_idx)}")

# Stratified sampling (preserving class proportions)
print(f"\n--- Stratified Sampling ---")
y_full = np.array([0]*80 + [1]*20)  # imbalanced: 80% class 0, 20% class 1

# Simple random split might give bad proportions
random_idx = np.random.permutation(len(y_full))[:20]
print(f"Random sample class balance: {np.mean(y_full[random_idx] == 1):.1%} class 1")

# Stratified: sample proportionally from each class
strat_idx = []
for cls in [0, 1]:
    cls_indices = np.where(y_full == cls)[0]
    n_sample = int(0.2 * len(cls_indices))
    strat_idx.extend(np.random.choice(cls_indices, n_sample, replace=False))

print(f"Stratified sample balance:  {np.mean(y_full[strat_idx] == 1):.1%} class 1")
print(f"Original balance:           {np.mean(y_full == 1):.1%} class 1")

# ──────────────────────────────────────────────────────────────────────
# 6.2 MONTE CARLO ESTIMATION
# ──────────────────────────────────────────────────────────────────────
print("\n" + "─" * 70)
print("6.2 MONTE CARLO ESTIMATION")
print("─" * 70)

# Monte Carlo: use random sampling to estimate quantities.
# Core idea: E[f(X)] ≈ (1/n) * Σ f(x_i) where x_i ~ some distribution

# Example 1: Estimate π using random points in a unit square
n_points = 100_000
x = np.random.uniform(0, 1, n_points)
y = np.random.uniform(0, 1, n_points)

# Point is inside quarter circle if x² + y² <= 1
inside_circle = (x**2 + y**2) <= 1
pi_estimate = 4 * np.mean(inside_circle)

print(f"--- Estimating π with Monte Carlo ---")
print(f"Points: {n_points:,}")
print(f"Estimated π: {pi_estimate:.6f}")
print(f"True π:      {np.pi:.6f}")
print(f"Error:       {abs(pi_estimate - np.pi):.6f}")

# Example 2: Estimate an integral using Monte Carlo
# Integral of sin(x) from 0 to π = 2.0
n_samples = 100_000
x_samples = np.random.uniform(0, np.pi, n_samples)
integral_estimate = np.pi * np.mean(np.sin(x_samples))

print(f"\n--- Estimating ∫₀^π sin(x)dx ---")
print(f"Monte Carlo estimate: {integral_estimate:.6f}")
print(f"True value:           {2.0:.6f}")

# Example 3: Monte Carlo for expected value of complex function
# E[e^(-X²)] where X ~ N(0, 1)
x_normal = np.random.randn(100_000)
expected_value = np.mean(np.exp(-x_normal**2))
true_value = np.sqrt(np.pi / 2) / np.sqrt(np.pi)  # analytical: 1/sqrt(2) ≈ 0.7071

print(f"\n--- E[exp(-X²)] where X ~ N(0,1) ---")
print(f"Monte Carlo: {expected_value:.6f}")
print(f"Analytical:  {1/np.sqrt(2):.6f}")

# Convergence: estimate improves with more samples
print(f"\n--- Convergence Rate ---")
for n in [100, 1000, 10000, 100000, 1000000]:
    x = np.random.uniform(0, 1, n)
    y = np.random.uniform(0, 1, n)
    pi_est = 4 * np.mean((x**2 + y**2) <= 1)
    error = abs(pi_est - np.pi)
    print(f"  n={n:>9,}: π ≈ {pi_est:.5f}, error = {error:.5f}")

# ──────────────────────────────────────────────────────────────────────
# 6.3 BOOTSTRAP METHOD
# ──────────────────────────────────────────────────────────────────────
print("\n" + "─" * 70)
print("6.3 BOOTSTRAP METHOD (Confidence Intervals)")
print("─" * 70)

# Bootstrap: resample WITH replacement from your data to estimate
# the sampling distribution of any statistic.
# This gives you confidence intervals without assumptions!

np.random.seed(42)
# Simulate some real data: response times (in ms)
response_times = np.concatenate([
    np.random.exponential(200, 80),    # normal users
    np.random.exponential(800, 20),    # slow connections
])
np.random.shuffle(response_times)

print(f"Sample data: {len(response_times)} response times")
print(f"Sample mean: {response_times.mean():.1f} ms")
print(f"Sample median: {np.median(response_times):.1f} ms")

def bootstrap_ci(data, statistic_func, n_bootstrap=10000, ci=95):
    """Compute bootstrap confidence interval for any statistic."""
    n = len(data)
    bootstrap_stats = np.zeros(n_bootstrap)

    for i in range(n_bootstrap):
        # Resample with replacement
        sample = data[np.random.randint(0, n, size=n)]
        bootstrap_stats[i] = statistic_func(sample)

    # Compute percentiles for confidence interval
    lower = np.percentile(bootstrap_stats, (100 - ci) / 2)
    upper = np.percentile(bootstrap_stats, 100 - (100 - ci) / 2)

    return bootstrap_stats, lower, upper

# Bootstrap CI for the mean
boot_means, lower_mean, upper_mean = bootstrap_ci(response_times, np.mean)
print(f"\n95% CI for mean: [{lower_mean:.1f}, {upper_mean:.1f}] ms")
print(f"Bootstrap SE: {boot_means.std():.1f} ms")

# Bootstrap CI for the median
boot_medians, lower_med, upper_med = bootstrap_ci(response_times, np.median)
print(f"95% CI for median: [{lower_med:.1f}, {upper_med:.1f}] ms")

# Bootstrap CI for 90th percentile
stat_90 = lambda x: np.percentile(x, 90)
boot_p90, lower_p90, upper_p90 = bootstrap_ci(response_times, stat_90)
print(f"95% CI for 90th percentile: [{lower_p90:.1f}, {upper_p90:.1f}] ms")

# Visualize bootstrap distributions
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, data, stat_name, lower, upper in [
    (axes[0], boot_means, 'Mean', lower_mean, upper_mean),
    (axes[1], boot_medians, 'Median', lower_med, upper_med),
    (axes[2], boot_p90, '90th Percentile', lower_p90, upper_p90),
]:
    ax.hist(data, bins=50, density=True, alpha=0.7, edgecolor='black', linewidth=0.5)
    ax.axvline(lower, color='red', linestyle='--', label=f'95% CI: [{lower:.0f}, {upper:.0f}]')
    ax.axvline(upper, color='red', linestyle='--')
    ax.set_title(f'Bootstrap Distribution of {stat_name}')
    ax.set_xlabel(f'{stat_name} (ms)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('probability_for_ml/06_bootstrap_distributions.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"\nBootstrap visualization saved to: 06_bootstrap_distributions.png")

# ──────────────────────────────────────────────────────────────────────
# 6.4 PERMUTATION TESTS (Hypothesis Testing)
# ──────────────────────────────────────────────────────────────────────
print("\n" + "─" * 70)
print("6.4 PERMUTATION TESTS")
print("─" * 70)

# Permutation test: test if two groups are truly different by
# randomly shuffling group labels many times and seeing if the
# observed difference is unusually large.

np.random.seed(42)

# Example: A/B test -- does a new website design increase time on page?
group_a = np.random.normal(120, 30, 50)   # old design: ~120 sec
group_b = np.random.normal(135, 30, 50)   # new design: ~135 sec

observed_diff = group_b.mean() - group_a.mean()
print(f"Group A (old) mean: {group_a.mean():.1f} sec (n={len(group_a)})")
print(f"Group B (new) mean: {group_b.mean():.1f} sec (n={len(group_b)})")
print(f"Observed difference: {observed_diff:.1f} sec")

# Permutation test
n_permutations = 10000
combined = np.concatenate([group_a, group_b])
n_a = len(group_a)

perm_diffs = np.zeros(n_permutations)
for i in range(n_permutations):
    shuffled = np.random.permutation(combined)
    perm_diffs[i] = shuffled[n_a:].mean() - shuffled[:n_a].mean()

# p-value: proportion of permutation differences >= observed
p_value = np.mean(perm_diffs >= observed_diff)

print(f"\nPermutation test ({n_permutations:,} permutations):")
print(f"  p-value = {p_value:.4f}")
print(f"  Significant at α=0.05? {'Yes' if p_value < 0.05 else 'No'}")

# Compare with t-test
t_stat, t_pvalue = stats.ttest_ind(group_b, group_a)
print(f"\nTraditional t-test:")
print(f"  t-statistic = {t_stat:.4f}")
print(f"  p-value = {t_pvalue:.4f}")

# Visualize
fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(perm_diffs, bins=50, density=True, alpha=0.7, edgecolor='black', linewidth=0.5)
ax.axvline(observed_diff, color='red', linewidth=2,
           label=f'Observed diff = {observed_diff:.1f} (p={p_value:.3f})')
ax.set_xlabel('Difference in Means (B - A)')
ax.set_ylabel('Density')
ax.set_title('Permutation Test: Is the New Design Better?')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('probability_for_ml/06_permutation_test.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"\nPermutation test plot saved to: 06_permutation_test.png")

# ──────────────────────────────────────────────────────────────────────
# 6.5 MARKOV CHAIN MONTE CARLO (MCMC) -- INTUITION
# ──────────────────────────────────────────────────────────────────────
print("\n" + "─" * 70)
print("6.5 MCMC -- METROPOLIS-HASTINGS ALGORITHM")
print("─" * 70)

# MCMC: sample from a distribution when you can only evaluate the
# density (up to a constant). Essential for Bayesian inference.
#
# Metropolis-Hastings:
# 1. Start at some point
# 2. Propose a new point (random step)
# 3. Accept with probability min(1, p(new)/p(current))
# 4. Repeat

def metropolis_hastings(target_log_pdf, n_samples, initial, proposal_std=1.0):
    """Simple Metropolis-Hastings sampler for 1D distributions."""
    samples = np.zeros(n_samples)
    current = initial
    n_accepted = 0

    for i in range(n_samples):
        # Propose new point (symmetric random walk)
        proposed = current + np.random.normal(0, proposal_std)

        # Acceptance ratio (in log space for numerical stability)
        log_alpha = target_log_pdf(proposed) - target_log_pdf(current)

        # Accept or reject
        if np.log(np.random.random()) < log_alpha:
            current = proposed
            n_accepted += 1

        samples[i] = current

    acceptance_rate = n_accepted / n_samples
    return samples, acceptance_rate

# Target: a mixture of two Gaussians (hard to sample directly)
def target_log_pdf(x):
    """Log PDF of a mixture of two Gaussians."""
    p1 = 0.3 * stats.norm.pdf(x, -2, 0.8)
    p2 = 0.7 * stats.norm.pdf(x, 3, 1.2)
    return np.log(p1 + p2 + 1e-300)

# Run MCMC
n_samples = 50000
samples_mcmc, acceptance_rate = metropolis_hastings(
    target_log_pdf, n_samples, initial=0, proposal_std=2.0
)

# Discard burn-in (first 1000 samples)
burn_in = 1000
samples_clean = samples_mcmc[burn_in:]

print(f"MCMC samples: {n_samples}")
print(f"Acceptance rate: {acceptance_rate:.2%}")
print(f"After burn-in ({burn_in}): {len(samples_clean)} samples")
print(f"Sample mean: {samples_clean.mean():.3f}")

# Visualize
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Trace plot (should look like "hairy caterpillar")
axes[0].plot(samples_mcmc[:5000], alpha=0.5, linewidth=0.5)
axes[0].axvline(burn_in, color='red', linestyle='--', label='Burn-in')
axes[0].set_title('Trace Plot (first 5000)')
axes[0].set_xlabel('Iteration')
axes[0].set_ylabel('Value')
axes[0].legend()

# Histogram vs true density
x_range = np.linspace(-6, 8, 200)
true_density = 0.3 * stats.norm.pdf(x_range, -2, 0.8) + 0.7 * stats.norm.pdf(x_range, 3, 1.2)

axes[1].hist(samples_clean, bins=80, density=True, alpha=0.7, label='MCMC samples')
axes[1].plot(x_range, true_density, 'r-', linewidth=2, label='True density')
axes[1].set_title('MCMC vs True Distribution')
axes[1].legend()

# Autocorrelation (should decay quickly for good mixing)
from numpy.fft import fft, ifft
def autocorrelation(x, max_lag=100):
    x = x - x.mean()
    result = np.correlate(x[:max_lag*10], x[:max_lag*10], mode='full')
    result = result[len(result)//2:]
    return result[:max_lag] / result[0]

acf = autocorrelation(samples_clean)
axes[2].plot(acf)
axes[2].axhline(0, color='gray', linestyle='-')
axes[2].set_title('Autocorrelation')
axes[2].set_xlabel('Lag')
axes[2].set_ylabel('Autocorrelation')

for ax in axes:
    ax.grid(True, alpha=0.3)

plt.suptitle('Markov Chain Monte Carlo (Metropolis-Hastings)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('probability_for_ml/06_mcmc.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"\nMCMC visualization saved to: 06_mcmc.png")

print("\n" + "=" * 70)
print("PART 6 COMPLETE -- You now understand sampling methods!")
print("Key takeaways:")
print("  - Monte Carlo: estimate anything with random sampling")
print("  - Bootstrap: confidence intervals without assumptions")
print("  - Permutation test: non-parametric hypothesis testing")
print("  - MCMC: sample from complex distributions (Bayesian inference)")
print("  - More samples = better estimates (but diminishing returns)")
print("=" * 70)
