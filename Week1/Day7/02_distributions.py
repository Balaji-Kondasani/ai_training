"""
=============================================================================
PART 2: RANDOM VARIABLES & PROBABILITY DISTRIBUTIONS
=============================================================================

Every ML model either assumes a distribution or learns one.
Understanding distributions is how you pick the right model and loss function.

Topics covered:
  2.1 Discrete Distributions (Bernoulli, Binomial, Poisson, Categorical)
  2.2 Continuous Distributions (Uniform, Gaussian/Normal, Exponential, Beta)
  2.3 The Gaussian (Normal) Distribution -- Deep Dive
  2.4 Multivariate Gaussian
"""

import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

print("=" * 70)
print("PART 2: RANDOM VARIABLES & PROBABILITY DISTRIBUTIONS")
print("=" * 70)

# ──────────────────────────────────────────────────────────────────────
# 2.1 DISCRETE DISTRIBUTIONS
# ──────────────────────────────────────────────────────────────────────
print("\n" + "─" * 70)
print("2.1 DISCRETE DISTRIBUTIONS")
print("─" * 70)

# ─── Bernoulli Distribution ─────────────────────────────────────────
# A single coin flip: success (1) or failure (0).
# Parameter: p = probability of success
# ML use: binary classification output

print("\n--- Bernoulli Distribution ---")
p = 0.7  # probability of success
n_trials = 10000
samples = np.random.binomial(1, p, n_trials)  # Bernoulli = Binomial with n=1

print(f"P(success) = {p}")
print(f"Simulated mean (should ≈ {p}): {samples.mean():.4f}")
print(f"Simulated var (should ≈ {p*(1-p):.4f}): {samples.var():.4f}")

# ─── Binomial Distribution ──────────────────────────────────────────
# Number of successes in n independent Bernoulli trials.
# Parameters: n (trials), p (success probability)
# ML use: number of correct predictions in n samples

print("\n--- Binomial Distribution ---")
n, p = 20, 0.6
samples_binom = np.random.binomial(n, p, 10000)

print(f"n={n} trials, p={p}")
print(f"Mean (should ≈ {n*p:.1f}): {samples_binom.mean():.2f}")
print(f"Variance (should ≈ {n*p*(1-p):.2f}): {samples_binom.var():.2f}")

# P(X = k) using scipy
for k in [10, 12, 15]:
    prob = stats.binom.pmf(k, n, p)
    print(f"P(X = {k}) = {prob:.4f}")

# P(X >= 15) using survival function
print(f"P(X >= 15) = {stats.binom.sf(14, n, p):.4f}")

# ─── Poisson Distribution ───────────────────────────────────────────
# Number of events in a fixed interval (time/space).
# Parameter: λ (lambda) = average rate of events
# ML use: count data, rare events, text (word counts)

print("\n--- Poisson Distribution ---")
lam = 5  # average 5 events per interval
samples_poisson = np.random.poisson(lam, 10000)

print(f"λ = {lam}")
print(f"Mean (should ≈ {lam}): {samples_poisson.mean():.2f}")
print(f"Variance (should ≈ {lam}): {samples_poisson.var():.2f}")
print(f"(Poisson: mean = variance = λ)")

# ─── Categorical Distribution ───────────────────────────────────────
# Generalization of Bernoulli to k outcomes.
# ML use: multi-class classification output (softmax gives these probs)

print("\n--- Categorical Distribution ---")
categories = ['cat', 'dog', 'bird']
probabilities = [0.5, 0.3, 0.2]  # must sum to 1

samples_cat = np.random.choice(categories, size=10000, p=probabilities)
print(f"Categories: {categories}")
print(f"True probs: {probabilities}")
print(f"Simulated frequencies:")
for cat, prob in zip(categories, probabilities):
    freq = np.mean(samples_cat == cat)
    print(f"  {cat}: {freq:.3f} (expected {prob})")

# ──────────────────────────────────────────────────────────────────────
# 2.2 CONTINUOUS DISTRIBUTIONS
# ──────────────────────────────────────────────────────────────────────
print("\n" + "─" * 70)
print("2.2 CONTINUOUS DISTRIBUTIONS")
print("─" * 70)

# ─── Uniform Distribution ───────────────────────────────────────────
# All values equally likely in [a, b].
# ML use: random initialization, random search hyperparameters

print("\n--- Uniform Distribution ---")
a, b = 2, 8
samples_uniform = np.random.uniform(a, b, 10000)
print(f"Uniform[{a}, {b}]")
print(f"Mean (should ≈ {(a+b)/2}): {samples_uniform.mean():.2f}")
print(f"Variance (should ≈ {(b-a)**2/12:.2f}): {samples_uniform.var():.2f}")

# ─── Gaussian (Normal) Distribution ─────────────────────────────────
# The most important distribution in all of ML!
# Parameters: μ (mean), σ (standard deviation)
# PDF: f(x) = (1/√(2πσ²)) * exp(-(x-μ)²/(2σ²))

print("\n--- Gaussian (Normal) Distribution ---")
mu, sigma = 5, 2
samples_normal = np.random.normal(mu, sigma, 10000)

print(f"Normal(μ={mu}, σ={sigma})")
print(f"Mean: {samples_normal.mean():.3f}")
print(f"Std:  {samples_normal.std():.3f}")

# The 68-95-99.7 rule (empirical rule)
within_1_sigma = np.mean(np.abs(samples_normal - mu) < 1 * sigma)
within_2_sigma = np.mean(np.abs(samples_normal - mu) < 2 * sigma)
within_3_sigma = np.mean(np.abs(samples_normal - mu) < 3 * sigma)

print(f"\nEmpirical Rule (68-95-99.7):")
print(f"  Within 1σ: {within_1_sigma:.3f} (expected ≈ 0.683)")
print(f"  Within 2σ: {within_2_sigma:.3f} (expected ≈ 0.954)")
print(f"  Within 3σ: {within_3_sigma:.3f} (expected ≈ 0.997)")

# Standard Normal (Z-distribution): μ=0, σ=1
# Any normal can be standardized: Z = (X - μ) / σ
z_scores = (samples_normal - mu) / sigma
print(f"\nStandardized: mean={z_scores.mean():.4f}, std={z_scores.std():.4f}")

# ─── Exponential Distribution ───────────────────────────────────────
# Time between events in a Poisson process.
# Parameter: λ (rate) or 1/λ (scale = mean time between events)
# ML use: survival analysis, time-to-event modeling

print("\n--- Exponential Distribution ---")
lam_exp = 2.0  # rate parameter
samples_exp = np.random.exponential(1/lam_exp, 10000)  # NumPy uses scale=1/λ

print(f"Exponential(λ={lam_exp})")
print(f"Mean (should ≈ {1/lam_exp:.2f}): {samples_exp.mean():.3f}")
print(f"Variance (should ≈ {1/lam_exp**2:.2f}): {samples_exp.var():.3f}")

# ─── Beta Distribution ──────────────────────────────────────────────
# Values in [0, 1] -- perfect for modeling probabilities!
# Parameters: α (alpha), β (beta)
# ML use: Bayesian priors for probabilities, A/B testing

print("\n--- Beta Distribution ---")
configs = [(1, 1, "Uniform prior (no info)"),
           (5, 5, "Symmetric, peaked at 0.5"),
           (2, 8, "Skewed toward 0 (rare events)"),
           (8, 2, "Skewed toward 1 (common events)")]

for alpha, beta, desc in configs:
    samples_beta = np.random.beta(alpha, beta, 10000)
    mean_expected = alpha / (alpha + beta)
    print(f"  Beta(α={alpha}, β={beta}): mean={samples_beta.mean():.3f} "
          f"(expected {mean_expected:.3f}) -- {desc}")

# ─── Visualize All Distributions ────────────────────────────────────

fig, axes = plt.subplots(2, 4, figsize=(20, 10))

# Bernoulli
ax = axes[0, 0]
ax.bar([0, 1], [1-0.7, 0.7], color=['#FF6B6B', '#4ECDC4'], edgecolor='black')
ax.set_title('Bernoulli (p=0.7)')
ax.set_xticks([0, 1])
ax.set_xticklabels(['Fail', 'Success'])

# Binomial
ax = axes[0, 1]
x_binom = np.arange(0, 21)
ax.bar(x_binom, stats.binom.pmf(x_binom, 20, 0.6), color='#45B7D1', edgecolor='black')
ax.set_title('Binomial (n=20, p=0.6)')

# Poisson
ax = axes[0, 2]
x_pois = np.arange(0, 20)
for lam_val in [2, 5, 10]:
    ax.plot(x_pois, stats.poisson.pmf(x_pois, lam_val), 'o-', label=f'λ={lam_val}')
ax.set_title('Poisson')
ax.legend()

# Categorical
ax = axes[0, 3]
ax.bar(categories, probabilities, color=['#FF6B6B', '#4ECDC4', '#45B7D1'], edgecolor='black')
ax.set_title('Categorical')

# Uniform
ax = axes[1, 0]
x_uni = np.linspace(0, 10, 200)
ax.plot(x_uni, stats.uniform.pdf(x_uni, 2, 6), 'b-', linewidth=2)
ax.fill_between(x_uni, stats.uniform.pdf(x_uni, 2, 6), alpha=0.3)
ax.set_title('Uniform [2, 8]')

# Gaussian
ax = axes[1, 1]
x_norm = np.linspace(-5, 15, 200)
for mu_val, sigma_val in [(0, 1), (5, 1), (5, 2)]:
    ax.plot(x_norm, stats.norm.pdf(x_norm, mu_val, sigma_val),
            label=f'μ={mu_val}, σ={sigma_val}')
ax.set_title('Gaussian (Normal)')
ax.legend(fontsize=8)

# Exponential
ax = axes[1, 2]
x_exp = np.linspace(0, 5, 200)
for lam_val in [0.5, 1, 2]:
    ax.plot(x_exp, stats.expon.pdf(x_exp, scale=1/lam_val), label=f'λ={lam_val}')
ax.set_title('Exponential')
ax.legend()

# Beta
ax = axes[1, 3]
x_beta = np.linspace(0, 1, 200)
for a_val, b_val in [(1, 1), (5, 5), (2, 8), (8, 2)]:
    ax.plot(x_beta, stats.beta.pdf(x_beta, a_val, b_val), label=f'α={a_val}, β={b_val}')
ax.set_title('Beta')
ax.legend(fontsize=8)

for ax in axes.flatten():
    ax.grid(True, alpha=0.3)

plt.suptitle('Common Probability Distributions', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('probability_for_ml/02_distributions_overview.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"\nDistributions overview saved to: 02_distributions_overview.png")

# ──────────────────────────────────────────────────────────────────────
# 2.3 THE GAUSSIAN DISTRIBUTION -- DEEP DIVE
# ──────────────────────────────────────────────────────────────────────
print("\n" + "─" * 70)
print("2.3 THE GAUSSIAN DISTRIBUTION -- DEEP DIVE")
print("─" * 70)

# Why Gaussian is everywhere in ML:
# 1. Central Limit Theorem: averages of anything → Gaussian
# 2. Maximum entropy distribution for given mean and variance
# 3. Many natural phenomena are approximately Gaussian
# 4. Makes math tractable (conjugate prior, closed-form solutions)

# Implementing the Gaussian PDF from scratch
def gaussian_pdf(x, mu, sigma):
    """Compute Gaussian probability density function."""
    coefficient = 1.0 / (sigma * np.sqrt(2 * np.pi))
    exponent = -0.5 * ((x - mu) / sigma) ** 2
    return coefficient * np.exp(exponent)

x = np.linspace(-4, 4, 100)
pdf_manual = gaussian_pdf(x, 0, 1)
pdf_scipy = stats.norm.pdf(x, 0, 1)

print(f"Our Gaussian PDF matches scipy: {np.allclose(pdf_manual, pdf_scipy)}")

# CDF: P(X <= x) -- "what fraction of data falls below x?"
print(f"\nStandard Normal CDF values:")
for z in [-2, -1, 0, 1, 2]:
    print(f"  P(Z <= {z:+d}) = {stats.norm.cdf(z):.4f}")

# Quantiles (inverse CDF) -- used for confidence intervals
print(f"\nQuantiles (inverse CDF):")
for q in [0.025, 0.05, 0.5, 0.95, 0.975]:
    print(f"  {q:.1%} quantile: z = {stats.norm.ppf(q):.4f}")

print(f"\n95% confidence interval: [{stats.norm.ppf(0.025):.3f}, {stats.norm.ppf(0.975):.3f}]")

# ──────────────────────────────────────────────────────────────────────
# 2.4 MULTIVARIATE GAUSSIAN
# ──────────────────────────────────────────────────────────────────────
print("\n" + "─" * 70)
print("2.4 MULTIVARIATE GAUSSIAN")
print("─" * 70)

# The multivariate Gaussian is defined by:
# - Mean vector μ (center of the distribution)
# - Covariance matrix Σ (shape/spread of the distribution)
#
# ML uses: Gaussian Mixture Models, LDA, Kalman filters, GPs

mu = np.array([1, 2])
cov = np.array([[2.0, 0.8],
                [0.8, 1.0]])

# Sample from multivariate Gaussian
np.random.seed(42)
samples_2d = np.random.multivariate_normal(mu, cov, 1000)

print(f"Mean vector μ: {mu}")
print(f"Covariance matrix Σ:\n{cov}")
print(f"\nSampled mean: {samples_2d.mean(axis=0).round(3)}")
print(f"Sampled covariance:\n{np.cov(samples_2d, rowvar=False).round(3)}")

# Compute PDF at a point
point = np.array([1, 2])
pdf_value = stats.multivariate_normal.pdf(point, mu, cov)
print(f"\nPDF at {point}: {pdf_value:.6f}")

# Mahalanobis distance: "how far is a point from the distribution?"
# Accounts for the covariance structure (unlike Euclidean distance)
from scipy.spatial.distance import mahalanobis

cov_inv = np.linalg.inv(cov)
point1 = np.array([1, 2])  # at the mean
point2 = np.array([4, 4])  # far from mean

d1 = mahalanobis(point1, mu, cov_inv)
d2 = mahalanobis(point2, mu, cov_inv)
print(f"\nMahalanobis distance:")
print(f"  Point {point1} (at mean): {d1:.4f}")
print(f"  Point {point2} (far away): {d2:.4f}")

# Visualize multivariate Gaussian
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Scatter plot
axes[0].scatter(samples_2d[:, 0], samples_2d[:, 1], alpha=0.3, s=10)
axes[0].plot(mu[0], mu[1], 'r+', markersize=20, markeredgewidth=3)
axes[0].set_title('Samples from 2D Gaussian')
axes[0].set_xlabel('X1')
axes[0].set_ylabel('X2')
axes[0].set_aspect('equal')
axes[0].grid(True, alpha=0.3)

# Different covariance structures
configs = [
    (np.array([[1, 0], [0, 1]]), "Identity (uncorrelated)"),
    (np.array([[2, 1.5], [1.5, 2]]), "Positive correlation"),
    (np.array([[2, -1.5], [-1.5, 2]]), "Negative correlation"),
]

for idx, (cov_i, title) in enumerate(configs):
    samples_i = np.random.multivariate_normal([0, 0], cov_i, 500)
    axes[idx].scatter(samples_i[:, 0], samples_i[:, 1], alpha=0.4, s=10)
    axes[idx].set_title(title)
    axes[idx].set_xlim(-5, 5)
    axes[idx].set_ylim(-5, 5)
    axes[idx].set_aspect('equal')
    axes[idx].grid(True, alpha=0.3)

plt.suptitle('Multivariate Gaussian -- Effect of Covariance', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('probability_for_ml/02_multivariate_gaussian.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"\nMultivariate Gaussian plot saved to: 02_multivariate_gaussian.png")

print("\n" + "=" * 70)
print("PART 2 COMPLETE -- You now understand probability distributions!")
print("Key takeaways:")
print("  - Bernoulli/Binomial → binary/count classification outcomes")
print("  - Poisson → count data, rare events")
print("  - Gaussian → most common assumption in ML (CLT, tractability)")
print("  - Beta → Bayesian priors for probabilities")
print("  - Multivariate Gaussian → covariance captures feature relationships")
print("=" * 70)
