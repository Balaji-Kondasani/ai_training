"""
=============================================================================
PART 4: MAXIMUM LIKELIHOOD ESTIMATION (MLE) & MAP
=============================================================================

MLE is HOW machine learning models learn their parameters.
When you call model.fit(), it's usually doing MLE (or MAP) under the hood.

Topics covered:
  4.1 Maximum Likelihood Estimation (MLE) -- concept & implementation
  4.2 MLE for Gaussian Distribution
  4.3 MLE for Linear Regression (connection to MSE loss)
  4.4 MLE for Logistic Regression (connection to cross-entropy loss)
  4.5 MAP Estimation (MLE + regularization)
"""

import numpy as np
from scipy import stats
from scipy.optimize import minimize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

print("=" * 70)
print("PART 4: MAXIMUM LIKELIHOOD ESTIMATION (MLE) & MAP")
print("=" * 70)

# ──────────────────────────────────────────────────────────────────────
# 4.1 MLE -- CONCEPT AND IMPLEMENTATION
# ──────────────────────────────────────────────────────────────────────
print("\n" + "─" * 70)
print("4.1 MAXIMUM LIKELIHOOD ESTIMATION (MLE)")
print("─" * 70)

# MLE asks: "What parameter values make the observed data MOST LIKELY?"
#
# Given data D and model with parameter θ:
#   L(θ|D) = P(D|θ) = ∏ P(x_i|θ)   (likelihood)
#   θ_MLE = argmax L(θ|D)
#
# In practice we maximize LOG-likelihood (products → sums):
#   log L(θ|D) = Σ log P(x_i|θ)

# Example: MLE for a coin's bias (Bernoulli parameter)
np.random.seed(42)
true_p = 0.7
n_flips = 100
flips = np.random.binomial(1, true_p, n_flips)

n_heads = flips.sum()
n_tails = n_flips - n_heads

# MLE for Bernoulli: p_MLE = (number of heads) / (total flips)
p_mle = n_heads / n_flips

print(f"True coin bias: p = {true_p}")
print(f"Observed: {n_heads} heads out of {n_flips} flips")
print(f"MLE estimate: p̂ = {p_mle:.4f}")

# Visualize the likelihood function
p_values = np.linspace(0.01, 0.99, 200)
# Log-likelihood: n_heads*log(p) + n_tails*log(1-p)
log_likelihoods = n_heads * np.log(p_values) + n_tails * np.log(1 - p_values)

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(p_values, log_likelihoods, 'b-', linewidth=2)
ax.axvline(x=p_mle, color='red', linestyle='--', label=f'MLE: p̂ = {p_mle:.3f}')
ax.axvline(x=true_p, color='green', linestyle=':', label=f'True: p = {true_p}')
ax.set_xlabel('p (coin bias)')
ax.set_ylabel('Log-Likelihood')
ax.set_title('Log-Likelihood Function for Bernoulli Parameter')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('probability_for_ml/04_bernoulli_likelihood.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"\nLikelihood plot saved to: 04_bernoulli_likelihood.png")

# ──────────────────────────────────────────────────────────────────────
# 4.2 MLE FOR GAUSSIAN DISTRIBUTION
# ──────────────────────────────────────────────────────────────────────
print("\n" + "─" * 70)
print("4.2 MLE FOR GAUSSIAN DISTRIBUTION")
print("─" * 70)

# For Gaussian N(μ, σ²):
#   μ_MLE = (1/n) * Σ x_i  (sample mean)
#   σ²_MLE = (1/n) * Σ (x_i - μ_MLE)²  (biased sample variance)

np.random.seed(42)
true_mu = 5.0
true_sigma = 2.0
n_samples = 500

data = np.random.normal(true_mu, true_sigma, n_samples)

# MLE estimates
mu_mle = data.mean()
sigma_sq_mle = np.mean((data - mu_mle) ** 2)  # biased (divides by n)
sigma_mle = np.sqrt(sigma_sq_mle)

# Unbiased estimate (divides by n-1)
sigma_sq_unbiased = np.var(data, ddof=1)

print(f"True parameters: μ = {true_mu}, σ = {true_sigma}")
print(f"MLE estimates:   μ̂ = {mu_mle:.4f}, σ̂ = {sigma_mle:.4f}")
print(f"Unbiased σ̂:    {np.sqrt(sigma_sq_unbiased):.4f}")

# Numerical MLE using optimization
def neg_log_likelihood_gaussian(params, data):
    mu, log_sigma = params  # use log(sigma) to ensure sigma > 0
    sigma = np.exp(log_sigma)
    n = len(data)
    nll = n/2 * np.log(2 * np.pi * sigma**2) + np.sum((data - mu)**2) / (2 * sigma**2)
    return nll

result = minimize(neg_log_likelihood_gaussian,
                  x0=[0, 0],  # initial guess
                  args=(data,),
                  method='Nelder-Mead')

mu_opt, log_sigma_opt = result.x
sigma_opt = np.exp(log_sigma_opt)

print(f"\nNumerical MLE (optimization):")
print(f"  μ̂ = {mu_opt:.4f}, σ̂ = {sigma_opt:.4f}")
print(f"  Matches analytical? μ: {np.isclose(mu_mle, mu_opt, atol=0.01)}, "
      f"σ: {np.isclose(sigma_mle, sigma_opt, atol=0.01)}")

# ──────────────────────────────────────────────────────────────────────
# 4.3 MLE FOR LINEAR REGRESSION (WHY WE USE MSE LOSS)
# ──────────────────────────────────────────────────────────────────────
print("\n" + "─" * 70)
print("4.3 MLE FOR LINEAR REGRESSION → MSE LOSS")
print("─" * 70)

# Linear regression assumes: y = Xw + ε, where ε ~ N(0, σ²)
# This means: y|X,w ~ N(Xw, σ²)
#
# Log-likelihood:
#   log L = -n/2 log(2πσ²) - (1/2σ²) Σ(y_i - x_i·w)²
#
# Maximizing this w.r.t. w is equivalent to MINIMIZING:
#   Σ(y_i - x_i·w)² = MSE * n
#
# KEY INSIGHT: Minimizing MSE IS Maximum Likelihood under Gaussian noise!

np.random.seed(42)
n = 200
X = np.random.randn(n, 2)
true_w = np.array([3.0, -1.5])
noise_std = 0.5
y = X @ true_w + np.random.randn(n) * noise_std

# Method 1: Analytical MLE (Normal Equation)
w_mle = np.linalg.inv(X.T @ X) @ X.T @ y

# Method 2: Numerical MLE
def neg_log_likelihood_linear(params, X, y):
    w = params[:X.shape[1]]
    log_sigma = params[X.shape[1]]
    sigma = np.exp(log_sigma)
    residuals = y - X @ w
    n = len(y)
    nll = n/2 * np.log(2 * np.pi * sigma**2) + np.sum(residuals**2) / (2 * sigma**2)
    return nll

result = minimize(neg_log_likelihood_linear,
                  x0=np.zeros(3),
                  args=(X, y),
                  method='Nelder-Mead')

w_numerical = result.x[:2]
sigma_estimated = np.exp(result.x[2])

print(f"True weights: {true_w}, noise σ = {noise_std}")
print(f"MLE (analytical):  w = {w_mle.round(4)}")
print(f"MLE (numerical):   w = {w_numerical.round(4)}")
print(f"Estimated noise σ: {sigma_estimated:.4f}")
print(f"\nKEY INSIGHT:")
print(f"  MSE loss = MLE under Gaussian noise assumption")
print(f"  This is WHY linear regression uses MSE as the loss function!")

# ──────────────────────────────────────────────────────────────────────
# 4.4 MLE FOR LOGISTIC REGRESSION (WHY WE USE CROSS-ENTROPY)
# ──────────────────────────────────────────────────────────────────────
print("\n" + "─" * 70)
print("4.4 MLE FOR LOGISTIC REGRESSION → CROSS-ENTROPY LOSS")
print("─" * 70)

# Logistic regression assumes: P(y=1|x) = sigmoid(x·w)
# sigmoid(z) = 1 / (1 + exp(-z))
#
# Log-likelihood:
#   log L = Σ [y_i * log(p_i) + (1-y_i) * log(1-p_i)]
#
# Maximizing this = Minimizing BINARY CROSS-ENTROPY!
# This is WHY classification uses cross-entropy as the loss.

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

# Generate binary classification data
np.random.seed(42)
n = 300
X = np.random.randn(n, 2)
true_w = np.array([2.0, -1.0])
true_b = 0.5
probs = sigmoid(X @ true_w + true_b)
y = (np.random.random(n) < probs).astype(float)

# Add bias column
X_b = np.column_stack([np.ones(n), X])

# MLE via gradient ascent (or equivalently, minimize negative log-likelihood)
def neg_log_likelihood_logistic(w, X, y):
    p = sigmoid(X @ w)
    p = np.clip(p, 1e-15, 1 - 1e-15)
    nll = -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))
    return nll

def gradient_logistic(w, X, y):
    p = sigmoid(X @ w)
    return -X.T @ (y - p)

# Gradient descent
w = np.zeros(3)
lr = 0.01
for epoch in range(2000):
    grad = gradient_logistic(w, X_b, y)
    w = w - lr * grad

# Compare with scipy minimize
result = minimize(neg_log_likelihood_logistic,
                  x0=np.zeros(3),
                  args=(X_b, y),
                  method='L-BFGS-B',
                  jac=lambda w, X, y: gradient_logistic(w, X, y))

print(f"True parameters: bias={true_b}, weights={true_w}")
print(f"MLE (gradient descent):  {w.round(4)}")
print(f"MLE (scipy optimizer):   {result.x.round(4)}")

# Accuracy
preds_gd = (sigmoid(X_b @ w) >= 0.5).astype(float)
acc = np.mean(preds_gd == y)
print(f"\nTraining accuracy: {acc:.2%}")

print(f"\nKEY INSIGHT:")
print(f"  Cross-entropy loss = negative log-likelihood of Bernoulli")
print(f"  This is WHY classification uses cross-entropy as the loss function!")

# ──────────────────────────────────────────────────────────────────────
# 4.5 MAP ESTIMATION (MLE + REGULARIZATION)
# ──────────────────────────────────────────────────────────────────────
print("\n" + "─" * 70)
print("4.5 MAP ESTIMATION (MLE + REGULARIZATION)")
print("─" * 70)

# MAP = Maximum A Posteriori
# Instead of just maximizing likelihood, we also include a PRIOR on θ:
#   θ_MAP = argmax P(θ|D) = argmax P(D|θ) * P(θ)
#   log posterior = log likelihood + log prior
#
# KEY INSIGHT:
# - Gaussian prior on weights N(0, σ²) → L2 regularization (Ridge)
# - Laplace prior on weights            → L1 regularization (Lasso)

np.random.seed(42)
n = 50  # small dataset → regularization matters more
X = np.random.randn(n, 5)
true_w = np.array([3.0, -1.0, 0, 0, 0])  # only 2 features matter
y = X @ true_w + np.random.randn(n) * 0.5

# MLE (no regularization)
w_mle = np.linalg.inv(X.T @ X) @ X.T @ y

# MAP with Gaussian prior (= Ridge regression)
# log posterior = -1/(2σ²) * ||y - Xw||² - λ/2 * ||w||²
# Equivalent to: minimize ||y - Xw||² + λ||w||²
lambda_ridge = 1.0
w_ridge = np.linalg.inv(X.T @ X + lambda_ridge * np.eye(X.shape[1])) @ X.T @ y

# MAP with Laplace prior (= Lasso regression, approximate)
from sklearn.linear_model import Lasso
lasso = Lasso(alpha=0.1, fit_intercept=False)
lasso.fit(X, y)
w_lasso = lasso.coef_

print(f"True weights:       {true_w}")
print(f"MLE (no prior):     {w_mle.round(4)}")
print(f"MAP/Ridge (L2):     {w_ridge.round(4)}")
print(f"MAP/Lasso (L1):     {w_lasso.round(4)}")

print(f"\nNotice:")
print(f"  MLE overfits → non-zero weights where true weight is 0")
print(f"  Ridge (L2/Gaussian prior) → shrinks all weights toward 0")
print(f"  Lasso (L1/Laplace prior) → some weights become exactly 0 (sparse!)")

# Visualize the comparison
fig, ax = plt.subplots(figsize=(10, 6))
x_pos = np.arange(len(true_w))
width = 0.2

ax.bar(x_pos - 1.5*width, true_w, width, label='True', color='black', alpha=0.8)
ax.bar(x_pos - 0.5*width, w_mle, width, label='MLE', color='#FF6B6B')
ax.bar(x_pos + 0.5*width, w_ridge, width, label='Ridge (MAP/L2)', color='#4ECDC4')
ax.bar(x_pos + 1.5*width, w_lasso, width, label='Lasso (MAP/L1)', color='#45B7D1')

ax.set_xlabel('Weight Index')
ax.set_ylabel('Weight Value')
ax.set_title('MLE vs MAP: Effect of Priors/Regularization')
ax.set_xticks(x_pos)
ax.set_xticklabels([f'w{i}' for i in range(len(true_w))])
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('probability_for_ml/04_mle_vs_map.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"\nMLE vs MAP comparison saved to: 04_mle_vs_map.png")

print("\n" + "=" * 70)
print("PART 4 COMPLETE -- You now understand MLE and MAP!")
print("Key takeaways:")
print("  - MLE: find parameters that maximize P(data|params)")
print("  - MSE loss = MLE under Gaussian noise → linear regression")
print("  - Cross-entropy = MLE under Bernoulli → logistic regression")
print("  - MAP = MLE + prior → regularization")
print("  - Gaussian prior → Ridge (L2), Laplace prior → Lasso (L1)")
print("  - Regularization = Bayesian prior on weights")
print("=" * 70)
