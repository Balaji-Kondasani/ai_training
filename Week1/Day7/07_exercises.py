"""
=============================================================================
EXERCISES: 7 Practice Problems Combining All Probability Concepts
=============================================================================

Each exercise applies probability/statistics to a real ML scenario.

  Exercise 1: Naive Bayes Spam Classifier
  Exercise 2: A/B Testing with Statistical Rigor
  Exercise 3: Gaussian Mixture Model (EM Algorithm)
  Exercise 4: MLE for Distribution Fitting
  Exercise 5: Bootstrap Model Evaluation
  Exercise 6: Information Gain -- Build a Decision Tree Split
  Exercise 7: Bayesian Coin Flip (Prior Updating)
"""

import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

print("=" * 70)
print("EXERCISES: 7 Practice Problems -- Probability for ML")
print("=" * 70)

# ╔════════════════════════════════════════════════════════════════════╗
# ║  EXERCISE 1: Naive Bayes Spam Classifier (Multinomial)           ║
# ╚════════════════════════════════════════════════════════════════════╝
print("\n" + "═" * 70)
print("EXERCISE 1: Multinomial Naive Bayes Spam Classifier")
print("═" * 70)

# Build a word-count based Naive Bayes from scratch.

np.random.seed(42)

# Simple vocabulary and training data
emails = [
    ("free money now click here", "spam"),
    ("win free prizes congratulations", "spam"),
    ("free offer limited time", "spam"),
    ("click here for discount", "spam"),
    ("meeting tomorrow at office", "ham"),
    ("project deadline next week", "ham"),
    ("lunch meeting confirmed", "ham"),
    ("please review the document", "ham"),
    ("urgent meeting schedule update", "ham"),
    ("free project review meeting", "ham"),
]

# Build vocabulary
all_words = set()
for text, _ in emails:
    all_words.update(text.split())
vocab = sorted(all_words)
word_to_idx = {w: i for i, w in enumerate(vocab)}

print(f"Vocabulary ({len(vocab)} words): {vocab}")

# Convert emails to word count vectors
def text_to_counts(text, word_to_idx):
    counts = np.zeros(len(word_to_idx))
    for word in text.split():
        if word in word_to_idx:
            counts[word_to_idx[word]] += 1
    return counts

X_train = np.array([text_to_counts(text, word_to_idx) for text, _ in emails])
y_train = np.array([1 if label == "spam" else 0 for _, label in emails])

# Train Multinomial Naive Bayes from scratch
class MultinomialNBScratch:
    def fit(self, X, y, alpha=1.0):
        """Train with Laplace smoothing (alpha=1)."""
        self.classes = np.unique(y)
        self.priors = {}
        self.word_probs = {}

        for c in self.classes:
            X_c = X[y == c]
            self.priors[c] = len(X_c) / len(X)

            # Word probabilities with Laplace smoothing
            word_counts = X_c.sum(axis=0) + alpha
            total_words = word_counts.sum()
            self.word_probs[c] = word_counts / total_words

        return self

    def predict_log_prob(self, x):
        log_probs = {}
        for c in self.classes:
            log_prior = np.log(self.priors[c])
            log_likelihood = np.sum(x * np.log(self.word_probs[c]))
            log_probs[c] = log_prior + log_likelihood
        return log_probs

    def predict(self, X):
        predictions = []
        for x in X:
            log_probs = self.predict_log_prob(x)
            predictions.append(max(log_probs, key=log_probs.get))
        return np.array(predictions)

nb = MultinomialNBScratch()
nb.fit(X_train, y_train)

# Test on new emails
test_emails = [
    "free money click",
    "meeting tomorrow project",
    "win free discount offer",
    "please schedule meeting",
]

print(f"\nPriors: P(spam)={nb.priors[1]:.2f}, P(ham)={nb.priors[0]:.2f}")
print(f"\n--- Predictions ---")
for email in test_emails:
    x = text_to_counts(email, word_to_idx)
    log_probs = nb.predict_log_prob(x)
    pred = "spam" if max(log_probs, key=log_probs.get) == 1 else "ham"
    print(f'  "{email}" → {pred}')

# Train accuracy
train_preds = nb.predict(X_train)
train_acc = np.mean(train_preds == y_train)
print(f"\nTraining accuracy: {train_acc:.0%}")

print("\n✓ Exercise 1 Complete!")

# ╔════════════════════════════════════════════════════════════════════╗
# ║  EXERCISE 2: A/B Testing with Statistical Rigor                  ║
# ╚════════════════════════════════════════════════════════════════════╝
print("\n" + "═" * 70)
print("EXERCISE 2: A/B Testing")
print("═" * 70)

# Scenario: testing if a new button color increases click-through rate.

np.random.seed(42)

# Simulate data
n_a = 1000  # control group
n_b = 1000  # treatment group
p_a = 0.10  # 10% click-through
p_b = 0.13  # 13% click-through (true effect)

clicks_a = np.random.binomial(1, p_a, n_a)
clicks_b = np.random.binomial(1, p_b, n_b)

rate_a = clicks_a.mean()
rate_b = clicks_b.mean()
observed_diff = rate_b - rate_a

print(f"Control (A):   {rate_a:.3f} CTR ({clicks_a.sum()}/{n_a})")
print(f"Treatment (B): {rate_b:.3f} CTR ({clicks_b.sum()}/{n_b})")
print(f"Observed lift: {observed_diff:+.3f} ({observed_diff/rate_a:+.1%})")

# Method 1: Z-test for proportions
p_pooled = (clicks_a.sum() + clicks_b.sum()) / (n_a + n_b)
se = np.sqrt(p_pooled * (1 - p_pooled) * (1/n_a + 1/n_b))
z_stat = observed_diff / se
p_value_z = 2 * (1 - stats.norm.cdf(abs(z_stat)))

print(f"\n--- Z-test for Proportions ---")
print(f"Z-statistic: {z_stat:.4f}")
print(f"p-value: {p_value_z:.4f}")
print(f"Significant at α=0.05? {'Yes' if p_value_z < 0.05 else 'No'}")

# Method 2: Permutation test
n_perms = 10000
combined = np.concatenate([clicks_a, clicks_b])
perm_diffs = np.zeros(n_perms)

for i in range(n_perms):
    shuffled = np.random.permutation(combined)
    perm_diffs[i] = shuffled[n_a:].mean() - shuffled[:n_a].mean()

p_value_perm = np.mean(np.abs(perm_diffs) >= abs(observed_diff))
print(f"\n--- Permutation Test ---")
print(f"p-value: {p_value_perm:.4f}")

# Method 3: Bootstrap CI for the difference
n_boot = 10000
boot_diffs = np.zeros(n_boot)
for i in range(n_boot):
    boot_a = np.random.choice(clicks_a, n_a, replace=True).mean()
    boot_b = np.random.choice(clicks_b, n_b, replace=True).mean()
    boot_diffs[i] = boot_b - boot_a

ci_lower = np.percentile(boot_diffs, 2.5)
ci_upper = np.percentile(boot_diffs, 97.5)

print(f"\n--- Bootstrap 95% CI for Difference ---")
print(f"CI: [{ci_lower:+.4f}, {ci_upper:+.4f}]")
print(f"Contains 0? {'Yes → not significant' if ci_lower <= 0 <= ci_upper else 'No → significant'}")

# Sample size calculation
print(f"\n--- Sample Size Planning ---")
mde = 0.02  # minimum detectable effect
baseline = 0.10
alpha_level = 0.05
power = 0.80

z_alpha = stats.norm.ppf(1 - alpha_level/2)
z_beta = stats.norm.ppf(power)
p1, p2 = baseline, baseline + mde
n_required = ((z_alpha * np.sqrt(2*baseline*(1-baseline)) +
               z_beta * np.sqrt(p1*(1-p1) + p2*(1-p2))) / mde) ** 2

print(f"To detect {mde:.0%} lift from {baseline:.0%} baseline:")
print(f"  Required n per group: {int(np.ceil(n_required)):,}")

print("\n✓ Exercise 2 Complete!")

# ╔════════════════════════════════════════════════════════════════════╗
# ║  EXERCISE 3: Gaussian Mixture Model (EM Algorithm)               ║
# ╚════════════════════════════════════════════════════════════════════╝
print("\n" + "═" * 70)
print("EXERCISE 3: Gaussian Mixture Model (EM Algorithm)")
print("═" * 70)

# GMM assumes data comes from K Gaussian distributions.
# EM Algorithm alternates between:
#   E-step: assign soft responsibilities (which cluster does each point belong to?)
#   M-step: update parameters (means, variances, weights)

np.random.seed(42)

# Generate data from 3 Gaussians
true_means = [-3, 0, 4]
true_stds = [0.8, 1.2, 0.6]
true_weights = [0.3, 0.4, 0.3]
n_total = 500

data = np.concatenate([
    np.random.normal(m, s, int(w * n_total))
    for m, s, w in zip(true_means, true_stds, true_weights)
])
np.random.shuffle(data)

print(f"Generated {len(data)} samples from 3 Gaussians")
print(f"True means: {true_means}, stds: {true_stds}, weights: {true_weights}")

# EM Algorithm from scratch
K = 3  # number of components
n = len(data)

# Initialize parameters randomly
means = np.random.choice(data, K)
variances = np.ones(K)
weights = np.ones(K) / K

print(f"\nInitial means: {means.round(3)}")

n_iterations = 50
log_likelihoods = []

for iteration in range(n_iterations):
    # E-step: compute responsibilities
    # r[i, k] = weight_k * N(x_i | mu_k, var_k) / sum_j(weight_j * N(x_i | mu_j, var_j))
    responsibilities = np.zeros((n, K))
    for k in range(K):
        responsibilities[:, k] = weights[k] * stats.norm.pdf(data, means[k], np.sqrt(variances[k]))

    # Normalize
    resp_sum = responsibilities.sum(axis=1, keepdims=True)
    responsibilities /= resp_sum

    # M-step: update parameters
    N_k = responsibilities.sum(axis=0)  # effective number of points per cluster

    for k in range(K):
        weights[k] = N_k[k] / n
        means[k] = np.sum(responsibilities[:, k] * data) / N_k[k]
        variances[k] = np.sum(responsibilities[:, k] * (data - means[k])**2) / N_k[k]

    # Log-likelihood
    ll = np.sum(np.log(resp_sum.flatten()))
    log_likelihoods.append(ll)

    if iteration < 5 or iteration == n_iterations - 1:
        print(f"  Iter {iteration:>2d}: means={means.round(3)}, weights={weights.round(3)}, LL={ll:.2f}")

print(f"\nFinal means:   {sorted(means.round(3))}")
print(f"True means:    {sorted(true_means)}")
print(f"Final weights: {sorted(weights.round(3))}")
print(f"True weights:  {sorted(true_weights)}")

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Data histogram with fitted Gaussians
x_range = np.linspace(data.min() - 1, data.max() + 1, 200)
axes[0].hist(data, bins=50, density=True, alpha=0.5, edgecolor='black', linewidth=0.5)
for k in range(K):
    pdf = weights[k] * stats.norm.pdf(x_range, means[k], np.sqrt(variances[k]))
    axes[0].plot(x_range, pdf, linewidth=2, label=f'Component {k+1}')
total_pdf = sum(weights[k] * stats.norm.pdf(x_range, means[k], np.sqrt(variances[k])) for k in range(K))
axes[0].plot(x_range, total_pdf, 'k--', linewidth=2, label='Mixture')
axes[0].set_title('GMM Fit')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Log-likelihood convergence
axes[1].plot(log_likelihoods, 'b.-')
axes[1].set_xlabel('Iteration')
axes[1].set_ylabel('Log-Likelihood')
axes[1].set_title('EM Convergence')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('probability_for_ml/07_ex3_gmm.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"\nGMM plot saved to: 07_ex3_gmm.png")

print("\n✓ Exercise 3 Complete!")

# ╔════════════════════════════════════════════════════════════════════╗
# ║  EXERCISE 4: MLE for Distribution Fitting                        ║
# ╚════════════════════════════════════════════════════════════════════╝
print("\n" + "═" * 70)
print("EXERCISE 4: MLE -- Fit the Right Distribution")
print("═" * 70)

# Given data, determine which distribution best fits it using MLE.

np.random.seed(42)

# Secret: data is from a Gamma distribution
secret_data = np.random.gamma(shape=3, scale=2, size=500)

print(f"Data statistics: mean={secret_data.mean():.2f}, "
      f"std={secret_data.std():.2f}, min={secret_data.min():.2f}, max={secret_data.max():.2f}")

# Fit several distributions using MLE
distributions_to_try = {
    'Normal': stats.norm,
    'Exponential': stats.expon,
    'Gamma': stats.gamma,
    'Lognormal': stats.lognorm,
}

print(f"\nFitting distributions via MLE:")
fits = {}
for name, dist in distributions_to_try.items():
    params = dist.fit(secret_data)
    # Compute log-likelihood
    ll = np.sum(dist.logpdf(secret_data, *params))
    # AIC = 2k - 2*log(L) where k = number of parameters
    aic = 2 * len(params) - 2 * ll
    fits[name] = {'params': params, 'll': ll, 'aic': aic}
    print(f"  {name:>12s}: LL = {ll:>10.2f}, AIC = {aic:>10.2f}, params = {[round(p, 3) for p in params]}")

best = min(fits, key=lambda x: fits[x]['aic'])
print(f"\nBest fit (lowest AIC): {best}")

# Visualize fits
fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(secret_data, bins=40, density=True, alpha=0.5, edgecolor='black', linewidth=0.5, label='Data')
x_range = np.linspace(0, secret_data.max() * 1.1, 200)

for name, dist in distributions_to_try.items():
    params = fits[name]['params']
    pdf_values = dist.pdf(x_range, *params)
    style = '-' if name == best else '--'
    lw = 2.5 if name == best else 1.5
    ax.plot(x_range, pdf_values, style, linewidth=lw,
            label=f'{name} (AIC={fits[name]["aic"]:.0f})')

ax.set_title('Distribution Fitting via MLE')
ax.set_xlabel('Value')
ax.set_ylabel('Density')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('probability_for_ml/07_ex4_distribution_fitting.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"\nDistribution fitting plot saved to: 07_ex4_distribution_fitting.png")

print("\n✓ Exercise 4 Complete!")

# ╔════════════════════════════════════════════════════════════════════╗
# ║  EXERCISE 5: Bootstrap Model Evaluation                          ║
# ╚════════════════════════════════════════════════════════════════════╝
print("\n" + "═" * 70)
print("EXERCISE 5: Bootstrap Confidence Intervals for Model Accuracy")
print("═" * 70)

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = LogisticRegression(max_iter=200, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
point_accuracy = np.mean(y_pred == y_test)

print(f"Point estimate of accuracy: {point_accuracy:.4f}")

# Bootstrap CI for accuracy
n_bootstrap = 5000
boot_accuracies = np.zeros(n_bootstrap)
n_test = len(y_test)

for i in range(n_bootstrap):
    idx = np.random.randint(0, n_test, n_test)
    boot_acc = np.mean(y_pred[idx] == y_test[idx])
    boot_accuracies[i] = boot_acc

ci_lower = np.percentile(boot_accuracies, 2.5)
ci_upper = np.percentile(boot_accuracies, 97.5)

print(f"Bootstrap 95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
print(f"Bootstrap mean:   {boot_accuracies.mean():.4f}")
print(f"Bootstrap std:    {boot_accuracies.std():.4f}")

# Bootstrap CI for precision of each class
print(f"\nPer-class precision with 95% CI:")
for cls in range(3):
    cls_mask = y_test == cls
    precisions = []
    for _ in range(5000):
        idx = np.random.randint(0, n_test, n_test)
        pred_cls = y_pred[idx] == cls
        true_cls = y_test[idx] == cls
        if pred_cls.sum() > 0:
            precisions.append(np.sum(pred_cls & true_cls) / pred_cls.sum())
    precisions = np.array(precisions)
    print(f"  Class {iris.target_names[cls]:>10s}: "
          f"{np.median(precisions):.3f} [{np.percentile(precisions, 2.5):.3f}, "
          f"{np.percentile(precisions, 97.5):.3f}]")

print("\n✓ Exercise 5 Complete!")

# ╔════════════════════════════════════════════════════════════════════╗
# ║  EXERCISE 6: Information Gain -- Build a Decision Tree Split     ║
# ╚════════════════════════════════════════════════════════════════════╝
print("\n" + "═" * 70)
print("EXERCISE 6: Decision Tree Split using Information Gain")
print("═" * 70)

# Build a single-level decision tree (stump) from scratch

from sklearn.datasets import load_iris

iris = load_iris()
X = iris.data
y = iris.target

def entropy_from_labels(labels):
    _, counts = np.unique(labels, return_counts=True)
    probs = counts / counts.sum()
    return -np.sum(probs * np.log2(probs + 1e-15))

def find_best_split(X, y):
    """Find the best feature and threshold to split on."""
    n_samples, n_features = X.shape
    best_ig = -1
    best_feature = None
    best_threshold = None

    parent_entropy = entropy_from_labels(y)

    for feature_idx in range(n_features):
        thresholds = np.unique(X[:, feature_idx])

        for threshold in thresholds:
            left_mask = X[:, feature_idx] <= threshold
            right_mask = ~left_mask

            if left_mask.sum() == 0 or right_mask.sum() == 0:
                continue

            # Weighted entropy of children
            w_left = left_mask.sum() / n_samples
            w_right = right_mask.sum() / n_samples
            child_entropy = (w_left * entropy_from_labels(y[left_mask]) +
                           w_right * entropy_from_labels(y[right_mask]))

            ig = parent_entropy - child_entropy

            if ig > best_ig:
                best_ig = ig
                best_feature = feature_idx
                best_threshold = threshold

    return best_feature, best_threshold, best_ig

best_feat, best_thresh, best_ig = find_best_split(X, y)

print(f"Parent entropy: {entropy_from_labels(y):.4f} bits")
print(f"\nBest split:")
print(f"  Feature: {iris.feature_names[best_feat]} (index {best_feat})")
print(f"  Threshold: {best_thresh:.2f}")
print(f"  Information Gain: {best_ig:.4f} bits")

# Show the split
left_mask = X[:, best_feat] <= best_thresh
print(f"\n  Left ({left_mask.sum()} samples):  {dict(zip(*np.unique(y[left_mask], return_counts=True)))}")
print(f"  Right ({(~left_mask).sum()} samples): {dict(zip(*np.unique(y[~left_mask], return_counts=True)))}")

# Accuracy of this simple stump
left_class = stats.mode(y[left_mask], keepdims=False).mode
right_class = stats.mode(y[~left_mask], keepdims=False).mode
preds_stump = np.where(X[:, best_feat] <= best_thresh, left_class, right_class)
stump_acc = np.mean(preds_stump == y)
print(f"\n  Decision stump accuracy: {stump_acc:.2%}")

print("\n✓ Exercise 6 Complete!")

# ╔════════════════════════════════════════════════════════════════════╗
# ║  EXERCISE 7: Bayesian Coin Flip (Prior Updating)                 ║
# ╚════════════════════════════════════════════════════════════════════╝
print("\n" + "═" * 70)
print("EXERCISE 7: Bayesian Inference -- Coin Flip")
print("═" * 70)

# Use Beta-Binomial conjugacy:
# Prior:      Beta(α, β)
# Likelihood: Binomial(n, p)
# Posterior:  Beta(α + heads, β + tails)

np.random.seed(42)

# Secret coin bias
true_p = 0.65

# Start with different priors
priors = {
    'Uniform (no info)':    (1, 1),
    'Weak prior (fair)':    (5, 5),
    'Strong prior (fair)':  (50, 50),
    'Prior (biased 0.7)':   (7, 3),
}

# Flip the coin incrementally
all_flips = np.random.binomial(1, true_p, 200)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
x = np.linspace(0, 1, 200)

for ax, (prior_name, (alpha_0, beta_0)) in zip(axes.flatten(), priors.items()):
    print(f"\n--- {prior_name}: Beta({alpha_0}, {beta_0}) ---")

    alpha_current = alpha_0
    beta_current = beta_0

    # Plot prior
    ax.plot(x, stats.beta.pdf(x, alpha_current, beta_current),
            'k--', alpha=0.5, label='Prior')

    for n_flips in [5, 20, 50, 200]:
        flips = all_flips[:n_flips]
        heads = flips.sum()
        tails = n_flips - heads

        alpha_post = alpha_0 + heads
        beta_post = beta_0 + tails

        posterior_mean = alpha_post / (alpha_post + beta_post)
        mle = heads / n_flips

        ax.plot(x, stats.beta.pdf(x, alpha_post, beta_post),
                label=f'n={n_flips} (mean={posterior_mean:.3f})')

        if n_flips in [5, 50, 200]:
            print(f"  After {n_flips:>3d} flips ({heads} H): "
                  f"posterior Beta({alpha_post},{beta_post}), "
                  f"mean={posterior_mean:.3f}, MLE={mle:.3f}")

    ax.axvline(true_p, color='red', linestyle=':', alpha=0.7, label=f'True p={true_p}')
    ax.set_title(prior_name)
    ax.legend(fontsize=8)
    ax.set_xlabel('p')
    ax.set_ylabel('Density')
    ax.grid(True, alpha=0.3)

plt.suptitle('Bayesian Updating: How Priors Get Overwhelmed by Data',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('probability_for_ml/07_ex7_bayesian_updating.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"\nBayesian updating plot saved to: 07_ex7_bayesian_updating.png")

print("\n✓ Exercise 7 Complete!")

# ══════════════════════════════════════════════════════════════════════
print("\n" + "═" * 70)
print("ALL 7 EXERCISES COMPLETE!")
print("═" * 70)
print("""
Summary of what you built from scratch:

  1. Multinomial Naive Bayes spam classifier (Bayes' theorem)
  2. A/B test with z-test, permutation test, and bootstrap CI
  3. Gaussian Mixture Model with EM algorithm
  4. Distribution fitting with MLE and AIC model selection
  5. Bootstrap confidence intervals for ML model metrics
  6. Decision tree split using information gain (entropy)
  7. Bayesian coin flip with Beta-Binomial conjugacy

Each exercise demonstrates a core probability concept
applied to a real ML problem!
""")
