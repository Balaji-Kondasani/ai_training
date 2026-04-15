"""
=============================================================================
PART 1: PROBABILITY BASICS
=============================================================================

Probability is the language of uncertainty. Every ML model outputs
probabilities or is trained using probabilistic principles.

Topics covered:
  1.1 Probability Axioms & Basic Rules
  1.2 Combinatorics (Counting: Permutations & Combinations)
  1.3 Conditional Probability
  1.4 Bayes' Theorem
  1.5 Naive Bayes Classifier from Scratch
"""

import numpy as np
from math import factorial, comb, perm

print("=" * 70)
print("PART 1: PROBABILITY BASICS")
print("=" * 70)

# ──────────────────────────────────────────────────────────────────────
# 1.1 PROBABILITY AXIOMS & BASIC RULES
# ──────────────────────────────────────────────────────────────────────
print("\n" + "─" * 70)
print("1.1 PROBABILITY AXIOMS & BASIC RULES")
print("─" * 70)

# The three axioms of probability:
# 1. P(A) >= 0          (non-negative)
# 2. P(sample_space) = 1 (certainty)
# 3. P(A or B) = P(A) + P(B) if A and B are mutually exclusive

# Let's simulate with dice rolls
np.random.seed(42)
n_rolls = 100_000
rolls = np.random.randint(1, 7, size=n_rolls)  # 1 to 6

print(f"Simulating {n_rolls:,} dice rolls...")
print(f"\nTheoretical P(any face) = 1/6 = {1/6:.4f}")
print(f"Simulated frequencies:")
for face in range(1, 7):
    freq = np.mean(rolls == face)
    print(f"  P({face}) = {freq:.4f}")

# Addition Rule: P(A or B) = P(A) + P(B) - P(A and B)
# P(even OR > 4) = P(even) + P(> 4) - P(even AND > 4)
p_even = np.mean(rolls % 2 == 0)       # {2, 4, 6}
p_gt4 = np.mean(rolls > 4)              # {5, 6}
p_even_and_gt4 = np.mean((rolls % 2 == 0) & (rolls > 4))  # {6}
p_even_or_gt4 = np.mean((rolls % 2 == 0) | (rolls > 4))   # {2, 4, 5, 6}

print(f"\n--- Addition Rule ---")
print(f"P(even) = {p_even:.4f}  (theoretical: 0.5)")
print(f"P(>4) = {p_gt4:.4f}  (theoretical: {2/6:.4f})")
print(f"P(even AND >4) = {p_even_and_gt4:.4f}  (theoretical: {1/6:.4f})")
print(f"P(even OR >4):")
print(f"  Simulated directly:  {p_even_or_gt4:.4f}")
print(f"  By addition rule:    {p_even + p_gt4 - p_even_and_gt4:.4f}")
print(f"  Theoretical:         {4/6:.4f}")

# Complement Rule: P(not A) = 1 - P(A)
p_not_six = 1 - np.mean(rolls == 6)
print(f"\n--- Complement Rule ---")
print(f"P(not 6) = 1 - P(6) = {p_not_six:.4f}  (theoretical: {5/6:.4f})")

# Independent events: P(A and B) = P(A) * P(B)
# Two dice: P(first=6 AND second=6)
rolls_a = np.random.randint(1, 7, n_rolls)
rolls_b = np.random.randint(1, 7, n_rolls)
p_both_six = np.mean((rolls_a == 6) & (rolls_b == 6))
print(f"\n--- Independence ---")
print(f"P(die1=6 AND die2=6) = {p_both_six:.4f}")
print(f"P(6) * P(6) = {(1/6) * (1/6):.4f}")

# ──────────────────────────────────────────────────────────────────────
# 1.2 COMBINATORICS (Counting)
# ──────────────────────────────────────────────────────────────────────
print("\n" + "─" * 70)
print("1.2 COMBINATORICS (Counting)")
print("─" * 70)

# Factorial: n! = n * (n-1) * ... * 1
print(f"5! = {factorial(5)}")  # 120
print(f"10! = {factorial(10)}")  # 3628800

# Permutations: order MATTERS
# P(n, k) = n! / (n-k)!
# "How many ways to arrange k items from n?"
print(f"\n--- Permutations (order matters) ---")
print(f"P(5, 3) = {perm(5, 3)}")  # 5*4*3 = 60
print(f"  '5 people, choose 3 for president/VP/secretary: {perm(5, 3)} ways'")

# Combinations: order does NOT matter
# C(n, k) = n! / (k! * (n-k)!)
# "How many ways to choose k items from n?"
print(f"\n--- Combinations (order doesn't matter) ---")
print(f"C(5, 3) = {comb(5, 3)}")  # 10
print(f"  '5 people, choose a committee of 3: {comb(5, 3)} ways'")

# ML Application: How many ways to split a dataset?
n_samples = 100
test_size = 20
ways_to_split = comb(n_samples, test_size)
print(f"\n--- ML Application ---")
print(f"Ways to choose {test_size} test samples from {n_samples}: {ways_to_split:.2e}")

# Poker hand probability: P(flush in 5-card hand)
# All 5 cards same suit: C(4,1)*C(13,5) / C(52,5)
total_hands = comb(52, 5)
flush_hands = 4 * comb(13, 5)
p_flush = flush_hands / total_hands
print(f"\nPoker: P(flush) = {flush_hands}/{total_hands} = {p_flush:.6f}")

# ──────────────────────────────────────────────────────────────────────
# 1.3 CONDITIONAL PROBABILITY
# ──────────────────────────────────────────────────────────────────────
print("\n" + "─" * 70)
print("1.3 CONDITIONAL PROBABILITY")
print("─" * 70)

# P(A|B) = P(A and B) / P(B)
# "Probability of A, given that B has happened"

# Example: Medical test
# Disease prevalence: 1%
# Test sensitivity (true positive rate): 95%   P(test+|disease)
# Test specificity (true negative rate): 90%   P(test-|no disease)

print("--- Medical Test Example ---")
n_people = 1_000_000
prevalence = 0.01

# Simulate the population
np.random.seed(42)
has_disease = np.random.random(n_people) < prevalence

# Simulate test results
test_results = np.zeros(n_people, dtype=bool)
sensitivity = 0.95  # P(test+|disease)
specificity = 0.90  # P(test-|no disease)

# People WITH disease: 95% test positive
test_results[has_disease] = np.random.random(has_disease.sum()) < sensitivity
# People WITHOUT disease: 10% test positive (false positives)
test_results[~has_disease] = np.random.random((~has_disease).sum()) < (1 - specificity)

# Conditional probability: P(disease | test+)
tested_positive = test_results
p_disease_given_positive = np.mean(has_disease[tested_positive])

print(f"Population: {n_people:,}")
print(f"Disease prevalence: {prevalence:.1%}")
print(f"Test sensitivity: {sensitivity:.0%}")
print(f"Test specificity: {specificity:.0%}")
print(f"\nResults:")
print(f"  Total with disease: {has_disease.sum():,}")
print(f"  Total tested positive: {tested_positive.sum():,}")
print(f"  True positives: {(has_disease & tested_positive).sum():,}")
print(f"  False positives: {(~has_disease & tested_positive).sum():,}")
print(f"\n  P(disease | test+) = {p_disease_given_positive:.4f}")
print(f"  Only {p_disease_given_positive:.1%} of positive tests actually have the disease!")
print(f"  This is the Base Rate Fallacy -- a crucial concept in ML evaluation.")

# Joint vs Marginal probability
print(f"\n--- Joint and Marginal Probabilities ---")
# Create a contingency table
tp = np.sum(has_disease & tested_positive)
fp = np.sum(~has_disease & tested_positive)
fn = np.sum(has_disease & ~tested_positive)
tn = np.sum(~has_disease & ~tested_positive)

print(f"                    Disease+    Disease-    Total")
print(f"  Test+         {tp:>10,}  {fp:>10,}  {tp + fp:>10,}")
print(f"  Test-         {fn:>10,}  {tn:>10,}  {fn + tn:>10,}")
print(f"  Total         {tp + fn:>10,}  {fp + tn:>10,}  {n_people:>10,}")

# ──────────────────────────────────────────────────────────────────────
# 1.4 BAYES' THEOREM
# ──────────────────────────────────────────────────────────────────────
print("\n" + "─" * 70)
print("1.4 BAYES' THEOREM")
print("─" * 70)

# Bayes' Theorem:
#   P(A|B) = P(B|A) * P(A) / P(B)
#
# In ML terms:
#   posterior = likelihood * prior / evidence
#
# This is the foundation of ALL Bayesian machine learning.

print("--- Bayes' Theorem: Medical Test (Exact Calculation) ---")

# Using the same medical test example
prior = 0.01                  # P(disease) = prevalence
likelihood = 0.95             # P(test+|disease) = sensitivity
false_positive_rate = 0.10    # P(test+|no disease) = 1 - specificity

# Evidence: P(test+) = P(test+|disease)*P(disease) + P(test+|no disease)*P(no disease)
evidence = likelihood * prior + false_positive_rate * (1 - prior)

# Posterior: P(disease|test+)
posterior = (likelihood * prior) / evidence

print(f"Prior P(disease): {prior}")
print(f"Likelihood P(test+|disease): {likelihood}")
print(f"Evidence P(test+): {evidence:.4f}")
print(f"Posterior P(disease|test+): {posterior:.4f}")
print(f"\nSimulated result was: {p_disease_given_positive:.4f}")
print(f"Bayes exact result:   {posterior:.4f}")

# Sequential Bayesian updating: what if they test positive TWICE?
print(f"\n--- Sequential Bayesian Updating ---")
print(f"What if the person tests positive TWICE (independent tests)?")

prior_after_first = posterior  # use posterior as new prior
evidence_2 = likelihood * prior_after_first + false_positive_rate * (1 - prior_after_first)
posterior_after_second = (likelihood * prior_after_first) / evidence_2

print(f"After 1st positive test: P(disease) = {posterior:.4f}")
print(f"After 2nd positive test: P(disease) = {posterior_after_second:.4f}")
print(f"Each positive test increases our confidence!")

# ──────────────────────────────────────────────────────────────────────
# 1.5 NAIVE BAYES CLASSIFIER FROM SCRATCH
# ──────────────────────────────────────────────────────────────────────
print("\n" + "─" * 70)
print("1.5 NAIVE BAYES CLASSIFIER FROM SCRATCH")
print("─" * 70)

# Naive Bayes uses Bayes' theorem with the "naive" assumption that
# features are conditionally independent given the class label.
#
# P(class|features) ∝ P(class) * ∏ P(feature_i|class)
#
# For Gaussian Naive Bayes:
# P(feature_i|class) = Normal(mean_i_class, var_i_class)

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print(f"Dataset: Iris ({X.shape[0]} samples, {X.shape[1]} features, 3 classes)")
print(f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")

class GaussianNaiveBayesScratch:
    """Gaussian Naive Bayes built from scratch using NumPy."""

    def fit(self, X, y):
        self.classes = np.unique(y)
        self.n_classes = len(self.classes)
        self.n_features = X.shape[1]

        # For each class, compute prior, mean, and variance of each feature
        self.priors = np.zeros(self.n_classes)
        self.means = np.zeros((self.n_classes, self.n_features))
        self.variances = np.zeros((self.n_classes, self.n_features))

        for idx, c in enumerate(self.classes):
            X_c = X[y == c]
            self.priors[idx] = len(X_c) / len(X)
            self.means[idx] = X_c.mean(axis=0)
            self.variances[idx] = X_c.var(axis=0) + 1e-9  # add small value for stability

        return self

    def _gaussian_log_likelihood(self, x, mean, var):
        """Log of Gaussian PDF: avoids numerical underflow from tiny probabilities."""
        return -0.5 * np.log(2 * np.pi * var) - 0.5 * ((x - mean) ** 2 / var)

    def predict(self, X):
        predictions = []
        for x in X:
            # Compute log P(class) + sum of log P(feature_i|class) for each class
            log_posteriors = []
            for idx in range(self.n_classes):
                log_prior = np.log(self.priors[idx])
                log_likelihood = np.sum(
                    self._gaussian_log_likelihood(x, self.means[idx], self.variances[idx])
                )
                log_posteriors.append(log_prior + log_likelihood)

            predictions.append(self.classes[np.argmax(log_posteriors)])
        return np.array(predictions)

    def score(self, X, y):
        return np.mean(self.predict(X) == y)

# Train and evaluate our implementation
nb_ours = GaussianNaiveBayesScratch()
nb_ours.fit(X_train, y_train)

print(f"\n--- Model Parameters ---")
for idx, c in enumerate(nb_ours.classes):
    print(f"Class {iris.target_names[c]}:")
    print(f"  Prior: {nb_ours.priors[idx]:.3f}")
    print(f"  Means: {nb_ours.means[idx].round(3)}")
    print(f"  Vars:  {nb_ours.variances[idx].round(3)}")

# Compare with sklearn
nb_sklearn = GaussianNB()
nb_sklearn.fit(X_train, y_train)

acc_ours = nb_ours.score(X_test, y_test)
acc_sklearn = nb_sklearn.score(X_test, y_test)

print(f"\n--- Results ---")
print(f"Our Naive Bayes accuracy:     {acc_ours:.4f}")
print(f"sklearn Naive Bayes accuracy: {acc_sklearn:.4f}")

# Show predictions for first 5 test samples
preds = nb_ours.predict(X_test[:5])
print(f"\nFirst 5 predictions: {[iris.target_names[p] for p in preds]}")
print(f"Actual labels:       {[iris.target_names[t] for t in y_test[:5]]}")

print("\n" + "=" * 70)
print("PART 1 COMPLETE -- You now understand probability basics!")
print("Key takeaways:")
print("  - P(A|B) = P(A and B) / P(B) -- conditional probability")
print("  - Bayes: posterior = likelihood * prior / evidence")
print("  - Base rate fallacy: rare events have many false positives")
print("  - Naive Bayes: simple, fast, surprisingly effective classifier")
print("=" * 70)
