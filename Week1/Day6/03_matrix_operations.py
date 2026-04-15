"""
=============================================================================
PART 3: MATRIX OPERATIONS FOR ML
=============================================================================

The operations in this file directly underpin core ML algorithms:
- Matrix inverse → Normal Equation (linear regression closed-form)
- Determinant → checking if a system has a unique solution
- Rank → detecting multicollinearity in features
- Solving Ax = b → the heart of many optimization problems

Topics covered:
  3.1 Matrix Inverse and Solving Linear Systems
  3.2 Determinant, Trace, and Rank
"""

import numpy as np

print("=" * 70)
print("PART 3: MATRIX OPERATIONS FOR ML")
print("=" * 70)

# ──────────────────────────────────────────────────────────────────────
# 3.1 MATRIX INVERSE AND SOLVING LINEAR SYSTEMS
# ──────────────────────────────────────────────────────────────────────
print("\n" + "─" * 70)
print("3.1 MATRIX INVERSE AND SOLVING LINEAR SYSTEMS")
print("─" * 70)

# The inverse of matrix A, written A⁻¹, satisfies: A @ A⁻¹ = I (identity)
# Only SQUARE, NON-SINGULAR matrices have an inverse.

A = np.array([[2, 1],
              [5, 3]])

A_inv = np.linalg.inv(A)
print(f"A:\n{A}\n")
print(f"A inverse:\n{A_inv}\n")

# Verify: A @ A⁻¹ should give the identity matrix
product = A @ A_inv
print(f"A @ A_inv (should be identity):\n{product.round(10)}\n")
print(f"Verified: {np.allclose(product, np.eye(2))}")

# ─── Solving Ax = b ─────────────────────────────────────────────────

# System of linear equations:
#   2x + y  = 4
#   5x + 3y = 7
#
# In matrix form: A @ x = b

b = np.array([4, 7])

# Method 1: Using np.linalg.solve (PREFERRED -- faster and numerically stable)
x_solve = np.linalg.solve(A, b)
print(f"\n--- Solving Ax = b ---")
print(f"A = {A.tolist()}")
print(f"b = {b}")
print(f"\nSolution (np.linalg.solve): x = {x_solve}")

# Method 2: Using the inverse (works but less stable for large matrices)
x_inv = A_inv @ b
print(f"Solution (A_inv @ b):       x = {x_inv}")

# Verify the solution
print(f"\nVerification: A @ x = {A @ x_solve} (should be {b})")

# ─── Pseudo-inverse (for non-square or singular matrices) ──────────

print(f"\n--- Pseudo-inverse (Moore-Penrose) ---")

# When A is not square (more equations than unknowns, or vice versa),
# we can't compute a regular inverse. The pseudo-inverse gives us the
# "best approximate" solution (least squares).

A_rect = np.array([[1, 2],
                    [3, 4],
                    [5, 6]])  # 3×2 (not square!)

A_pinv = np.linalg.pinv(A_rect)
print(f"A (3×2):\n{A_rect}")
print(f"\nPseudo-inverse (2×3):\n{A_pinv.round(4)}")
print(f"\nA_pinv @ A (should be close to identity):\n{(A_pinv @ A_rect).round(4)}")

# ─── THE NORMAL EQUATION (Linear Regression Closed-Form) ────────────

print(f"\n--- THE NORMAL EQUATION ---")
print(f"θ = (X^T X)^(-1) X^T y")
print(f"This is the exact solution for linear regression!\n")

# Generate synthetic data: y = 3*x1 + (-2)*x2 + 1*x3 + noise
np.random.seed(42)
n_samples = 200
true_weights = np.array([3.0, -2.0, 1.0])

X = np.random.randn(n_samples, 3)  # 200 samples, 3 features
noise = np.random.randn(n_samples) * 0.1
y = X @ true_weights + noise

print(f"True weights: {true_weights}")
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")

# Normal Equation: θ = (X^T X)^(-1) X^T y
theta_normal = np.linalg.inv(X.T @ X) @ X.T @ y
print(f"\nWeights found by Normal Equation: {theta_normal.round(4)}")

# More numerically stable version using np.linalg.lstsq
theta_lstsq, residuals, rank, sv = np.linalg.lstsq(X, y, rcond=None)
print(f"Weights found by np.linalg.lstsq: {theta_lstsq.round(4)}")

# Adding a bias term (intercept) by prepending a column of 1s
X_with_bias = np.column_stack([np.ones(n_samples), X])
true_weights_with_bias = np.array([5.0, 3.0, -2.0, 1.0])  # bias = 5
y_biased = X_with_bias @ true_weights_with_bias + noise

theta_with_bias = np.linalg.inv(X_with_bias.T @ X_with_bias) @ X_with_bias.T @ y_biased
print(f"\nTrue weights (with bias):  {true_weights_with_bias}")
print(f"Found weights (with bias): {theta_with_bias.round(4)}")
print(f"  Bias (intercept): {theta_with_bias[0]:.4f}")
print(f"  Feature weights:  {theta_with_bias[1:].round(4)}")

# ──────────────────────────────────────────────────────────────────────
# 3.2 DETERMINANT, TRACE, AND RANK
# ──────────────────────────────────────────────────────────────────────
print("\n" + "─" * 70)
print("3.2 DETERMINANT, TRACE, AND RANK")
print("─" * 70)

# ─── Determinant ─────────────────────────────────────────────────────

# The determinant tells you:
# 1. Is the matrix invertible? (det ≠ 0 → yes)
# 2. How much does the matrix stretch/compress space?
# 3. Does it flip orientation? (det < 0 → yes)

A = np.array([[1, 2],
              [3, 4]])
print(f"A:\n{A}")
print(f"det(A) = {np.linalg.det(A):.4f}")
print(f"  Non-zero → A is invertible ✓")

# For a 2×2 matrix: det([[a,b],[c,d]]) = ad - bc
print(f"  Manual: 1*4 - 2*3 = {1*4 - 2*3}")

# Singular matrix (NOT invertible -- its rows are linearly dependent)
S = np.array([[1, 2],
              [2, 4]])  # row 2 = 2 * row 1
print(f"\nSingular matrix S:\n{S}")
print(f"det(S) = {np.linalg.det(S):.4f}")
print(f"  Zero → S is NOT invertible ✗")
print(f"  Row 2 is just 2 × Row 1 (linearly dependent)")

# 3×3 determinant
B = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 10]])
print(f"\n3×3 matrix B:\n{B}")
print(f"det(B) = {np.linalg.det(B):.4f}")

# ─── Trace ───────────────────────────────────────────────────────────

# Trace = sum of diagonal elements.
# Properties: trace(A + B) = trace(A) + trace(B)
#             trace(A @ B) = trace(B @ A)  ← useful identity

A = np.array([[1, 2],
              [3, 4]])
print(f"\n--- Trace ---")
print(f"A:\n{A}")
print(f"trace(A) = {np.trace(A)}")  # 1 + 4 = 5
print(f"  Manual: {A[0, 0]} + {A[1, 1]} = {A[0, 0] + A[1, 1]}")

# The trace equals the sum of eigenvalues (we'll verify in Part 4)
eigenvalues = np.linalg.eigvals(A)
print(f"\nEigenvalues of A: {eigenvalues.round(4)}")
print(f"Sum of eigenvalues: {eigenvalues.sum().real:.4f}")
print(f"Trace of A:         {np.trace(A)}")
print(f"They match: {np.isclose(eigenvalues.sum().real, np.trace(A))}")

# ─── Rank ────────────────────────────────────────────────────────────

# Rank = number of linearly independent rows (or columns).
# Full rank → all rows are independent → matrix is invertible.
# Not full rank → some features are redundant → multicollinearity!

print(f"\n--- Rank ---")

# Full rank matrix (all rows independent)
A = np.array([[1, 0, 0],
              [0, 1, 0],
              [0, 0, 1]])
print(f"Identity matrix rank: {np.linalg.matrix_rank(A)}")  # 3

# Rank-deficient matrix (row 3 = row 1 + row 2)
B = np.array([[1, 2, 3],
              [4, 5, 6],
              [5, 7, 9]])  # row 3 = row 1 + row 2
print(f"\nRank-deficient matrix:\n{B}")
print(f"Rank: {np.linalg.matrix_rank(B)}")  # 2, not 3!
print(f"  Row 3 = Row 1 + Row 2 → one row is redundant")

# ML Application: Detecting multicollinearity
print(f"\n--- ML Application: Multicollinearity Detection ---")

np.random.seed(42)
n = 100
feature1 = np.random.randn(n)
feature2 = np.random.randn(n)
feature3 = 2 * feature1 + 3 * feature2  # perfectly correlated!

X_good = np.column_stack([feature1, feature2])
X_bad = np.column_stack([feature1, feature2, feature3])

print(f"Good feature matrix rank: {np.linalg.matrix_rank(X_good)} (full rank = {X_good.shape[1]})")
print(f"Bad feature matrix rank:  {np.linalg.matrix_rank(X_bad)} (full rank would be {X_bad.shape[1]})")
print(f"  Feature 3 = 2*Feature1 + 3*Feature2 → redundant!")
print(f"  This causes the Normal Equation to be numerically unstable.")

# Condition number: measures how sensitive the solution is to input changes
# High condition number = numerically unstable = potential multicollinearity
cond_good = np.linalg.cond(X_good.T @ X_good)
cond_bad = np.linalg.cond(X_bad.T @ X_bad)
print(f"\nCondition number (good features): {cond_good:.2f}")
print(f"Condition number (bad features):  {cond_bad:.2e}")
print(f"  Very high condition number → regularization needed (Ridge/Lasso)")

# ─── Symmetric and Positive Definite Matrices ───────────────────────

print(f"\n--- Symmetric & Positive Definite Matrices ---")

# A symmetric matrix equals its transpose: A = A^T
# The covariance matrix in ML is always symmetric!
data = np.random.randn(100, 3)
cov = np.cov(data, rowvar=False)  # covariance matrix

print(f"Covariance matrix:\n{cov.round(4)}")
print(f"Is symmetric: {np.allclose(cov, cov.T)}")

# Positive definite: all eigenvalues > 0
# The covariance matrix is always positive semi-definite
eigvals = np.linalg.eigvalsh(cov)
print(f"Eigenvalues: {eigvals.round(4)}")
print(f"All positive: {np.all(eigvals > 0)} (positive definite)")

print("\n" + "=" * 70)
print("PART 3 COMPLETE -- You now understand matrix operations for ML!")
print("Key takeaways:")
print("  - np.linalg.solve(A, b) is better than inv(A) @ b")
print("  - Normal Equation θ = (X^T X)^(-1) X^T y solves linear regression")
print("  - det = 0 → matrix not invertible → system has no unique solution")
print("  - Low rank → redundant features → regularization needed")
print("  - High condition number → numerically unstable → use Ridge/Lasso")
print("=" * 70)
