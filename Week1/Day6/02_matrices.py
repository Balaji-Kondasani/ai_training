"""
=============================================================================
PART 2: MATRICES -- The Workhorse of ML
=============================================================================

Your entire dataset IS a matrix: rows = samples, columns = features.
Every ML model operates on matrices.

Topics covered:
  2.1 Matrix Basics (creation, special matrices, indexing, slicing)
  2.2 Matrix Arithmetic (element-wise ops, matrix multiplication)
  2.3 Broadcasting (NumPy's superpower for efficient ML code)
"""

import numpy as np

print("=" * 70)
print("PART 2: MATRICES -- The Workhorse of ML")
print("=" * 70)

# ──────────────────────────────────────────────────────────────────────
# 2.1 MATRIX BASICS
# ──────────────────────────────────────────────────────────────────────
print("\n" + "─" * 70)
print("2.1 MATRIX BASICS")
print("─" * 70)

# A matrix is a 2D array of numbers arranged in rows and columns.
# In ML: rows = data samples, columns = features.

A = np.array([[1, 2, 3],
              [4, 5, 6]])
print(f"Matrix A:\n{A}")
print(f"  Shape: {A.shape}")    # (2, 3) -- 2 rows, 3 columns
print(f"  Rows: {A.shape[0]}, Columns: {A.shape[1]}")

# ─── Special Matrices ───────────────────────────────────────────────

# Identity matrix: the "1" of matrix multiplication (A @ I = A)
I = np.eye(3)
print(f"\nIdentity matrix (3x3):\n{I}")

# Zero matrix
Z = np.zeros((3, 4))
print(f"\nZero matrix (3x4):\n{Z}")

# Ones matrix
O = np.ones((2, 3))
print(f"\nOnes matrix (2x3):\n{O}")

# Diagonal matrix: only has values on the main diagonal
D = np.diag([1, 2, 3])
print(f"\nDiagonal matrix:\n{D}")

# Extract diagonal from an existing matrix
print(f"Diagonal of A: {np.diag(A)}")  # [1, 5] -- main diagonal

# Random matrices (used constantly in ML for initialization)
R_uniform = np.random.rand(2, 3)    # uniform [0, 1)
R_normal = np.random.randn(2, 3)    # standard normal (mean=0, std=1)
R_int = np.random.randint(0, 10, size=(2, 3))  # random integers

print(f"\nRandom uniform:\n{R_uniform}")
print(f"\nRandom normal:\n{R_normal}")
print(f"\nRandom integers (0-9):\n{R_int}")

# ─── Indexing and Slicing ───────────────────────────────────────────

M = np.array([[10, 20, 30, 40],
              [50, 60, 70, 80],
              [90, 100, 110, 120]])
print(f"\nMatrix M:\n{M}")

# Single element access: M[row, col]
print(f"\nM[0, 0] = {M[0, 0]}")   # 10 (top-left)
print(f"M[2, 3] = {M[2, 3]}")     # 120 (bottom-right)
print(f"M[-1, -1] = {M[-1, -1]}") # 120 (last element)

# Row and column slicing
print(f"\nFirst row:    M[0, :] = {M[0, :]}")      # [10, 20, 30, 40]
print(f"Last row:     M[-1, :] = {M[-1, :]}")      # [90, 100, 110, 120]
print(f"First column: M[:, 0] = {M[:, 0]}")        # [10, 50, 90]
print(f"Last column:  M[:, -1] = {M[:, -1]}")      # [40, 80, 120]

# Submatrix slicing: M[row_start:row_end, col_start:col_end]
print(f"\nTop-left 2x2: M[:2, :2] =\n{M[:2, :2]}")
print(f"\nBottom-right 2x3: M[1:, 1:] =\n{M[1:, 1:]}")

# Boolean indexing (used for filtering data in ML)
print(f"\nElements > 50: {M[M > 50]}")
print(f"Elements in first col > 30: rows = {M[M[:, 0] > 30]}")

# ─── Reshape, Transpose, Flatten ────────────────────────────────────

print(f"\n--- Reshape, Transpose, Flatten ---")

v = np.arange(12)  # [0, 1, 2, ..., 11]
print(f"\nOriginal 1D: {v}")

# Reshape: rearrange elements into a new shape (total elements must match)
M1 = v.reshape(3, 4)   # 3 rows, 4 columns
M2 = v.reshape(4, 3)   # 4 rows, 3 columns
M3 = v.reshape(2, 6)   # 2 rows, 6 columns
M4 = v.reshape(-1, 4)  # -1 means "figure it out" -> 3 rows, 4 cols

print(f"\nReshaped to (3,4):\n{M1}")
print(f"\nReshaped to (4,3):\n{M2}")
print(f"\nReshaped to (-1,4) -> {M4.shape}:\n{M4}")

# Transpose: swap rows and columns (rows become columns and vice versa)
A = np.array([[1, 2, 3],
              [4, 5, 6]])
print(f"\nA:\n{A}")
print(f"A shape: {A.shape}")
print(f"\nA transposed (A.T):\n{A.T}")
print(f"A.T shape: {A.T.shape}")

# Flatten and ravel: convert matrix back to 1D
print(f"\nA flattened: {A.flatten()}")  # returns a copy
print(f"A raveled:   {A.ravel()}")     # returns a view (more efficient)

# ──────────────────────────────────────────────────────────────────────
# 2.2 MATRIX ARITHMETIC
# ──────────────────────────────────────────────────────────────────────
print("\n" + "─" * 70)
print("2.2 MATRIX ARITHMETIC")
print("─" * 70)

A = np.array([[1, 2],
              [3, 4]])
B = np.array([[5, 6],
              [7, 8]])

# ─── Element-wise Operations ────────────────────────────────────────
print("--- Element-wise Operations ---")
print(f"A:\n{A}\n")
print(f"B:\n{B}\n")
print(f"A + B =\n{A + B}\n")   # [[6, 8], [10, 12]]
print(f"A - B =\n{A - B}\n")   # [[-4, -4], [-4, -4]]
print(f"A * B (element-wise, NOT matrix mult) =\n{A * B}\n")  # [[5, 12], [21, 32]]
print(f"A / B =\n{A / B}\n")
print(f"A ** 2 =\n{A ** 2}\n") # square each element

# ─── MATRIX MULTIPLICATION ──────────────────────────────────────────
# This is DIFFERENT from element-wise multiplication!
# Rule: (m×n) @ (n×p) = (m×p) -- inner dimensions must match
print("--- Matrix Multiplication ---")
print("Rule: (m×n) @ (n×p) = (m×p)")
print()

# Three equivalent ways to do matrix multiplication:
result1 = A @ B           # preferred (most readable)
result2 = np.matmul(A, B) # equivalent
result3 = np.dot(A, B)    # works for 2D

print(f"A @ B =\n{result1}\n")

# Manual calculation to understand what's happening:
# [1,2] @ [5,6] = [1*5+2*7, 1*6+2*8] = [19, 22]
# [3,4]   [7,8]   [3*5+4*7, 3*6+4*8]   [43, 50]
print("Manual verification:")
print(f"  Row 0: [1*5+2*7, 1*6+2*8] = [{1*5+2*7}, {1*6+2*8}]")
print(f"  Row 1: [3*5+4*7, 3*6+4*8] = [{3*5+4*7}, {3*6+4*8}]")

# The Shape Rule in Action
# This is THE most important rule to memorize for ML:
print(f"\n--- Shape Rule Examples ---")
X = np.random.randn(100, 5)  # 100 samples, 5 features
W = np.random.randn(5, 1)    # 5 weights, 1 output
b = 0.5                       # bias

# Forward pass of linear regression / single neuron
y = X @ W + b  # (100,5) @ (5,1) = (100,1)
print(f"X shape: {X.shape}")
print(f"W shape: {W.shape}")
print(f"y = X @ W + b -> shape: {y.shape}")
print(f"  This gives us 100 predictions, one per sample!")

# Matrix multiplication is NOT commutative: A @ B != B @ A
print(f"\nA @ B =\n{A @ B}")
print(f"\nB @ A =\n{B @ A}")
print(f"\nA @ B == B @ A? {np.array_equal(A @ B, B @ A)}")

# But it IS associative: (A @ B) @ C == A @ (B @ C)
C = np.array([[1, 0], [0, 1]])
print(f"\n(A @ B) @ C == A @ (B @ C)? {np.allclose((A @ B) @ C, A @ (B @ C))}")

# Identity matrix is the "1" of matrix multiplication
print(f"\nA @ I = A? {np.allclose(A @ np.eye(2), A)}")
print(f"I @ A = A? {np.allclose(np.eye(2) @ A, A)}")

# Matrix-vector multiplication (most common in ML)
x = np.array([1, 2])
print(f"\nMatrix-vector: A @ x = {A @ x}")
# [1,2] @ [1] = [1*1 + 2*2] = [5]
# [3,4]   [2]   [3*1 + 4*2]   [11]

# ──────────────────────────────────────────────────────────────────────
# 2.3 BROADCASTING
# ──────────────────────────────────────────────────────────────────────
print("\n" + "─" * 70)
print("2.3 BROADCASTING (NumPy's Superpower)")
print("─" * 70)

# Broadcasting lets NumPy do operations on arrays of different shapes
# without needing explicit loops. This is how ML code stays fast.

# Broadcasting rules:
# 1. If arrays have different ndim, pad the smaller shape with 1s on the left
# 2. Dimensions of size 1 are stretched to match the other array
# 3. If dimensions don't match and neither is 1 -> error

A = np.array([[1, 2, 3],
              [4, 5, 6]])  # shape (2, 3)

# Case 1: Scalar broadcast (adds to every element)
print(f"A:\n{A}\n")
print(f"A + 10 =\n{A + 10}\n")

# Case 2: Row vector broadcast (adds to each row)
row = np.array([10, 20, 30])  # shape (3,) -> broadcasts as (1, 3)
print(f"Row vector: {row}")
print(f"A + row =\n{A + row}\n")
# What happened:
# [[1,2,3],   +  [[10,20,30],   =  [[11,22,33],
#  [4,5,6]]       [10,20,30]]       [14,25,36]]

# Case 3: Column vector broadcast (adds to each column)
col = np.array([[100], [200]])  # shape (2, 1)
print(f"Column vector:\n{col}")
print(f"A + col =\n{A + col}\n")
# What happened:
# [[1,2,3],   +  [[100,100,100],   =  [[101,102,103],
#  [4,5,6]]       [200,200,200]]       [204,205,206]]

# ─── ML Application: Feature Scaling ─────────────────────────────────

print("--- ML Application: Feature Scaling with Broadcasting ---")

# Simulating a dataset: 5 samples, 3 features
# Features have very different scales (common in real data)
np.random.seed(42)
data = np.array([
    [2500, 3, 15],     # [house_area, bedrooms, age]
    [1800, 2, 25],
    [3200, 4, 5],
    [1500, 1, 30],
    [2800, 3, 10]
], dtype=float)

print(f"Raw data:\n{data}\n")

# Standardization: (x - mean) / std  (zero mean, unit variance)
# This uses broadcasting twice!
mean = data.mean(axis=0)  # mean of each column -> shape (3,)
std = data.std(axis=0)    # std of each column -> shape (3,)

# shape: (5,3) - (3,) -> broadcasts to (5,3) - (1,3) -> (5,3)
data_standardized = (data - mean) / std

print(f"Column means: {mean}")
print(f"Column stds:  {std}")
print(f"\nStandardized data:\n{data_standardized}")
print(f"  Mean of each column (≈0): {data_standardized.mean(axis=0).round(6)}")
print(f"  Std of each column (≈1):  {data_standardized.std(axis=0).round(6)}")

# Min-Max scaling: (x - min) / (max - min)  (scales to [0, 1])
min_val = data.min(axis=0)
max_val = data.max(axis=0)
data_minmax = (data - min_val) / (max_val - min_val)

print(f"\nMin-Max scaled data:\n{data_minmax}")
print(f"  Min of each column: {data_minmax.min(axis=0)}")
print(f"  Max of each column: {data_minmax.max(axis=0)}")

# ─── Broadcasting Error Example ─────────────────────────────────────

print("\n--- Broadcasting Shape Compatibility ---")
a = np.ones((3, 4))
b = np.ones((3, 1))
c = np.ones((1, 4))

print(f"(3,4) + (3,1) -> works! Result shape: {(a + b).shape}")
print(f"(3,4) + (1,4) -> works! Result shape: {(a + c).shape}")

# This would fail: np.ones((3,4)) + np.ones((2,4)) -> shapes (3,4) and (2,4)
# because 3 != 2 and neither is 1
print(f"\n(3,4) + (2,4) -> would FAIL! 3 != 2 and neither is 1.")

# ─── Outer Product via Broadcasting ──────────────────────────────────

# The outer product creates a matrix from two vectors
a = np.array([1, 2, 3])
b = np.array([4, 5])

outer = a.reshape(-1, 1) * b.reshape(1, -1)  # (3,1) * (1,2) -> (3,2)
outer_np = np.outer(a, b)  # equivalent

print(f"\nOuter product of {a} and {b}:")
print(f"  Using broadcasting:\n{outer}")
print(f"  Using np.outer:\n{outer_np}")

print("\n" + "=" * 70)
print("PART 2 COMPLETE -- You now understand matrices!")
print("Key takeaways:")
print("  - Matrix multiplication: (m×n) @ (n×p) = (m×p)")
print("  - y = X @ W + b is linear regression AND neural network forward pass")
print("  - Broadcasting eliminates loops → fast, clean ML code")
print("  - Feature scaling uses broadcasting: (data - mean) / std")
print("=" * 70)
