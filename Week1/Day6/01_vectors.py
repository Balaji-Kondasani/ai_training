"""
=============================================================================
PART 1: VECTORS -- The Foundation of Everything in ML
=============================================================================

Every data sample in your dataset is a vector.
A dataset with 1000 samples and 10 features = 1000 vectors in 10D space.

Topics covered:
  1.1 Vector Basics (creation, shapes, types)
  1.2 Vector Operations (add, subtract, dot product, Hadamard)
  1.3 Vector Norms (L1, L2, L-inf, normalization)
  1.4 Dot Product Deep Dive + Cosine Similarity
"""

import numpy as np

print("=" * 70)
print("PART 1: VECTORS -- The Foundation of Everything in ML")
print("=" * 70)

# ──────────────────────────────────────────────────────────────────────
# 1.1 VECTOR BASICS
# ──────────────────────────────────────────────────────────────────────
print("\n" + "─" * 70)
print("1.1 VECTOR BASICS")
print("─" * 70)

# A vector is just a list of numbers. In ML, it represents a data point.
# For example, a house might be described by: [area, bedrooms, price]

# Creating vectors in NumPy
v1 = np.array([1, 2, 3])  # 1D array -- the most common way
print(f"v1 = {v1}")
print(f"  shape: {v1.shape}")  # (3,) -- a 1D array with 3 elements
print(f"  ndim:  {v1.ndim}")   # 1 -- one dimension
print(f"  size:  {v1.size}")   # 3 -- three elements
print(f"  dtype: {v1.dtype}")  # int64 or int32 depending on OS

# Row vector vs Column vector (matters for matrix multiplication)
v_row = np.array([[1, 2, 3]])      # shape (1, 3) -- 1 row, 3 columns
v_col = np.array([[1], [2], [3]])  # shape (3, 1) -- 3 rows, 1 column

print(f"\nRow vector: {v_row}, shape = {v_row.shape}")
print(f"Column vector:\n{v_col}\n  shape = {v_col.shape}")

# Converting between row and column using reshape or transpose
v_as_col = v1.reshape(-1, 1)  # -1 means "figure out this dimension"
v_as_row = v1.reshape(1, -1)
print(f"\nv1 reshaped to column:\n{v_as_col}, shape = {v_as_col.shape}")
print(f"v1 reshaped to row: {v_as_row}, shape = {v_as_row.shape}")

# Scalar vs Vector vs Matrix vs Tensor
scalar = np.array(5)           # 0D -- a single number
vector = np.array([1, 2, 3])   # 1D -- a list of numbers
matrix = np.array([[1, 2],
                   [3, 4]])    # 2D -- a table of numbers
tensor = np.array([[[1, 2],
                    [3, 4]],
                   [[5, 6],
                    [7, 8]]])  # 3D -- a "cube" of numbers

print(f"\nScalar: ndim={scalar.ndim}, shape={scalar.shape}")
print(f"Vector: ndim={vector.ndim}, shape={vector.shape}")
print(f"Matrix: ndim={matrix.ndim}, shape={matrix.shape}")
print(f"Tensor: ndim={tensor.ndim}, shape={tensor.shape}")

# Special vectors
zeros = np.zeros(5)          # [0, 0, 0, 0, 0]
ones = np.ones(5)            # [1, 1, 1, 1, 1]
range_v = np.arange(0, 10, 2)  # [0, 2, 4, 6, 8]
linspace_v = np.linspace(0, 1, 5)  # [0, 0.25, 0.5, 0.75, 1.0]
random_v = np.random.randn(5)  # 5 random numbers from standard normal

print(f"\nZeros:    {zeros}")
print(f"Ones:     {ones}")
print(f"Arange:   {range_v}")
print(f"Linspace: {linspace_v}")
print(f"Random:   {random_v}")

# ──────────────────────────────────────────────────────────────────────
# 1.2 VECTOR OPERATIONS
# ──────────────────────────────────────────────────────────────────────
print("\n" + "─" * 70)
print("1.2 VECTOR OPERATIONS")
print("─" * 70)

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# Element-wise addition and subtraction
print(f"a = {a}")
print(f"b = {b}")
print(f"a + b = {a + b}")  # [5, 7, 9] -- each element added
print(f"a - b = {a - b}")  # [-3, -3, -3]

# Scalar multiplication (scales every element)
print(f"3 * a = {3 * a}")  # [3, 6, 9]

# Element-wise multiplication (Hadamard product)
# NOT the same as the dot product!
print(f"a * b (element-wise) = {a * b}")  # [4, 10, 18]

# DOT PRODUCT -- the single most important operation in ML
# Formula: sum of element-wise products
# Geometric meaning: measures how much two vectors "agree" in direction
dot_manual = np.sum(a * b)  # 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
dot_numpy = np.dot(a, b)    # same thing, using NumPy
dot_at = a @ b               # same thing, modern syntax (preferred)

print(f"\nDot product (manual):  {dot_manual}")
print(f"Dot product (np.dot): {dot_numpy}")
print(f"Dot product (a @ b):  {dot_at}")

# ML CONNECTION: Linear regression prediction is just a dot product
# prediction = weights @ features + bias
weights = np.array([0.5, -0.3, 0.8])  # learned model weights
features = np.array([3.0, 1.5, 2.0])  # one data sample
prediction = weights @ features  # 0.5*3 + (-0.3)*1.5 + 0.8*2 = 2.65
print(f"\nML Example -- Linear Regression Prediction:")
print(f"  weights  = {weights}")
print(f"  features = {features}")
print(f"  prediction = weights @ features = {prediction}")

# Cross product (3D only, less common in ML)
cross = np.cross(a, b)
print(f"\nCross product a x b = {cross}")

# ──────────────────────────────────────────────────────────────────────
# 1.3 VECTOR NORMS (Measuring Size / Distance)
# ──────────────────────────────────────────────────────────────────────
print("\n" + "─" * 70)
print("1.3 VECTOR NORMS (Measuring Size / Distance)")
print("─" * 70)

v = np.array([3, 4])
print(f"v = {v}")

# L1 Norm (Manhattan distance): sum of absolute values
# Think: walking on a grid city like Manhattan -- you can only go
# along streets (horizontal) and avenues (vertical).
l1 = np.linalg.norm(v, ord=1)
l1_manual = np.sum(np.abs(v))  # |3| + |4| = 7
print(f"\nL1 Norm (Manhattan):  {l1}")
print(f"  Manual calculation: |3| + |4| = {l1_manual}")
print(f"  ML use: Lasso regularization (makes weights sparse/zero)")

# L2 Norm (Euclidean distance): "straight line" distance
# Think: the hypotenuse of a right triangle.
l2 = np.linalg.norm(v, ord=2)
l2_manual = np.sqrt(np.sum(v ** 2))  # sqrt(9 + 16) = sqrt(25) = 5
print(f"\nL2 Norm (Euclidean):  {l2}")
print(f"  Manual calculation: sqrt(3² + 4²) = sqrt(25) = {l2_manual}")
print(f"  ML use: Ridge regularization (keeps weights small)")

# L-infinity Norm: maximum absolute value
linf = np.linalg.norm(v, ord=np.inf)
print(f"\nL∞ Norm: {linf}")
print(f"  Just the largest absolute element: max(|3|, |4|) = 4")

# NORMALIZATION (creating unit vectors)
# A unit vector has magnitude (L2 norm) = 1
# Used everywhere: word embeddings, feature scaling, cosine similarity
unit_v = v / np.linalg.norm(v)
print(f"\nUnit vector of {v}: {unit_v}")
print(f"  Its magnitude: {np.linalg.norm(unit_v):.6f} (should be 1.0)")

# Higher-dimensional example
v_high = np.array([1, -2, 3, -4, 5])
print(f"\nHigher-dimensional vector: {v_high}")
print(f"  L1 norm: {np.linalg.norm(v_high, ord=1):.2f}")
print(f"  L2 norm: {np.linalg.norm(v_high, ord=2):.2f}")
print(f"  L∞ norm: {np.linalg.norm(v_high, ord=np.inf):.2f}")

# Distance between two vectors (used in KNN, clustering)
p1 = np.array([1, 2])
p2 = np.array([4, 6])
euclidean_dist = np.linalg.norm(p1 - p2)
manhattan_dist = np.linalg.norm(p1 - p2, ord=1)
print(f"\nDistance from {p1} to {p2}:")
print(f"  Euclidean (L2): {euclidean_dist:.2f}")
print(f"  Manhattan (L1): {manhattan_dist:.2f}")

# ──────────────────────────────────────────────────────────────────────
# 1.4 DOT PRODUCT DEEP DIVE + COSINE SIMILARITY
# ──────────────────────────────────────────────────────────────────────
print("\n" + "─" * 70)
print("1.4 DOT PRODUCT DEEP DIVE + COSINE SIMILARITY")
print("─" * 70)

# The dot product has a geometric interpretation:
# a · b = ||a|| * ||b|| * cos(θ)
# where θ is the angle between the vectors
#
# This means:
# - If vectors point the same direction:  dot > 0 (cos(0°) = 1)
# - If vectors are perpendicular:         dot = 0 (cos(90°) = 0)
# - If vectors point opposite directions: dot < 0 (cos(180°) = -1)

a = np.array([1, 0])  # pointing right
b = np.array([0, 1])  # pointing up
c = np.array([-1, 0]) # pointing left (opposite of a)
d = np.array([1, 1])  # pointing diagonally

print(f"a = {a} (right)")
print(f"b = {b} (up)")
print(f"c = {c} (left)")
print(f"d = {d} (diagonal)")
print(f"\na · b = {a @ b} (perpendicular -> 0)")
print(f"a · c = {a @ c} (opposite -> negative)")
print(f"a · d = {a @ d} (same-ish direction -> positive)")

# COSINE SIMILARITY
# Measures the angle between two vectors, ignoring their magnitude.
# Range: -1 (opposite) to 0 (perpendicular) to 1 (identical direction)
#
# Formula: cos_sim(a, b) = (a · b) / (||a|| * ||b||)
#
# Why it matters in NLP: Document length shouldn't affect similarity.
# A long positive review and a short positive review should be "similar".

def cosine_similarity(a, b):
    """Compute cosine similarity between two vectors."""
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot_product / (norm_a * norm_b)

# Example: comparing text vectors (simplified TF-IDF-like representation)
# Vocabulary: [cat, dog, fish, bird]
doc1 = np.array([3, 0, 1, 0])  # talks about cats and fish
doc2 = np.array([2, 0, 2, 0])  # also talks about cats and fish
doc3 = np.array([0, 3, 0, 1])  # talks about dogs and birds

print(f"\n--- NLP Example: Document Similarity ---")
print(f"doc1 (cats & fish):  {doc1}")
print(f"doc2 (cats & fish):  {doc2}")
print(f"doc3 (dogs & birds): {doc3}")
print(f"\nCosine similarity(doc1, doc2) = {cosine_similarity(doc1, doc2):.4f} (similar!)")
print(f"Cosine similarity(doc1, doc3) = {cosine_similarity(doc1, doc3):.4f} (very different!)")
print(f"Cosine similarity(doc2, doc3) = {cosine_similarity(doc2, doc3):.4f} (very different!)")

# Compare with Euclidean distance
print(f"\nEuclidean distance(doc1, doc2) = {np.linalg.norm(doc1 - doc2):.4f}")
print(f"Euclidean distance(doc1, doc3) = {np.linalg.norm(doc1 - doc3):.4f}")

# Using scipy for cosine (note: it returns DISTANCE = 1 - similarity)
from scipy.spatial.distance import cosine as cosine_dist
print(f"\nUsing scipy:")
print(f"  cosine distance(doc1, doc2) = {cosine_dist(doc1, doc2):.4f}")
print(f"  cosine similarity(doc1, doc2) = {1 - cosine_dist(doc1, doc2):.4f}")

# Projection of vector a onto vector b
# proj_b(a) = (a · b / b · b) * b
a = np.array([3, 4])
b = np.array([1, 0])
proj_scalar = np.dot(a, b) / np.dot(b, b)
proj_vector = proj_scalar * b
print(f"\n--- Vector Projection ---")
print(f"a = {a}, b = {b}")
print(f"Projection of a onto b: {proj_vector}")
print(f"  (The 'shadow' of a on the x-axis is [{a[0]}, 0])")

print("\n" + "=" * 70)
print("PART 1 COMPLETE -- You now understand vectors!")
print("Key takeaways:")
print("  - Vectors are the building blocks of ML data")
print("  - Dot product is THE core operation (used in every ML model)")
print("  - L1/L2 norms → regularization (Lasso/Ridge)")
print("  - Cosine similarity → text/document similarity in NLP")
print("=" * 70)
