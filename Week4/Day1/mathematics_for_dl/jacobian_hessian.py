import numpy as np

def compute_jacobian(x):
    """
    Computes the Jacobian matrix for the multi-variable function:
    f(x1, x2) = [x1^2 + x2, x2^2 - x1]
    
    J = [[df1/dx1, df1/dx2],
         [df2/dx1, df2/dx2]]
      = [[2*x1,   1],
         [-1,     2*x2]]
    """
    x1, x2 = x[0], x[1]
    return np.array([
        [2 * x1, 1.0],
        [-1.0, 2 * x2]
    ])

def compute_hessian(x):
    """
    Computes the Hessian matrix for the scalar function:
    f(x1, x2) = x1^3 + x1*x2 + x2^2
    
    H = [[d2f/dx1^2,   d2f/dx1dx2],
         [d2f/dx2dx1,  d2f/dx2^2]]
      = [[6*x1,       1],
         [1,          2]]
    """
    x1, x2 = x[0], x[1]
    return np.array([
        [6 * x1, 1.0],
        [1.0, 2.0]
    ])

def main():
    print("=== Day 2: Jacobian and Hessian Matrices ===")
    point = np.array([2.0, 3.0])
    print(f"Given point: (x1={point[0]}, x2={point[1]})")
    print(f"Jacobian Matrix at point:\n{compute_jacobian(point)}")
    print(f"Hessian Matrix at point:\n{compute_hessian(point)}")
    print("\nConcepts:")
    print("  * Jacobian: First-order partial derivatives. Used in backpropagation to map updates between activation dimensions.")
    print("  * Hessian: Second-order partial derivatives. Represents local curvature of loss landscape; used in advanced second-order optimizers.")

if __name__ == "__main__":
    main()
