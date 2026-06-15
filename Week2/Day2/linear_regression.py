import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def simple_linear_regression_scratch(x, y):
    """
    Computes Simple Linear Regression using closed-form analytical formulas:
    y = m * x + c
    m = cov(x, y) / var(x)
    c = mean(y) - m * mean(x)
    """
    n = len(x)
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    
    # Calculate terms for slope (m) and intercept (c)
    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sum((x - x_mean) ** 2)
    
    m = numerator / denominator
    c = y_mean - m * x_mean
    return m, c

def main():
    print("=== Day 1: Linear Regression ===")
    
    # 1. Generate Synthetic Data
    np.random.seed(42)
    x = 2 * np.random.rand(100, 1)
    y = 4 + 3 * x + np.random.randn(100, 1)  # y = 3x + 4 + noise
    
    # Flatten x and y for scratch implementation
    x_flat = x.flatten()
    y_flat = y.flatten()
    
    # 2. Simple Linear Regression from Scratch
    m, c = simple_linear_regression_scratch(x_flat, y_flat)
    print(f"[Scratch Simple Regression] Calculated line: y = {m:.4f} * x + {c:.4f}")
    
    # 3. Multiple Linear Regression using Scikit-Learn
    # Add a second feature: x2 = 0.5 * x + random noise
    x2 = 0.5 * x + np.random.randn(100, 1)
    X_mult = np.hstack((x, x2))  # Matrix with 2 features
    
    model = LinearRegression()
    model.fit(X_mult, y)
    
    # Coefficients & Intercept
    coef = model.coef_[0]
    intercept = model.intercept_[0]
    print(f"[Scikit-Learn Multiple Regression] Intercept: {intercept:.4f}")
    print(f"[Scikit-Learn Multiple Regression] Coefficients: Feature 1={coef[0]:.4f}, Feature 2={coef[1]:.4f}")
    
    # 4. Evaluation Metrics
    y_pred = model.predict(X_mult)
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"R-squared (R2 Score): {r2:.4f}")
    
    # 5. Save Visualization (Simple Regression Line)
    plt.figure(figsize=(8, 6))
    plt.scatter(x_flat, y_flat, color='blue', label='Data points')
    plt.plot(x_flat, m * x_flat + c, color='red', linewidth=2, label=f'Scratch Fit (y={m:.2f}x+{c:.2f})')
    plt.title("Simple Linear Regression (Scratch)")
    plt.xlabel("X")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)
    
    # Save the plot image
    import os
    os.makedirs("plots", exist_ok=True)
    plot_path = os.path.join("plots", "day_01_linear_regression.png")
    plt.savefig(plot_path)
    print(f"Saved visualization plot to: {plot_path}")
    plt.close()

if __name__ == "__main__":
    main()
