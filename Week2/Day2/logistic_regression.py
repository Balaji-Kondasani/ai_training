import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

def sigmoid(z):
    """
    Sigmoid activation function maps any real value into the range [0, 1].
    """
    return 1 / (1 + np.exp(-z))

def compute_loss(y_true, y_pred_prob):
    """
    Computes Binary Cross-Entropy (Log Loss).
    """
    epsilon = 1e-15  # Avoid log(0)
    y_pred_prob = np.clip(y_pred_prob, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred_prob) + (1 - y_true) * np.log(1 - y_pred_prob))

def main():
    print("=== Day 2: Logistic Regression ===")
    
    # 1. Generate Synthetic Binary Classification Dataset
    X, y = make_classification(
        n_samples=200, n_features=2, n_informative=2, n_redundant=0, 
        n_clusters_per_class=1, random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    
    # 2. Demonstrate Sigmoid & Loss concepts
    sample_logits = np.array([-2.0, 0.0, 2.0])
    probabilities = sigmoid(sample_logits)
    print(f"Sigmoid outputs for logits {sample_logits}: {probabilities}")
    
    # 3. Train Logistic Regression using Scikit-Learn
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    # Make Predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # 4. Evaluation Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    bce_loss = compute_loss(y_test, y_prob)
    
    print("\n--- Evaluation Metrics ---")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"Binary Cross-Entropy Loss (Log Loss): {bce_loss:.4f}\n")
    
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # 5. Save Visualization (Decision Boundary)
    plt.figure(figsize=(8, 6))
    
    # Plot data points
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='bwr', edgecolors='k', alpha=0.6, label='Train Data')
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='bwr', edgecolors='k', marker='X', s=100, label='Test Data')
    
    # Calculate decision boundary line: w0*x0 + w1*x1 + b = 0 -> x1 = -(w0*x0 + b) / w1
    coef = model.coef_[0]
    intercept = model.intercept_[0]
    x0_vals = np.linspace(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5, 100)
    x1_vals = -(coef[0] * x0_vals + intercept) / coef[1]
    
    plt.plot(x0_vals, x1_vals, color='black', linestyle='--', linewidth=2, label='Decision Boundary')
    plt.title("Logistic Regression Decision Boundary")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.grid(True)
    
    import os
    os.makedirs("plots", exist_ok=True)
    plot_path = os.path.join("plots", "day_02_logistic_regression.png")
    plt.savefig(plot_path)
    print(f"Saved decision boundary plot to: {plot_path}")
    plt.close()

if __name__ == "__main__":
    main()
