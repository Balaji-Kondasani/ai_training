import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def main():
    print("=== Day 10: Ensemble Techniques - Bagging (Random Forest) ===")
    
    # 1. Load Dataset
    data = load_breast_cancer()
    X, y = data.data, data.target
    feature_names = data.feature_names
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # 2. Train Random Forest Classifier
    # We enable `oob_score=True` to compute the out-of-bag error estimation
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        oob_score=True,
        random_state=42
    )
    rf.fit(X_train, y_train)
    
    # 3. Model Evaluation
    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    oob_accuracy = rf.oob_score_
    
    print(f"Random Forest Test Accuracy: {accuracy:.4f}")
    print(f"Out-of-Bag (OOB) Validation Accuracy: {oob_accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # 4. Feature Importance Analysis
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    print("Top 5 Most Important Features:")
    for rank in range(5):
        idx = indices[rank]
        print(f"  {rank + 1}. {feature_names[idx]}: importance={importances[idx]:.4f}")
        
    # 5. Save Feature Importance Plot
    plt.figure(figsize=(10, 6))
    plt.title("Random Forest - Feature Importances (Top 10)")
    top_n = 10
    plt.bar(range(top_n), importances[indices[:top_n]], color='skyblue', align='center')
    plt.xticks(range(top_n), [feature_names[i] for i in indices[:top_n]], rotation=45, ha='right')
    plt.xlabel("Features")
    plt.ylabel("Relative Importance")
    plt.tight_layout()
    
    import os
    os.makedirs("plots", exist_ok=True)
    plot_path = os.path.join("plots", "day_10_random_forest.png")
    plt.savefig(plot_path)
    print(f"\nSaved feature importance plot to: {plot_path}")
    plt.close()

if __name__ == "__main__":
    main()
