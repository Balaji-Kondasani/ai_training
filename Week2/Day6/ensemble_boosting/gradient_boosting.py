from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def main():
    print("=== Day 11: Ensemble Techniques - Boosting (Gradient Boosting) ===")
    
    # 1. Load Dataset
    data = load_breast_cancer()
    X, y = data.data, data.target
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # 2. Train Gradient Boosting Classifier
    # Gradient boosting trains models sequentially, minimizing pseudo-residuals of loss functions.
    gb = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,  # Shrinkage factor (controls contribution of each tree)
        max_depth=3,        # Depth of individual decision stumps/trees
        random_state=42
    )
    gb.fit(X_train, y_train)
    
    # 3. Model Evaluation
    y_pred = gb.predict(X_test)
    gb_accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Gradient Boosting Test Accuracy: {gb_accuracy:.4f}")
    print("\nGradient Boosting Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # 4. Compare Bagging vs Boosting
    print("--- Comparing Bagging vs Boosting (on Breast Cancer Dataset) ---")
    
    # Random Forest (Bagging)
    rf = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42)
    rf.fit(X_train, y_train)
    rf_accuracy = accuracy_score(y_test, rf.predict(X_test))
    
    print(f"  Random Forest (Bagging)  Test Accuracy: {rf_accuracy:.4f}")
    print(f"  Gradient Boosting (Boosting) Test Accuracy: {gb_accuracy:.4f}")
    
    print("\nSummary of Concepts:")
    print("  * Bagging (Random Forest) trains trees in PARALLEL. Each tree is independent and gets a bootstrap sample. It aims to reduce VARIANCE.")
    print("  * Boosting (Gradient Boosting) trains trees SEQUENTIALLY. Each tree learns from the mistakes (residuals) of previous trees. It aims to reduce BIAS.")

if __name__ == "__main__":
    main()
