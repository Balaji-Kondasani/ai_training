import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score, classification_report

def calculate_gini(labels):
    """
    Computes Gini Impurity for a list of binary or multiclass labels:
    Gini = 1 - sum(p_i ^ 2)
    """
    if len(labels) == 0:
        return 0
    counts = np.bincount(labels)
    probabilities = counts / len(labels)
    return 1 - np.sum(probabilities ** 2)

def calculate_entropy(labels):
    """
    Computes Entropy for a list of binary or multiclass labels:
    Entropy = - sum(p_i * log2(p_i))
    """
    if len(labels) == 0:
        return 0
    counts = np.bincount(labels)
    probabilities = counts / len(labels)
    # Filter out probabilities that are 0 to avoid log2(0) error
    probabilities = probabilities[probabilities > 0]
    return -np.sum(probabilities * np.log2(probabilities))

def main():
    print("=== Day 3: Decision Trees ===")
    
    # 1. Concept Demonstration (Gini and Entropy)
    toy_labels_pure = np.array([1, 1, 1, 1])
    toy_labels_mixed = np.array([1, 1, 0, 0])
    
    print("Toy Set Pure [1, 1, 1, 1]:")
    print(f"  Gini Impurity: {calculate_gini(toy_labels_pure):.4f}")
    print(f"  Entropy: {calculate_entropy(toy_labels_pure):.4f}")
    
    print("Toy Set Mixed [1, 1, 0, 0]:")
    print(f"  Gini Impurity: {calculate_gini(toy_labels_mixed):.4f}")
    print(f"  Entropy: {calculate_entropy(toy_labels_mixed):.4f}")
    
    # 2. Load Breast Cancer Dataset
    data = load_breast_cancer()
    X, y = data.data, data.target
    feature_names = data.feature_names
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # 3. Train Decision Tree Classifier (Fully grown / prone to overfitting)
    clf_full = DecisionTreeClassifier(random_state=42)
    clf_full.fit(X_train, y_train)
    
    # 4. Train Decision Tree Classifier with Tuning (Regularized)
    clf_tuned = DecisionTreeClassifier(
        criterion="gini",
        max_depth=3,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42
    )
    clf_tuned.fit(X_train, y_train)
    
    # 5. Compare Results
    y_pred_full = clf_full.predict(X_test)
    y_pred_tuned = clf_tuned.predict(X_test)
    
    print("\n--- Model Performance Comparison ---")
    print(f"Fully Grown Tree - Train Accuracy: {clf_full.score(X_train, y_train):.4f}")
    print(f"Fully Grown Tree - Test Accuracy : {accuracy_score(y_test, y_pred_full):.4f}")
    print(f"Tuned Tree (depth=3) - Train Accuracy: {clf_tuned.score(X_train, y_train):.4f}")
    print(f"Tuned Tree (depth=3) - Test Accuracy : {accuracy_score(y_test, y_pred_tuned):.4f}")
    
    # 6. Export Decision Tree Structure as Text
    print("\n--- Decision Tree Structure (Tuned Model) ---")
    tree_rules = export_text(clf_tuned, feature_names=list(feature_names))
    print(tree_rules)

if __name__ == "__main__":
    main()
