import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def main():
    print("=== Day 4: SVM & KNN ===")
    
    # 1. Load Dataset
    data = load_breast_cancer()
    X, y = data.data, data.target
    
    # Split into train/test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # 2. Scale Features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("--- Demonstrating Importance of Feature Scaling ---")
    
    # Train KNN without scaling
    knn_unscaled = KNeighborsClassifier(n_neighbors=5)
    knn_unscaled.fit(X_train, y_train)
    y_pred_knn_unscaled = knn_unscaled.predict(X_test)
    print(f"KNN (K=5) without Scaling - Accuracy: {accuracy_score(y_test, y_pred_knn_unscaled):.4f}")
    
    # Train KNN with scaling
    knn_scaled = KNeighborsClassifier(n_neighbors=5)
    knn_scaled.fit(X_train_scaled, y_train)
    y_pred_knn_scaled = knn_scaled.predict(X_test_scaled)
    print(f"KNN (K=5) with Scaling    - Accuracy: {accuracy_score(y_test, y_pred_knn_scaled):.4f}")
    
    # Train SVM without scaling
    svm_unscaled = SVC(kernel='rbf', random_state=42)
    svm_unscaled.fit(X_train, y_train)
    y_pred_svm_unscaled = svm_unscaled.predict(X_test)
    print(f"SVM (RBF) without Scaling - Accuracy: {accuracy_score(y_test, y_pred_svm_unscaled):.4f}")
    
    # Train SVM with scaling
    svm_scaled = SVC(kernel='rbf', random_state=42)
    svm_scaled.fit(X_train_scaled, y_train)
    y_pred_svm_scaled = svm_scaled.predict(X_test_scaled)
    print(f"SVM (RBF) with Scaling    - Accuracy: {accuracy_score(y_test, y_pred_svm_scaled):.4f}")
    
    print("\n--- Hyperparameter / Model Architecture Tuning (Scaled Data) ---")
    
    # SVM Linear Kernel
    svm_linear = SVC(kernel='linear', random_state=42)
    svm_linear.fit(X_train_scaled, y_train)
    print(f"SVM (Linear Kernel)       - Accuracy: {accuracy_score(y_test, svm_linear.predict(X_test_scaled)):.4f}")
    
    # SVM RBF Kernel
    svm_rbf = SVC(kernel='rbf', C=10.0, gamma='scale', random_state=42)
    svm_rbf.fit(X_train_scaled, y_train)
    print(f"SVM (RBF Kernel, C=10)    - Accuracy: {accuracy_score(y_test, svm_rbf.predict(X_test_scaled)):.4f}")
    
    # KNN K=1
    knn_1 = KNeighborsClassifier(n_neighbors=1)
    knn_1.fit(X_train_scaled, y_train)
    print(f"KNN (K=1)                 - Accuracy: {accuracy_score(y_test, knn_1.predict(X_test_scaled)):.4f}")
    
    # KNN K=15
    knn_15 = KNeighborsClassifier(n_neighbors=15)
    knn_15.fit(X_train_scaled, y_train)
    print(f"KNN (K=15)                - Accuracy: {accuracy_score(y_test, knn_15.predict(X_test_scaled)):.4f}")

if __name__ == "__main__":
    main()
