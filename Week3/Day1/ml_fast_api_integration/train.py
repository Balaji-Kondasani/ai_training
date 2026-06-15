import os
import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

def main():
    print("=== Training Day 14 Model ===")
    
    # 1. Load Breast Cancer Dataset
    data = load_breast_cancer()
    X = data.data
    y = data.target
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 2. Build Pipeline
    # Include Scaling + Model in a single pipeline to ensure inputs are scaled correctly during inference
    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42))
        ]
    )
    
    # Train
    pipeline.fit(X_train, y_train)
    score = pipeline.score(X_test, y_test)
    print(f"Model trained successfully. Test Accuracy: {score*100:.2f}%")
    
    # 3. Save model inside this day's directory
    os.makedirs("model", exist_ok=True)
    model_path = os.path.join("model", "cancer_prediction_pipeline.joblib")
    joblib.dump(pipeline, model_path)
    print(f"Model saved to: {model_path}")

if __name__ == "__main__":
    main()
