import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
import tensorflow as tf
from tensorflow.keras import layers, Sequential

def generate_synthetic_churn_data(n_samples=1000):
    """
    Generates structured churn dataset containing customer features.
    """
    np.random.seed(42)
    # Features
    age = np.random.randint(18, 80, size=n_samples)
    tenure = np.random.randint(1, 72, size=n_samples)  # Months
    monthly_charges = np.random.uniform(20.0, 120.0, size=n_samples)
    total_charges = monthly_charges * tenure + np.random.normal(0, 50, size=n_samples)
    total_charges = np.clip(total_charges, 20.0, None)
    
    # Categorical: Contract (0 = Month-to-month, 1 = One year, 2 = Two year)
    contract = np.random.choice([0, 1, 2], size=n_samples, p=[0.5, 0.3, 0.2])
    
    # Calculate churn probability based on features
    # Older age, higher monthly charges, month-to-month contract -> higher churn probability
    logit = (age * 0.02) - (tenure * 0.05) + (monthly_charges * 0.01) + (contract * -1.5) - 0.5
    prob = 1 / (1 + np.exp(-logit))
    churn = (np.random.rand(n_samples) < prob).astype(int)
    
    df = pd.DataFrame({
        "Age": age,
        "Tenure": tenure,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
        "Contract": contract,
        "Churn": churn
    })
    return df

def main():
    print("=== Day 6: Project 1 - Customer Churn Prediction ===")
    
    # 1. Load/Generate Data
    df = generate_synthetic_churn_data()
    print(f"Dataset generated: {len(df)} samples")
    print(f"Churn Distribution:\n{df['Churn'].value_counts(normalize=True)}")
    
    # 2. Preprocess Data
    X = df.drop(columns=["Churn"])
    y = df["Churn"]
    
    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Standardize numerical features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 3. Build Keras ANN Model
    model = Sequential([
        layers.Input(shape=(X_train_scaled.shape[1],)),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.2),  # Regularization
        layers.Dense(16, activation='relu'),
        layers.Dense(1, activation='sigmoid')  # Binary output
    ])
    
    # 4. Compile Model
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')]
    )
    
    # 5. Train Model
    print("\nTraining Customer Churn Classifier ANN...")
    model.fit(
        X_train_scaled, y_train,
        validation_split=0.1,
        epochs=15,
        batch_size=32,
        verbose=1
    )
    
    # 6. Evaluation
    print("\nEvaluating on Test Set...")
    y_pred_prob = model.predict(X_test_scaled).flatten()
    y_pred = (y_pred_prob >= 0.5).astype(int)
    
    auc = roc_auc_score(y_test, y_pred_prob)
    print(f"Area Under ROC Curve (AUC-ROC): {auc:.4f}\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    main()
