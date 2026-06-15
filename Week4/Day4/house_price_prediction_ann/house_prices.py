import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras import layers, Sequential

def generate_synthetic_housing_data(n_samples=1000):
    """
    Generates structured housing prices dataset.
    """
    np.random.seed(42)
    sqft = np.random.normal(2000, 500, size=n_samples)
    sqft = np.clip(sqft, 800, 5000)
    
    rooms = np.round(sqft / 500 + np.random.normal(0, 0.5, size=n_samples))
    rooms = np.clip(rooms, 1, 8).astype(int)
    
    age = np.random.randint(0, 60, size=n_samples)  # Years since build
    neighborhood_quality = np.random.uniform(1, 10, size=n_samples)  # Rating 1-10
    
    # Calculate target price: Price (in thousands of dollars)
    price = (sqft * 0.15) + (rooms * 25) - (age * 1.2) + (neighborhood_quality * 15) + np.random.normal(0, 30, size=n_samples)
    price = np.clip(price, 50, None)  # Min price $50k
    
    df = pd.DataFrame({
        "SqFt": sqft,
        "Rooms": rooms,
        "Age": age,
        "NeighborhoodQuality": neighborhood_quality,
        "Price": price
    })
    return df

def main():
    print("=== Day 7: Project 2 - House Price Prediction ===")
    
    # 1. Load/Generate Data
    df = generate_synthetic_housing_data()
    print(f"Dataset generated: {len(df)} samples")
    print(f"House Price Summary (in $k):\n{df['Price'].describe()}\n")
    
    # 2. Preprocess Data
    X = df.drop(columns=["Price"])
    y = df["Price"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 3. Build Keras ANN Regression Model
    model = Sequential([
        layers.Input(shape=(X_train_scaled.shape[1],)),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(16, activation='relu'),
        layers.Dense(1)  # Single linear node for regression
    ])
    
    # 4. Compile Model
    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae']
    )
    
    # 5. Train Model
    print("Training House Price Prediction Regression ANN...")
    model.fit(
        X_train_scaled, y_train,
        validation_split=0.1,
        epochs=30,
        batch_size=32,
        verbose=1
    )
    
    # 6. Evaluation
    print("\nEvaluating on Test Set...")
    y_pred = model.predict(X_test_scaled).flatten()
    
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Root Mean Squared Error (RMSE): ${rmse:.2f}k")
    print(f"R-squared (R2 Score): {r2:.4f}")
    
    # Save a comparison sample of true vs predicted prices
    compare_df = pd.DataFrame({
        "True Price ($k)": y_test,
        "Predicted Price ($k)": y_pred
    }).head(10)
    print("\nSample Comparisons:")
    print(compare_df.round(2))

if __name__ == "__main__":
    main()
