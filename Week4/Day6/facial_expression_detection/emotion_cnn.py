import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Sequential

def generate_synthetic_facial_data(n_samples=300, img_size=48):
    """
    Generates synthetic grayscale facial images (n_samples, 48, 48, 1)
    and class labels (0: Happy, 1: Sad, 2: Angry, 3: Neutral).
    """
    np.random.seed(42)
    X = np.random.uniform(0.1, 0.9, size=(n_samples, img_size, img_size, 1)).astype('float32')
    y = np.random.choice([0, 1, 2, 3], size=n_samples)
    
    # Let's add patterns to mock mouth curvatures for different emotions
    for i in range(n_samples):
        # Center coordinates
        cy, cx = img_size // 2, img_size // 2
        
        # Add eyes (constant across emotions)
        X[i, cy - 8: cy - 5, cx - 10: cx - 7] = 0.9  # Left Eye
        X[i, cy - 8: cy - 5, cx + 7: cx + 10] = 0.9  # Right Eye
        
        if y[i] == 0:
            # Happy: Smile mouth pattern (upward curve)
            X[i, cy + 8, cx - 6: cx + 7] = 0.95
            X[i, cy + 7, [cx - 7, cx + 7]] = 0.95
            X[i, cy + 6, [cx - 8, cx + 8]] = 0.95
        elif y[i] == 1:
            # Sad: Frown mouth pattern (downward curve)
            X[i, cy + 6, cx - 6: cx + 7] = 0.95
            X[i, cy + 7, [cx - 7, cx + 7]] = 0.95
            X[i, cy + 8, [cx - 8, cx + 8]] = 0.95
        elif y[i] == 2:
            # Angry: Eyebrows slanted inward
            X[i, cy - 11, cx - 11: cx - 6] = 0.95  # Slanted left brow
            X[i, cy - 10, cx - 10: cx - 5] = 0.95
            X[i, cy - 11, cx + 6: cx + 11] = 0.95  # Slanted right brow
            X[i, cy - 10, cx + 5: cx + 10] = 0.95
            # Straight mouth
            X[i, cy + 7, cx - 6: cx + 7] = 0.95
        else:
            # Neutral: Straight mouth line
            X[i, cy + 7, cx - 8: cx + 9] = 0.95
            
    return X, y

def main():
    print("=== Day 11: Project 5 - Facial Expression Recognition ===")
    
    # 1. Generate Synthetic Data
    img_size = 48
    X, y = generate_synthetic_facial_data(n_samples=400, img_size=img_size)
    
    # Split
    split = 320
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    print(f"Dataset summary:")
    print(f"  Training shape: {X_train.shape} | Labels count: {len(y_train)}")
    print(f"  Testing shape:  {X_test.shape} | Labels count: {len(y_test)}")
    
    # 2. Build Deep CNN Architecture
    # Features BatchNormalization to stabilize training dynamics and accelerate convergence
    model = Sequential([
        layers.Input(shape=(img_size, img_size, 1)),
        
        # Conv block 1
        layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2),
        
        # Conv block 2
        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),
        
        # Classification dense block
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.Dense(4, activation='softmax')  # 4 classes: happy, sad, angry, neutral
    ])
    
    # 3. Compile Model
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    model.summary()
    
    # 4. Train Model
    print("\nTraining Deep CNN Facial Expression Classifier...")
    model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=10,
        batch_size=32,
        verbose=1
    )
    
    # 5. Evaluate
    print("\nEvaluating on Test Set...")
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Accuracy: {accuracy*100:.2f}%")
    
    # Sample Prediction
    emotions = ["Happy", "Sad", "Angry", "Neutral"]
    sample_img = X_test[0].reshape(1, img_size, img_size, 1)
    probs = model.predict(sample_img, verbose=0)[0]
    pred_idx = np.argmax(probs)
    print(f"Prediction result -> True Emotion: {emotions[y_test[0]]} | Predicted: {emotions[pred_idx]} (Confidence: {probs[pred_idx]:.2%})")

if __name__ == "__main__":
    main()
