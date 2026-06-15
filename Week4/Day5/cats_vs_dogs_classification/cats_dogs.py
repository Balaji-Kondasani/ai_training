import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Sequential

def generate_synthetic_image_dataset(n_samples=200, img_height=64, img_width=64):
    """
    Generates synthetic image tensors (n_samples, height, width, channels)
    representing mock cats (channel 0 dominant) vs dogs (channel 1 dominant).
    """
    np.random.seed(42)
    # Generate random colored noise matrices
    X = np.random.uniform(0.0, 1.0, size=(n_samples, img_height, img_width, 3)).astype('float32')
    y = np.random.choice([0, 1], size=n_samples)  # 0 = Cat, 1 = Dog
    
    # Modify images based on labels so the CNN has something meaningful to learn
    for i in range(n_samples):
        if y[i] == 0:
            # Cats: Add a mock circular pattern in the red/green channels (centered circle)
            for r in range(20, 44):
                for c in range(20, 44):
                    if (r - 32)**2 + (c - 32)**2 < 144:
                        X[i, r, c, 0] = 0.9  # Dominant red
        else:
            # Dogs: Add a mock square pattern in the blue channel (centered square)
            X[i, 20:44, 20:44, 2] = 0.9  # Dominant blue
            
    return X, y

def main():
    print("=== Day 10: Project 4 - Cats vs Dogs Classification ===")
    
    # 1. Generate Synthetic Dataset (Avoids downloading 800MB Kaggle images)
    img_height, img_width = 64, 64
    print("Generating synthetic image classification dataset...")
    X, y = generate_synthetic_image_dataset(n_samples=300, img_height=img_height, img_width=img_width)
    
    # Split train/test
    split = 240
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    print(f"Generated data:")
    print(f"  Training shape: {X_train.shape} | Labels: {len(y_train)}")
    print(f"  Testing shape:  {X_test.shape} | Labels: {len(y_test)}")
    
    # 2. Build Data Augmentation layers
    # Data augmentation expands training diversity by modifying images randomly on the fly
    data_augmentation = Sequential([
        layers.RandomFlip("horizontal", input_shape=(img_height, img_width, 3)),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ], name="data_augmentation")
    
    # 3. Build CNN Model
    model = Sequential([
        # Data Augmentation Layer
        data_augmentation,
        
        # Conv block 1
        layers.Conv2D(16, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Conv block 2
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Conv block 3
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Dense blocks
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='sigmoid')  # Binary classification
    ])
    
    # 4. Compile Model
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    model.summary()
    
    # 5. Train Model
    print("\nTraining CNN classifier with Data Augmentation...")
    model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=8,
        batch_size=32,
        verbose=1
    )
    
    # 6. Evaluation
    print("\nEvaluating on Test Set...")
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Accuracy: {accuracy*100:.2f}%")
    
    # Sample Predict
    sample = X_test[0].reshape(1, img_height, img_width, 3)
    prob = model.predict(sample, verbose=0)[0][0]
    pred = "Dog" if prob >= 0.5 else "Cat"
    true = "Dog" if y_test[0] == 1 else "Cat"
    print(f"Sample Prediction -> True: {true} | Predicted: {pred} (Confidence: {prob if prob >= 0.5 else 1 - prob:.4%})")

if __name__ == "__main__":
    main()
