import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Sequential

def main():
    print("=== Day 9: Project 3 - MNIST Digit Classification ===")
    
    # 1. Load MNIST Dataset
    # MNIST consists of 70,000 28x28 grayscale images of handwritten digits (0-9)
    mnist = tf.keras.datasets.mnist
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    
    print(f"Loaded dataset: MNIST")
    print(f"  Training size: {X_train.shape} images")
    print(f"  Testing size: {X_test.shape} images")
    
    # 2. Preprocess: Normalize and Reshape
    # Reshape images to add a single channel (grayscale) -> (28, 28, 1)
    # Scale pixel values from [0, 255] to [0.0, 1.0]
    X_train = X_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    X_test = X_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    
    # 3. Build Convolutional Neural Network (CNN)
    model = Sequential([
        layers.Input(shape=(28, 28, 1)),
        
        # Conv block 1
        layers.Conv2D(32, kernel_size=(3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        
        # Conv block 2
        layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        
        # Flatten and Dense layers
        layers.Flatten(),
        layers.Dropout(0.25),  # Prevents overfitting
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')  # 10 output classes (0-9)
    ])
    
    model.summary()
    
    # 4. Compile Model
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',  # Target labels are integers, not one-hot
        metrics=['accuracy']
    )
    
    # 5. Train Model (For speed in verification, we do 2 epochs on a subset or full dataset)
    print("\nTraining CNN classifier on MNIST digits...")
    # Train on first 10,000 samples for fast training run, validating on 1,000 samples
    model.fit(
        X_train[:10000], y_train[:10000],
        validation_data=(X_test[:1000], y_test[:1000]),
        epochs=3,
        batch_size=64,
        verbose=1
    )
    
    # 6. Evaluation
    print("\nEvaluating on full Test Set...")
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Accuracy: {accuracy*100:.2f}%")
    
    # Run a prediction sample
    sample_img = X_test[0].reshape(1, 28, 28, 1)
    prediction = np.argmax(model.predict(sample_img, verbose=0))
    print(f"Sample Prediction -> Image True Label: {y_test[0]} | Predicted Label: {prediction}")

if __name__ == "__main__":
    main()
