import os
# Disable TensorFlow logs to make output clean
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow.keras import layers, Model, Input, Sequential

def main():
    print("=== Day 5: TensorFlow & Keras Basics ===")
    
    # 1. Basic Tensor Operations
    print("--- 1. Tensor Operations ---")
    x = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    y = tf.constant([[5.0, 6.0], [7.0, 8.0]])
    z = tf.matmul(x, y)
    print(f"Matrix Multiplication (x * y):\n{z.numpy()}\n")
    
    # 2. Auto-differentiation using tf.GradientTape
    print("--- 2. Auto-Differentiation (GradientTape) ---")
    w = tf.Variable(3.0)  # Trainable parameter
    with tf.GradientTape() as tape:
        loss = w ** 2  # f(w) = w^2
    grad = tape.gradient(loss, w)
    print(f"For loss = w^2, at w = {w.numpy():.2f}, derivative dLoss/dw = {grad.numpy():.2f}\n")
    
    # 3. Model construction using Sequential API
    print("--- 3. Keras Sequential API ---")
    # Ideal for simple stack of layers
    seq_model = Sequential([
        layers.Input(shape=(10,)),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ])
    seq_model.summary()
    print()
    
    # 4. Model construction using Functional API
    print("--- 4. Keras Functional API ---")
    # Ideal for complex models (multiple inputs/outputs, shared layers, residual connections)
    inputs = Input(shape=(10,))
    x_dense1 = layers.Dense(64, activation='relu')(inputs)
    x_dense2 = layers.Dense(32, activation='relu')(x_dense1)
    outputs = layers.Dense(1)(x_dense2)
    
    func_model = Model(inputs=inputs, outputs=outputs, name="functional_model")
    func_model.summary()
    
    # 5. Compile & Train Model
    print("\nCompiling and running on synthetic regression data...")
    func_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    # Generate synthetic regression data
    import numpy as np
    X_train = np.random.randn(100, 10)
    y_train = np.random.randn(100, 1)
    
    # Train for 5 epochs
    history = func_model.fit(X_train, y_train, epochs=5, batch_size=16, verbose=1)
    print("\nTraining completed successfully!")

if __name__ == "__main__":
    main()
