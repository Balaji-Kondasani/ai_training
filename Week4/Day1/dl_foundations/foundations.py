import numpy as np
import matplotlib.pyplot as plt

# --- Activation Functions ---

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(0, x)

def softmax(x):
    exp_x = np.exp(x - np.max(x))  # Shift for numerical stability
    return exp_x / exp_x.sum(axis=0)

# --- Perceptron Simulation ---

class Perceptron:
    def __init__(self, input_size, lr=0.1):
        self.weights = np.zeros(input_size + 1)  # +1 for bias
        self.lr = lr
        
    def predict(self, x):
        # Adding bias term to input
        x_with_bias = np.insert(x, 0, 1)
        activation = np.dot(self.weights, x_with_bias)
        return 1 if activation >= 0 else 0
        
    def train(self, X, y, epochs=10):
        for _ in range(epochs):
            for xi, target in zip(X, y):
                prediction = self.predict(xi)
                error = target - prediction
                if error != 0:
                    # Update weights: w = w + lr * error * xi
                    xi_with_bias = np.insert(xi, 0, 1)
                    self.weights += self.lr * error * xi_with_bias

def main():
    print("=== Day 1: Deep Learning Foundations ===")
    
    # 1. Test Activation Functions
    test_inputs = np.array([-2.0, -0.5, 0.0, 1.5, 3.0])
    print(f"Test inputs: {test_inputs}")
    print(f"  Sigmoid: {sigmoid(test_inputs)}")
    print(f"  Tanh:    {tanh(test_inputs)}")
    print(f"  ReLU:    {relu(test_inputs)}")
    print(f"  Softmax (on test inputs): {softmax(test_inputs)}")
    print()
    
    # 2. Train a Simple Perceptron on AND Gate
    print("--- Training Perceptron on AND Gate ---")
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 0, 0, 1])
    
    perceptron = Perceptron(input_size=2)
    perceptron.train(X, y, epochs=20)
    
    print("AND Gate Predictions:")
    for xi in X:
        print(f"  Input: {xi} -> Prediction: {perceptron.predict(xi)}")
    print(f"Trained Weights (including bias at index 0): {perceptron.weights}")
    
    # 3. Save plots of activation functions
    x_range = np.linspace(-5, 5, 200)
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(x_range, sigmoid(x_range), color='blue')
    plt.title("Sigmoid")
    plt.grid(True)
    
    plt.subplot(2, 2, 2)
    plt.plot(x_range, tanh(x_range), color='orange')
    plt.title("Tanh")
    plt.grid(True)
    
    plt.subplot(2, 2, 3)
    plt.plot(x_range, relu(x_range), color='green')
    plt.title("ReLU")
    plt.grid(True)
    
    plt.subplot(2, 2, 4)
    # For softmax, we compute across the range to show distribution shape
    plt.plot(x_range, softmax(x_range), color='purple')
    plt.title("Softmax (Distribution)")
    plt.grid(True)
    
    import os
    os.makedirs("../plots", exist_ok=True)
    plot_path = os.path.join("../plots", "dl_day_01_activations.png")
    plt.savefig(plot_path)
    print(f"\nSaved activation function plots to: {plot_path}")
    plt.close()

if __name__ == "__main__":
    main()
