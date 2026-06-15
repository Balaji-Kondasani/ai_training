import numpy as np
import matplotlib.pyplot as plt
import os

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(0, x)

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, x * alpha)

def elu(x, alpha=1.0):
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))

def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum(axis=0)

def main():
    print("=== Day 1: Activation Functions ===")
    test_inputs = np.array([-3.0, -0.5, 0.0, 1.5, 3.0])
    print(f"Test inputs: {test_inputs}")
    print(f"  Sigmoid:    {sigmoid(test_inputs)}")
    print(f"  Tanh:       {tanh(test_inputs)}")
    print(f"  ReLU:       {relu(test_inputs)}")
    print(f"  Leaky ReLU: {leaky_relu(test_inputs)}")
    print(f"  ELU:        {elu(test_inputs)}")
    print(f"  Softmax (distribution): {softmax(test_inputs)}")
    
    # Save plots
    x_range = np.linspace(-4, 4, 200)
    plt.figure(figsize=(14, 10))
    
    plt.subplot(3, 2, 1)
    plt.plot(x_range, sigmoid(x_range), color='blue')
    plt.title("Sigmoid")
    plt.grid(True)
    
    plt.subplot(3, 2, 2)
    plt.plot(x_range, tanh(x_range), color='orange')
    plt.title("Tanh")
    plt.grid(True)
    
    plt.subplot(3, 2, 3)
    plt.plot(x_range, relu(x_range), color='green')
    plt.title("ReLU")
    plt.grid(True)
    
    plt.subplot(3, 2, 4)
    plt.plot(x_range, leaky_relu(x_range), color='red')
    plt.title("Leaky ReLU (alpha=0.1)")
    plt.grid(True)
    
    plt.subplot(3, 2, 5)
    plt.plot(x_range, elu(x_range), color='purple')
    plt.title("ELU (alpha=1.0)")
    plt.grid(True)
    
    plt.subplot(3, 2, 6)
    plt.plot(x_range, softmax(x_range), color='cyan')
    plt.title("Softmax Distribution")
    plt.grid(True)
    
    plt.tight_layout()
    os.makedirs("../plots", exist_ok=True)
    plot_path = os.path.join("../plots", "dl_day_01_activations.png")
    plt.savefig(plot_path)
    print(f"\nSaved activation function plots to: {plot_path}")
    plt.close()

if __name__ == "__main__":
    main()
