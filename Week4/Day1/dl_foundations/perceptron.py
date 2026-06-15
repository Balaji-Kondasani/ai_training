import numpy as np

class Perceptron:
    """
    Biological Neuron vs Artificial Neuron:
    - Biological: Dendrites (Inputs) -> Soma (Cell Body/Summation) -> Axon (Output)
    - Artificial: Inputs (x) -> Weighted Summation (wx + b) -> Activation (Step function)
    """
    def __init__(self, input_size, lr=0.1):
        self.weights = np.zeros(input_size + 1)  # +1 for bias
        self.lr = lr
        
    def predict(self, x):
        x_with_bias = np.insert(x, 0, 1)
        activation = np.dot(self.weights, x_with_bias)
        return 1 if activation >= 0 else 0
        
    def train(self, X, y, epochs=10):
        for _ in range(epochs):
            for xi, target in zip(X, y):
                prediction = self.predict(xi)
                error = target - prediction
                if error != 0:
                    xi_with_bias = np.insert(xi, 0, 1)
                    self.weights += self.lr * error * xi_with_bias

def main():
    print("=== Day 1: Perceptron AND Gate Training ===")
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 0, 0, 1])
    
    perceptron = Perceptron(input_size=2)
    perceptron.train(X, y, epochs=15)
    
    print("AND Gate Predictions:")
    for xi in X:
        print(f"  Input: {xi} -> Prediction: {perceptron.predict(xi)}")
    print(f"Weights (including bias at index 0): {perceptron.weights}")

if __name__ == "__main__":
    main()
