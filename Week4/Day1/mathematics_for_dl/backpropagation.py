import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def main():
    print("=== Day 2: Backpropagation from scratch ===")
    
    # Dataset: XOR Gate
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])
    
    np.random.seed(42)
    
    # Network dimensions
    input_size = 2
    hidden_size = 3
    output_size = 1
    
    # Weight & Bias Initialization
    weights_1 = np.random.uniform(size=(input_size, hidden_size))
    weights_2 = np.random.uniform(size=(hidden_size, output_size))
    
    bias_1 = np.random.uniform(size=(1, hidden_size))
    bias_2 = np.random.uniform(size=(1, output_size))
    
    lr = 0.5
    epochs = 6001
    
    print("Training neural network to solve XOR...")
    for epoch in range(epochs):
        # Forward pass
        hidden_input = np.dot(X, weights_1) + bias_1
        hidden_output = sigmoid(hidden_input)
        
        output_input = np.dot(hidden_output, weights_2) + bias_2
        predicted_output = sigmoid(output_input)
        
        # Mean Squared Error Loss
        error = y - predicted_output
        mse_loss = np.mean(error ** 2)
        
        if epoch % 2000 == 0:
            print(f"  Epoch {epoch:5d} | MSE Loss: {mse_loss:.6f}")
            
        # Backpropagation (using chain rule)
        output_delta = error * sigmoid_derivative(predicted_output)
        hidden_error = output_delta.dot(weights_2.T)
        hidden_delta = hidden_error * sigmoid_derivative(hidden_output)
        
        # Gradient updates
        weights_2 += hidden_output.T.dot(output_delta) * lr
        weights_1 += X.T.dot(hidden_delta) * lr
        bias_2 += np.sum(output_delta, axis=0, keepdims=True) * lr
        bias_1 += np.sum(hidden_delta, axis=0, keepdims=True) * lr

    print("\nFinal XOR Gate Predictions:")
    hidden_output = sigmoid(np.dot(X, weights_1) + bias_1)
    final_predictions = sigmoid(np.dot(hidden_output, weights_2) + bias_2)
    for inputs, true, pred in zip(X, y, final_predictions):
        print(f"  Input: {inputs} -> True: {true[0]} -> Pred: {pred[0]:.4f} (Class: {1 if pred[0] >= 0.5 else 0})")

if __name__ == "__main__":
    main()
