import numpy as np

def initialize_weights(input_dim, output_dim, method='random'):
    if method == 'random':
        return np.random.randn(input_dim, output_dim) * 0.01
    elif method == 'xavier':
        limit = np.sqrt(2.0 / (input_dim + output_dim))
        return np.random.randn(input_dim, output_dim) * limit
    elif method == 'he':
        limit = np.sqrt(2.0 / input_dim)
        return np.random.randn(input_dim, output_dim) * limit
    else:
        raise ValueError(f"Unknown method {method}")

def main():
    print("=== Day 4: Weight Initialization ===")
    np.random.seed(42)
    layer_size = 500
    num_layers = 5
    
    methods = ['random', 'xavier', 'he']
    for method in methods:
        print(f"\nMethod: {method.upper()} Initialization")
        x = np.random.randn(1, layer_size)
        
        for i in range(1, num_layers + 1):
            w = initialize_weights(layer_size, layer_size, method=method)
            z = np.dot(x, w)
            x = np.maximum(0, z) if method == 'he' else np.tanh(z)
            print(f"  Layer {i} Activations Std Dev: {np.std(x):.6f}")

if __name__ == "__main__":
    main()
