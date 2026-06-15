import numpy as np

def batch_normalization(X, gamma=1.0, beta=0.0, eps=1e-5):
    """
    Batch Normalization: Normalizes inputs across the batch dimension (axis=0).
    """
    mean = np.mean(X, axis=0, keepdims=True)
    var = np.var(X, axis=0, keepdims=True)
    X_norm = (X - mean) / np.sqrt(var + eps)
    return gamma * X_norm + beta

def layer_normalization(X, gamma=1.0, beta=0.0, eps=1e-5):
    """
    Layer Normalization: Normalizes inputs across the feature dimension (axis=1).
    """
    mean = np.mean(X, axis=1, keepdims=True)
    var = np.var(X, axis=1, keepdims=True)
    X_norm = (X - mean) / np.sqrt(var + eps)
    return gamma * X_norm + beta

def main():
    print("=== Day 4: Normalization (Batch vs Layer) ===")
    
    # Input batch X: shape (batch_size=3, num_features=4)
    X = np.array([
        [1.0,  2.0,  3.0,  10.0],
        [2.0,  4.0,  6.0,  20.0],
        [3.0,  6.0,  9.0,  30.0]
    ])
    
    print("Input Batch X (3 samples, 4 features):")
    print(X)
    print()
    
    print("Batch Normalization (normalizing each feature column across samples):")
    print(batch_normalization(X).round(4))
    print()
    
    print("Layer Normalization (normalizing each sample row across features):")
    print(layer_normalization(X).round(4))
    print()
    
    print("Core Conceptual Differences:")
    print("  * Batch Normalization: Calculates statistics over the mini-batch. Great for Feedforward/CNN architectures, but behaves poorly when batch sizes are small (batch_size < 4) or vary dynamically.")
    print("  * Layer Normalization: Calculates statistics over the hidden units of a single sample. Fits sequence models (RNNs/Transformers) perfectly, since it is independent of batch size.")

if __name__ == "__main__":
    main()
