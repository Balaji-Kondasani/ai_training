import numpy as np

def softmax(x, axis=-1):
    """
    Computes softmax along the specified axis.
    """
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

def scaled_dot_product_attention(Q, K, V):
    """
    Computes Scaled Dot-Product Attention:
    Attention(Q, K, V) = softmax(Q * K.T / sqrt(d_k)) * V
    """
    # Dimension of the keys (d_k)
    d_k = Q.shape[-1]
    
    # 1. Compute dot product similarity between query and keys
    # Matmul(Q, K.T)
    scores = np.dot(Q, K.T)
    print("1. Raw Attention Scores (Q * K.T):")
    print(scores)
    print()
    
    # 2. Scale scores by square root of d_k
    # Prevents variance explosion in high dimensions (which drives softmax into flat gradients)
    scaled_scores = scores / np.sqrt(d_k)
    print(f"2. Scaled Attention Scores (divided by sqrt(d_k) = {np.sqrt(d_k):.2f}):")
    print(scaled_scores)
    print()
    
    # 3. Softmax activation to convert scores to weights/probabilities [0, 1]
    attention_weights = softmax(scaled_scores, axis=-1)
    print("3. Attention Weights (Softmax output row-wise):")
    print(attention_weights)
    print()
    
    # 4. Multiply attention weights by Value vectors
    # Net output is a weighted sum of the values
    output = np.dot(attention_weights, V)
    print("4. Final Weighted Output (Weights * V):")
    print(output)
    print()
    
    return output, attention_weights

def main():
    print("=== Day 15: Scaled Dot-Product Attention (Self-Attention) ===")
    
    # Representing a sequence of 3 tokens: e.g. ["The", "fox", "jumps"]
    # Let word embedding dimensions be d_k = 4
    # Query (Q), Key (K), and Value (V) matrices
    np.random.seed(42)
    Q = np.array([
        [1.0, 0.0, 1.0, 0.0],  # Token 1
        [0.0, 2.0, 0.0, 1.0],  # Token 2
        [1.0, 1.0, 0.0, 2.0]   # Token 3
    ])
    K = Q  # Self-attention means keys and queries are derived from the same inputs
    V = np.array([
        [10.0, 0.0],           # Value mapping for Token 1 (dim = 2)
        [ 0.0, 5.0],           # Value mapping for Token 2
        [ 5.0, 5.0]            # Value mapping for Token 3
    ])
    
    print("Input Query (Q) / Key (K) matrix (3 tokens, 4 dimensions):")
    print(Q)
    print("\nInput Value (V) matrix (3 tokens, 2 dimensions):")
    print(V)
    print("-" * 60 + "\n")
    
    output, weights = scaled_dot_product_attention(Q, K, V)
    
    print("Summary of Concepts:")
    print("  * Query (Q): What features a token is looking for.")
    print("  * Key (K): What features a token possesses.")
    print("  * Value (V): The actual content/information of the token to be compiled.")
    print("  * Scaled Dot-Product Attention scales the similarity dot product to avoid extreme values before applying softmax.")

if __name__ == "__main__":
    main()
