import numpy as np

def softmax(x, axis=-1):
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

def luong_dot_product_attention(query, keys):
    """
    Luong (Dot-Product) Attention scoring:
    score(q, k) = q.T * k
    """
    # query shape: (hidden_dim,)
    # keys shape: (seq_len, hidden_dim)
    scores = np.dot(keys, query)  # Dot product for each key
    weights = softmax(scores)
    return weights

def bahdanau_additive_attention(query, keys, W_q, W_k, v_a):
    """
    Bahdanau (Additive) Attention scoring:
    score(q, k) = v_a.T * tanh(W_q * q + W_k * k)
    """
    # query shape: (hidden_dim,)
    # keys shape: (seq_len, hidden_dim)
    seq_len = keys.shape[0]
    scores = []
    
    # Calculate scores step-by-step for each key
    for i in range(seq_len):
        k = keys[i]
        # Linear projections
        q_proj = np.dot(W_q, query)
        k_proj = np.dot(W_k, k)
        # Add and pass through tanh activation
        combined = np.tanh(q_proj + k_proj)
        # Project to scalar score using v_a weight vector
        score = np.dot(v_a, combined)
        scores.append(score)
        
    scores = np.array(scores)
    weights = softmax(scores)
    return weights

def main():
    print("=== Day 18: Attention Mechanisms (Luong vs Bahdanau) ===")
    
    # Dimensions: hidden state size = 4, sequence length = 3
    hidden_dim = 4
    seq_len = 3
    np.random.seed(42)
    
    # Simulated encoder states (Keys)
    keys = np.array([
        [0.5, 0.1, -0.2, 0.8],  # Encoder Hidden State 1
        [-0.1, 0.9, 0.3, 0.2],  # Encoder Hidden State 2
        [0.8, -0.2, 0.7, 0.1]   # Encoder Hidden State 3
    ])
    
    # Simulated decoder state (Query) looking for specific features
    query = np.array([0.6, 0.0, 0.5, 0.2])
    
    print("Encoder Hidden States (Keys/Values):")
    print(keys)
    print("\nDecoder Hidden State (Query):")
    print(query)
    print()
    
    # 1. Luong Attention
    luong_weights = luong_dot_product_attention(query, keys)
    print("--- 1. Luong (Dot-Product) Attention ---")
    print(f"Alignment Scores / Weights: {luong_weights}")
    print(f"Index of highest attention: {np.argmax(luong_weights)}\n")
    
    # 2. Bahdanau Attention Setup
    # Additive attention uses projection matrices (learnable parameters)
    W_q = np.random.randn(hidden_dim, hidden_dim) * 0.1
    W_k = np.random.randn(hidden_dim, hidden_dim) * 0.1
    v_a = np.random.randn(hidden_dim) * 0.1
    
    bahdanau_weights = bahdanau_additive_attention(query, keys, W_q, W_k, v_a)
    print("--- 2. Bahdanau (Additive) Attention ---")
    print(f"Alignment Scores / Weights: {bahdanau_weights}")
    print(f"Index of highest attention: {np.argmax(bahdanau_weights)}\n")
    
    print("Core Conceptual Differences:")
    print("  * Luong (Dot-Product): Simpler and faster. Measures geometric alignment using simple projection/multiplication.")
    print("  * Bahdanau (Additive): Uses a small single-layer network. More flexible, performs better on low-dimensional sequence alignments.")

if __name__ == "__main__":
    main()
