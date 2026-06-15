import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow.keras import layers

def create_causal_mask(size):
    """
    Generates a lower-triangular causal mask for autoregressive decoders.
    1s allow attention, 0s block/mask future tokens.
    """
    # tf.linalg.band_part(input, num_lower, num_upper)
    # Setting num_lower=-1 keeps all lower elements, num_upper=0 masks upper elements
    mask = tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask

def main():
    print("=== Day 16: Transformer Architectures (BERT, GPT & LLMs) ===")
    
    # 1. Causal Masking (Look-Ahead Mask) for Decoders (GPT)
    # Autoregressive generation requires preventing tokens from attending to future tokens
    seq_len = 5
    causal_mask = create_causal_mask(seq_len)
    
    print("Causal Look-Ahead Mask (5x5):")
    print(causal_mask.numpy())
    print("\nHow this works during decoding:")
    print("  * Token at index 0 can ONLY attend to index 0.")
    print("  * Token at index 2 can attend to indexes 0, 1, and 2.")
    print("  * Future tokens (e.g. index 3 and 4) are masked out (set to -inf before softmax).\n")
    
    # 2. Multi-Head Attention instantiation in Keras
    # Illustrating how Multi-Head Attention layers are structured
    num_heads = 4
    key_dim = 16  # Dimensionality of query and key per head
    
    mha_layer = layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)
    
    # Dummy tensors representing query (Q), key (K), value (V)
    # Shape: (batch_size, sequence_length, embedding_dim) -> (1, 5, 64)
    q_dummy = tf.random.uniform((1, seq_len, 64))
    k_dummy = q_dummy
    v_dummy = q_dummy
    
    # Run MHA layer
    # For decoders, we can pass the causal mask to block future context
    # Attention mask shape: (batch_size, query_length, key_length)
    mask_input = tf.reshape(causal_mask, (1, seq_len, seq_len))
    
    output_tensor, attention_weights = mha_layer(
        query=q_dummy,
        value=v_dummy,
        key=k_dummy,
        attention_mask=mask_input,
        return_attention_scores=True
    )
    
    print("Keras MultiHeadAttention Output:")
    print(f"  Input query tensor shape:  {q_dummy.shape}")
    print(f"  Causal mask shape:         {mask_input.shape}")
    print(f"  Output attention tensor:   {output_tensor.shape}")
    print(f"  Attention weights shape:   {attention_weights.shape} (batch, heads, q_len, k_len)\n")
    
    # 3. Architectural Comparisons
    print("--- Core Architecture Comparison ---")
    print("BERT (Bidirectional Encoder Representations from Transformers):")
    print("  * Architecture: Transformer Encoder stack.")
    print("  * Attention: Full Bidirectional Self-Attention (tokens can look left and right).")
    print("  * Objective: Masked Language Modeling (MLM) and Next Sentence Prediction (NSP).")
    print("  * Primary Use: Understading, classification, named entity recognition, question answering.")
    print()
    print("GPT (Generative Pre-trained Transformer):")
    print("  * Architecture: Transformer Decoder stack.")
    print("  * Attention: Masked/Causal Self-Attention (tokens can only look to the left/past).")
    print("  * Objective: Causal Language Modeling (predict next token autoregressively).")
    print("  * Primary Use: Text generation, conversational AI (chatbots), text completion.")

if __name__ == "__main__":
    main()
