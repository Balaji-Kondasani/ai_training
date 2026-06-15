import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, Input

class CausalTransformerDecoder(layers.Layer):
    """
    Custom Decoder-Only Transformer layer for Autoregressive Generation (GPT).
    Incorporates self-attention with causal look-ahead masking.
    """
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(embed_dim),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training=None):
        seq_len = tf.shape(inputs)[1]
        
        # 1. Create Causal Look-Ahead Mask dynamically
        # Shape: (seq_len, seq_len)
        causal_mask = tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
        # Reshape to fit MHA: (1, seq_len, seq_len)
        causal_mask = tf.reshape(causal_mask, (1, seq_len, seq_len))
        
        # 2. Causal Multi-Head Self-Attention
        attn_output = self.att(query=inputs, value=inputs, key=inputs, attention_mask=causal_mask)
        attn_output = self.dropout1(attn_output, training=training)
        # Residual + LayerNorm
        out1 = self.layernorm1(inputs + attn_output)
        
        # 3. Feed-Forward Network
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        # Residual + LayerNorm
        return self.layernorm2(out1 + ffn_output)

def main():
    print("=== Day 20: Project 8 - Text Generation with Decoder-Only Transformer (Mini-GPT) ===")
    
    # 1. Corpus and Character-level indexing
    text = "hello world. learn deep learning and build artificial intelligence models."
    chars = sorted(list(set(text)))
    char_to_idx = {c: i for i, c in enumerate(chars)}
    idx_to_char = {i: c for i, c in enumerate(chars)}
    vocab_size = len(chars)
    
    # 2. Create Sequences & Targets
    # Input sequence: "hello" -> Target sequence: "ello " (Predicting next token for EACH step)
    seq_len = 8
    X_list, y_list = [], []
    
    for i in range(len(text) - seq_len):
        seq_in = text[i : i + seq_len]
        seq_out = text[i + 1 : i + seq_len + 1]
        
        X_list.append([char_to_idx[c] for c in seq_in])
        y_list.append([char_to_idx[c] for c in seq_out])
        
    X = np.array(X_list, dtype=np.int32)
    y = np.array(y_list, dtype=np.int32)
    
    embed_dim = 16
    num_heads = 2
    ff_dim = 16
    
    # 3. Build Decoder-Only Transformer Model
    inputs = Input(shape=(seq_len,))
    embedding_layer = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)(inputs)
    # Custom causal block
    decoder_block = CausalTransformerDecoder(embed_dim=embed_dim, num_heads=num_heads, ff_dim=ff_dim)(embedding_layer)
    # Dense output layer at EACH sequence step (predict next char distribution)
    outputs = layers.Dense(vocab_size, activation="softmax")(decoder_block)
    
    model = Model(inputs=inputs, outputs=outputs, name="mini_gpt")
    
    # 4. Compile Model
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    model.summary()
    
    # 5. Train Model
    print("\nTraining Mini-GPT model character auto-complete...")
    model.fit(X, y, epochs=100, batch_size=16, verbose=0)
    print("Training finished!")
    
    # 6. Autoregressive Text Generation
    seed = "hello wo"  # Must be exactly seq_len (8 chars)
    generated = seed
    
    print(f"\n--- Generating text from seed: '{seed}' ---")
    for _ in range(40):
        # Format seed indices
        input_seq = np.zeros((1, seq_len), dtype=np.int32)
        for t, char in enumerate(seed):
            input_seq[0, t] = char_to_idx.get(char, 0)
            
        # Predict probability distribution for the NEXT token at the LAST step
        preds = model.predict(input_seq, verbose=0)[0]  # shape: (seq_len, vocab_size)
        last_step_preds = preds[-1]  # We only care about the prediction of the last token
        
        next_idx = np.argmax(last_step_preds)
        next_char = idx_to_char[next_idx]
        
        generated += next_char
        seed = seed[1:] + next_char
        
    print(f"Generated Result: '{generated}'")

if __name__ == "__main__":
    main()
