import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.preprocessing.sequence import pad_sequences

class TransformerEncoderBlock(layers.Layer):
    """
    Custom Transformer Encoder layer implementing:
    1. Multi-Head Self-Attention (MHA) + Residual Add & Norm.
    2. Feed-Forward Network (FFN) + Residual Add & Norm.
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
        # 1. Self-Attention Block
        attn_output = self.att(query=inputs, value=inputs, key=inputs)
        attn_output = self.dropout1(attn_output, training=training)
        # Residual connection + LayerNorm
        out1 = self.layernorm1(inputs + attn_output)
        
        # 2. Feed-Forward Block
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        # Residual connection + LayerNorm
        return self.layernorm2(out1 + ffn_output)

def main():
    print("=== Day 19: Project 7 - Sentiment Classification with Transformer Encoder ===")
    
    # 1. Setup Mock Dataset
    vocab = {
        "<PAD>": 0, "<UNK>": 1, "great": 2, "awesome": 3, "love": 4, "wonderful": 5,
        "bad": 6, "worst": 7, "terrible": 8, "waste": 9, "boring": 10, "movie": 11
    }
    vocab_size = len(vocab)
    max_len = 5
    embed_dim = 16
    num_heads = 2
    ff_dim = 16
    
    # 1 = Positive, 0 = Negative
    data_raw = [
        ("great movie love movie", 1),
        ("awesome movie wonderful movie", 1),
        ("love awesome great movie", 1),
        ("bad movie worst movie", 0),
        ("terrible waste boring movie", 0),
        ("worst boring bad movie", 0)
    ] * 20  # 120 samples
    
    X_indices = []
    y = []
    
    for text, label in data_raw:
        indices = [vocab.get(word, 1) for word in text.split()]
        X_indices.append(indices)
        y.append(label)
        
    X_indices = np.array(X_indices, dtype=object)
    y = np.array(y)
    
    X_padded = pad_sequences(X_indices, maxlen=max_len, padding='post', value=0)
    
    # Split
    split = 95
    X_train, X_test = X_padded[:split], X_padded[split:]
    y_train, y_test = y[:split], y[split:]
    
    # 2. Build Model using Custom Encoder Block
    inputs = Input(shape=(max_len,))
    # Embedding maps token IDs to vectors
    embedding_layer = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)(inputs)
    # Custom Transformer block
    transformer_block = TransformerEncoderBlock(embed_dim=embed_dim, num_heads=num_heads, ff_dim=ff_dim)(embedding_layer)
    # Global pooling collapse dimensions (sequence length to average vector)
    pooling_layer = layers.GlobalAveragePooling1D()(transformer_block)
    dropout_layer = layers.Dropout(0.1)(pooling_layer)
    outputs = layers.Dense(1, activation="sigmoid")(dropout_layer)
    
    model = Model(inputs=inputs, outputs=outputs, name="transformer_classifier")
    
    # 3. Compile Model
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    model.summary()
    
    # 4. Train Model
    print("\nTraining Transformer Encoder Classifier...")
    model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=15,
        batch_size=16,
        verbose=1
    )
    
    # 5. Evaluate
    print("\nEvaluating on Test Set...")
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Accuracy: {accuracy*100:.2f}%")

if __name__ == "__main__":
    main()
