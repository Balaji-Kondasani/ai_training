import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Sequential

def main():
    print("=== Day 17: Next Token Predictor using LSTM ===")
    
    # 1. Corpus & Vocabulary Setup (Character-level language modeling)
    text = "deep learning is fun and deep learning is powerful. neural networks are awesome."
    chars = sorted(list(set(text)))
    char_to_idx = {char: idx for idx, char in enumerate(chars)}
    idx_to_char = {idx: char for idx, char in enumerate(chars)}
    vocab_size = len(chars)
    
    print(f"Text Corpus length: {len(text)} characters")
    print(f"Unique characters (Vocabulary size): {vocab_size}")
    print(f"Characters list: {chars}\n")
    
    # 2. Create Dataset
    # Slice text into overlapping sequences of length `seq_len`
    seq_len = 10
    step = 1
    sequences = []
    next_chars = []
    
    for i in range(0, len(text) - seq_len, step):
        sequences.append(text[i : i + seq_len])
        next_chars.append(text[i + seq_len])
        
    X_indices = np.zeros((len(sequences), seq_len), dtype=np.int32)
    y_indices = np.zeros((len(sequences)), dtype=np.int32)
    
    for idx, seq in enumerate(sequences):
        for t, char in enumerate(seq):
            X_indices[idx, t] = char_to_idx[char]
        y_indices[idx] = char_to_idx[next_chars[idx]]
        
    print(f"Total training sequences generated: {len(sequences)}")
    print(f"Sample Sequence input: '{sequences[0]}' -> Next character target: '{next_chars[0]}'\n")
    
    # 3. Build LSTM Model
    model = Sequential([
        layers.Embedding(input_dim=vocab_size, output_dim=16, input_length=seq_len),
        layers.LSTM(64, return_sequences=False),
        layers.Dense(vocab_size, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    model.summary()
    
    # 4. Train Model
    print("\nTraining LSTM character-level next token predictor...")
    model.fit(X_indices, y_indices, epochs=60, batch_size=16, verbose=0)
    print("Training finished!")
    
    # 5. Text Generation (Autoregressive Generation)
    seed = "deep learn"  # Must be exactly seq_len (10 chars)
    generated = seed
    
    print(f"\n--- Generating text from seed: '{seed}' ---")
    for _ in range(30):
        # Format seed to indices
        input_seq = np.zeros((1, seq_len), dtype=np.int32)
        for t, char in enumerate(seed):
            input_seq[0, t] = char_to_idx.get(char, 0)
            
        # Predict probability distribution for next char
        preds = model.predict(input_seq, verbose=0)[0]
        # Choose index with highest probability
        next_idx = np.argmax(preds)
        next_char = idx_to_char[next_idx]
        
        # Append and slide the seed window
        generated += next_char
        seed = seed[1:] + next_char
        
    print(f"Generated Result: '{generated}'")

if __name__ == "__main__":
    main()
