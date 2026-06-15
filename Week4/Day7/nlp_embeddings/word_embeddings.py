import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Sequential

def cosine_similarity(v1, v2):
    """
    Computes cosine similarity between two vectors.
    """
    dot = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    return dot / (norm1 * norm2)

def main():
    print("=== Day 13: NLP Word Embeddings ===")
    
    # 1. Vocabulary Definition
    vocab = {
        "<PAD>": 0,
        "king": 1,
        "queen": 2,
        "man": 3,
        "woman": 4,
        "apple": 5,
        "orange": 6
    }
    vocab_size = len(vocab)
    embedding_dim = 4  # Low dimensional representation for demonstration
    
    # 2. Build Keras model with Embedding Layer
    # Embedding maps integers (word indices) to continuous vectors of size embedding_dim
    model = Sequential([
        layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=1, name="word_embeddings"),
        layers.Flatten(),
        layers.Dense(3, activation='softmax')  # Dummy classification head
    ])
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    
    # Train briefly on dummy targets to update weights
    X_dummy = np.array([[1], [2], [3], [4], [5], [6]])  # Vocab items
    y_dummy = np.array([0, 0, 1, 1, 2, 2])
    model.fit(X_dummy, y_dummy, epochs=10, verbose=0)
    
    # 3. Retrieve Embedding Weights
    embedding_layer = model.get_layer("word_embeddings")
    weights = embedding_layer.get_weights()[0]
    
    print("Embedding Weights Matrix (Vocabulary size = 7, Dim = 4):")
    print(weights)
    print()
    
    # 4. Compare Word Similarities in the Embedding Space
    king_vec = weights[vocab["king"]]
    queen_vec = weights[vocab["queen"]]
    man_vec = weights[vocab["man"]]
    apple_vec = weights[vocab["apple"]]
    
    print("Cosine Similarities:")
    print(f"  Similarity('king', 'queen'): {cosine_similarity(king_vec, queen_vec):.4f}")
    print(f"  Similarity('king', 'man'):   {cosine_similarity(king_vec, man_vec):.4f}")
    print(f"  Similarity('king', 'apple'): {cosine_similarity(king_vec, apple_vec):.4f}")
    
    print("\nSummary of Concepts:")
    print("  * One-Hot Encoding: Represents words as orthogonal vectors. Lacks semantic similarity context.")
    print("  * Embedding Layer: Learns low-dimensional continuous vector mappings for words, placing semantically similar words closer in geometric space.")

if __name__ == "__main__":
    main()
