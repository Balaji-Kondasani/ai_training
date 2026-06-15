import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences

def main():
    print("=== Day 14: Project 6 - Sentiment Analysis using LSTMs ===")
    
    # 1. Vocabulary & Mock Dataset
    vocab = {
        "<PAD>": 0, "<UNK>": 1, "good": 2, "great": 3, "excellent": 4, "love": 5, 
        "bad": 6, "worst": 7, "terrible": 8, "waste": 9, "boring": 10,
        "movie": 11, "acting": 12, "plot": 13, "amazing": 14, "poor": 15
    }
    vocab_size = len(vocab)
    max_len = 6
    
    # Text reviews converted to word indices
    # 1 = Positive, 0 = Negative
    data_raw = [
        ("great movie love plot", 1),
        ("excellent acting amazing plot", 1),
        ("good acting love movie", 1),
        ("bad movie worst plot", 0),
        ("terrible waste boring acting", 0),
        ("poor acting bad plot", 0),
    ] * 20  # 120 samples
    
    X_indices = []
    y = []
    
    for text, label in data_raw:
        indices = [vocab.get(word, 1) for word in text.split()]
        X_indices.append(indices)
        y.append(label)
        
    X_indices = np.array(X_indices, dtype=object)
    y = np.array(y)
    
    # Pad sequences to ensure uniform input length
    X_padded = pad_sequences(X_indices, maxlen=max_len, padding='post', value=0)
    
    # Split
    split = 90
    X_train, X_test = X_padded[:split], X_padded[split:]
    y_train, y_test = y[:split], y[split:]
    
    print(f"Dataset generated:")
    print(f"  Training samples: {X_train.shape} | Labels: {len(y_train)}")
    print(f"  Testing samples:  {X_test.shape} | Labels: {len(y_test)}")
    
    # 2. Build RNN/LSTM Model
    model = Sequential([
        layers.Embedding(input_dim=vocab_size, output_dim=8, input_length=max_len),
        layers.Bidirectional(layers.LSTM(16, return_sequences=False)),
        layers.Dropout(0.2),
        layers.Dense(8, activation='relu'),
        layers.Dense(1, activation='sigmoid')  # Binary output
    ])
    
    # 3. Compile Model
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    model.summary()
    
    # 4. Train Model
    print("\nTraining LSTM Sentiment Classifier...")
    model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=12,
        batch_size=16,
        verbose=1
    )
    
    # 5. Evaluate
    print("\nEvaluating on Test Set...")
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Accuracy: {accuracy*100:.2f}%")
    
    # Predict on new unseen sequence
    # "excellent movie plot" -> Positive
    test_review = "excellent movie plot"
    test_indices = [vocab.get(word, 1) for word in test_review.split()]
    test_padded = pad_sequences([test_indices], maxlen=max_len, padding='post', value=0)
    
    prob = model.predict(test_padded, verbose=0)[0][0]
    sentiment = "Positive" if prob >= 0.5 else "Negative"
    print(f"New Review: '{test_review}' -> Predicted Sentiment: {sentiment} (Confidence: {prob if prob >= 0.5 else 1 - prob:.2%})")

if __name__ == "__main__":
    main()
