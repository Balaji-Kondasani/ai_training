import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow.keras import layers, Sequential

def main():
    print("=== Day 12: Recurrent Architectures (RNN vs LSTM vs GRU) ===")
    
    # Define sequence dimensions
    # batch_size=1, timesteps=5 (sequence length), input_features=10 (vocabulary size or embedding dimensions)
    timesteps = 5
    input_dim = 10
    units = 16  # Dimension of hidden state
    
    # 1. SimpleRNN
    # Single tanh gate: h_t = tanh( W_hh * h_prev + W_xh * x_t + b )
    rnn_model = Sequential([
        layers.Input(shape=(timesteps, input_dim)),
        layers.SimpleRNN(units, return_sequences=False)
    ])
    
    # 2. GRU (Gated Recurrent Unit)
    # 2 gates (Reset and Update gates)
    gru_model = Sequential([
        layers.Input(shape=(timesteps, input_dim)),
        layers.GRU(units, return_sequences=False)
    ])
    
    # 3. LSTM (Long Short-Term Memory)
    # 3 gates (Input, Forget, and Output gates) + cell state
    lstm_model = Sequential([
        layers.Input(shape=(timesteps, input_dim)),
        layers.LSTM(units, return_sequences=False)
    ])
    
    # Print parameter count differences
    print("Architecture Comparison (units = 16):")
    print(f"  * SimpleRNN Parameter Count : {rnn_model.count_params()} (Simple state update)")
    print(f"  * GRU Parameter Count       : {gru_model.count_params()} (Reset & Update gates)")
    print(f"  * LSTM Parameter Count      : {lstm_model.count_params()} (Input, Forget, & Output gates)")
    
    # 4. Run simple forward pass with random sequence
    import numpy as np
    dummy_input = np.random.randn(1, timesteps, input_dim).astype('float32')
    
    rnn_output = rnn_model.predict(dummy_input, verbose=0)
    gru_output = gru_model.predict(dummy_input, verbose=0)
    lstm_output = lstm_model.predict(dummy_input, verbose=0)
    
    print("\nForward Pass output shapes:")
    print(f"  Input sequence shape: {dummy_input.shape}")
    print(f"  SimpleRNN Output    : {rnn_output.shape} -> Value sample: {rnn_output[0, :3]}")
    print(f"  GRU Output          : {gru_output.shape} -> Value sample: {gru_output[0, :3]}")
    print(f"  LSTM Output         : {lstm_output.shape} -> Value sample: {lstm_output[0, :3]}")
    
    print("\nSummary of Concepts:")
    print("  * SimpleRNN: Simple structure, but suffers heavily from vanishing/exploding gradients over long sequences.")
    print("  * LSTM: Preserves cell state across long gaps using forget/input/output gates to control memory retention.")
    print("  * GRU: Streamlined version of LSTM, merging cell and hidden state and using only update and reset gates (faster to train).")

if __name__ == "__main__":
    main()
