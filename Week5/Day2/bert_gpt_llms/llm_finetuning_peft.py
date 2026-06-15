import numpy as np

def lora_forward(x, W, A, B, scaling=1.0):
    """
    Simulates a LoRA forward pass:
    h = W * x + (alpha / r) * (B * A) * x
    """
    # Standard forward pass
    h_standard = np.dot(x, W)
    
    # LoRA adaptation path (A projects down to rank r, B projects back up)
    low_rank_path = np.dot(x, A)  # Project down
    lora_update = np.dot(low_rank_path, B)  # Project up
    
    # Combined output
    return h_standard + scaling * lora_update

def main():
    print("=== Day 16: LLM Fine-Tuning & LoRA (Low-Rank Adaptation) ===")
    
    # 1. Parameter Reduction Demonstration
    # Assume a Dense layer weight matrix of dimension d = 4096, k = 4096
    d, k = 4096, 4096
    full_params = d * k
    
    # Set LoRA rank r = 8
    r = 8
    # Matrix A has shape (d, r), Matrix B has shape (r, k)
    lora_params = (d * r) + (r * k)
    
    saving_pct = (1 - (lora_params / full_params)) * 100
    
    print(f"Dense Layer Weight matrix size: {d} x {k}")
    print(f"  * Full Fine-Tuning Parameters to update: {full_params:,}")
    print(f"  * LoRA Fine-Tuning Parameters (rank r={r}) : {lora_params:,}")
    print(f"  * Parameter Reduction: {saving_pct:.2f}% savings!\n")
    
    # 2. Forward pass simulation
    np.random.seed(42)
    x = np.array([[1.0, -0.5, 2.0]])  # Input vector (1, 3)
    W = np.random.randn(3, 3)          # Pre-trained base weights (3, 3)
    
    # LoRA low-rank updates (rank r = 1)
    A = np.random.randn(3, 1)          # Down projection (3, 1)
    B = np.random.randn(1, 3)          # Up projection (1, 3)
    
    # Forward pass outputs
    output_standard = np.dot(x, W)
    output_lora = lora_forward(x, W, A, B, scaling=0.5)
    
    print("Forward Pass outputs:")
    print(f"  Standard Forward Output: {output_standard}")
    print(f"  LoRA Adapted Output    : {output_lora}\n")
    
    # 3. Fine-Tuning Concepts Summary
    print("LLM Adaptation Concepts:")
    print("  * Full Fine-Tuning: Updates all parameters of the network. Computationally expensive, requires huge GPU resources.")
    print("  * PEFT (Parameter-Efficient Fine-Tuning): Freezes the base LLM weights and trains a small set of added parameters.")
    print("  * LoRA: Core PEFT method. Factorizes weight update matrices into two low-rank matrices, dramatically reducing training GPU memory usage while matching performance.")

if __name__ == "__main__":
    main()
