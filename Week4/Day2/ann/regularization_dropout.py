import numpy as np

def l1_regularization_penalty(weights, lambda_l1):
    """
    L1 Regularization (Lasso) penalty: lambda_l1 * sum(|w|)
    Pushes weights to be exactly zero, creating sparse models.
    """
    return lambda_l1 * np.sum(np.abs(weights))

def l2_regularization_penalty(weights, lambda_l2):
    """
    L2 Regularization (Ridge) penalty: 0.5 * lambda_l2 * sum(w^2)
    Pushes weights close to zero, preventing extreme values.
    """
    return 0.5 * lambda_l2 * np.sum(weights ** 2)

def inverted_dropout(x, keep_prob=0.8, training=True):
    """
    Inverted Dropout: Randomly sets activations to zero.
    Scales remaining activations by 1/keep_prob to preserve expectation values.
    No changes are needed during inference.
    """
    if not training or keep_prob >= 1.0:
        return x
        
    # Generate binary mask where keep_prob determines likelihood of 1s
    mask = (np.random.rand(*x.shape) < keep_prob)
    # Scale values by 1/keep_prob (Inverted Dropout)
    return (x * mask) / keep_prob

def main():
    print("=== Day 4: Regularization & Dropout ===")
    
    np.random.seed(42)
    weights = np.array([0.5, -1.2, 0.05, 2.0, -0.01])
    print(f"Sample weights: {weights}")
    print(f"  L1 Penalty (lambda=0.01): {l1_regularization_penalty(weights, 0.01):.6f}")
    print(f"  L2 Penalty (lambda=0.01): {l2_regularization_penalty(weights, 0.01):.6f}\n")
    
    # Inverted Dropout test
    activations = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
    print(f"Original activations: {activations}")
    print("Applying Inverted Dropout (keep_prob=0.6) during Training:")
    for i in range(3):
        drop_act = inverted_dropout(activations, keep_prob=0.6, training=True)
        print(f"  Run {i+1}: {drop_act} (Mean: {np.mean(drop_act):.2f})")
        
    print("\nInference Output (Dropout inactive):")
    print(f"  {inverted_dropout(activations, keep_prob=0.6, training=False)}")

if __name__ == "__main__":
    main()
