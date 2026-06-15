import numpy as np

def binary_cross_entropy_loss(y_true, y_pred):
    # Loss is for a single sample
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def binary_cross_entropy_cost(y_true, y_pred):
    # Cost is the average of individual losses
    losses = binary_cross_entropy_loss(y_true, y_pred)
    return np.mean(losses)

def main():
    print("=== Day 1: Loss vs Cost Functions & Bias Initialization ===")
    
    # 1. Loss vs Cost Demonstration
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([0.9, 0.2, 0.4, 0.75])
    
    print("Individual Losses:")
    for idx, (t, p) in enumerate(zip(y_true, y_pred)):
        loss = binary_cross_entropy_loss(t, p)
        print(f"  Sample {idx+1} | Target: {t} | Pred: {p:.2f} | Loss: {loss:.4f}")
        
    cost = binary_cross_entropy_cost(y_true, y_pred)
    print(f"\nOverall Cost (Average Loss): {cost:.4f}\n")
    
    # 2. Bias Initialization Explanation
    print("Bias Initialization Methods:")
    print("  * Standard: Initialize to zeros (e.g. np.zeros(bias_shape)).")
    print("  * Reason: Weight parameters break symmetry since they are randomized. Zero biases work perfectly.")
    print("  * Dead ReLU Prevention: Sometimes initialized to a small positive constant like 0.01 to ensure early neuron firing.")

if __name__ == "__main__":
    main()
