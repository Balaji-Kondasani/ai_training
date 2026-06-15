import numpy as np

def step_decay(initial_lr, epoch, drop_rate=0.5, epochs_drop=5.0):
    """
    Halves learning rate every epochs_drop steps.
    """
    return initial_lr * np.power(drop_rate, np.floor((1 + epoch) / epochs_drop))

def cosine_annealing(initial_lr, epoch, total_epochs):
    """
    Cosine Annealing decay to smoothly decrease learning rate to near-zero.
    """
    return 0.5 * initial_lr * (1 + np.cos(np.pi * epoch / total_epochs))

def main():
    print("=== Day 3: Learning Rate Scheduling ===")
    
    initial_lr = 0.1
    total_epochs = 15
    
    print(f"Initial LR: {initial_lr} | Total Epochs: {total_epochs}\n")
    print(f"{'Epoch':<5} | {'Step Decay LR':<15} | {'Cosine Annealing LR':<20}")
    print("-" * 48)
    for epoch in range(total_epochs):
        step_lr = step_decay(initial_lr, epoch)
        cos_lr = cosine_annealing(initial_lr, epoch, total_epochs)
        print(f"{epoch:<5d} | {step_lr:<15.6f} | {cos_lr:<20.6f}")
        
    print("\nSummary of Concepts:")
    print("  * Step Decay: Drops learning rate abruptly at fixed intervals. Simple, but can cause shocks to training.")
    print("  * Cosine Annealing: Smoothly decays learning rate using a cosine curve. Recommends starting high and shrinking toward zero near target convergence.")

if __name__ == "__main__":
    main()
