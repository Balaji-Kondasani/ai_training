import numpy as np

# Quadratic loss surface function: f(w) = w^2
# Gradient: df(w)/dw = 2*w
def loss_function(w):
    return w ** 2

def gradient(w):
    return 2 * w

class SGDOptimizer:
    def __init__(self, lr=0.1):
        self.lr = lr
    def step(self, w):
        return w - self.lr * gradient(w)

class NesterovMomentumOptimizer:
    def __init__(self, lr=0.1, beta=0.9):
        self.lr = lr
        self.beta = beta
        self.v = 0
    def step(self, w):
        # Lookahead step
        w_lookahead = w - self.beta * self.v
        # Compute gradient at lookahead position
        g = gradient(w_lookahead)
        # Update velocity
        self.v = self.beta * self.v + self.lr * g
        return w - self.v

class AdaGradOptimizer:
    def __init__(self, lr=0.1, eps=1e-8):
        self.lr = lr
        self.eps = eps
        self.sum_squared_g = 0
    def step(self, w):
        g = gradient(w)
        # Accumulate squared gradients
        self.sum_squared_g += g ** 2
        # Scale learning rate per parameter
        return w - (self.lr / (np.sqrt(self.sum_squared_g) + self.eps)) * g

class AdamWOptimizer:
    def __init__(self, lr=0.1, beta1=0.9, beta2=0.999, weight_decay=0.01, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.weight_decay = weight_decay
        self.eps = eps
        self.m = 0
        self.v = 0
        self.t = 0
    def step(self, w):
        self.t += 1
        g = gradient(w)
        
        # 1. Update biased moment estimates
        self.m = self.beta1 * self.m + (1 - self.beta1) * g
        self.v = self.beta2 * self.v + (1 - self.beta2) * (g ** 2)
        
        # 2. Bias correction
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)
        
        # 3. Apply decoupled weight decay (L2 regularization) directly to weights,
        # rather than mixing it inside the gradient averages (which standard Adam does)
        w = w - self.lr * self.weight_decay * w
        
        # 4. Standard Adam step update
        return w - (self.lr / (np.sqrt(v_hat) + self.eps)) * m_hat

def run_optimization(optimizer, start_w, steps=8):
    w = start_w
    history = [w]
    for _ in range(steps):
        w = optimizer.step(w)
        history.append(w)
    return history

def main():
    print("=== Day 3: Optimization & Gradient Descent (Enriched) ===")
    
    start_weight = 10.0
    learning_rate = 0.1
    steps = 8
    
    print(f"Starting weight: {start_weight}")
    print(f"Goal: Minimize f(w) = w^2 (optimal w = 0)\n")
    
    # Initialize optimizers
    sgd = SGDOptimizer(lr=learning_rate)
    nesterov = NesterovMomentumOptimizer(lr=learning_rate)
    adagrad = AdaGradOptimizer(lr=learning_rate)
    adamw = AdamWOptimizer(lr=learning_rate, weight_decay=0.05)
    
    # Run optimization
    sgd_history = run_optimization(sgd, start_weight, steps)
    nesterov_history = run_optimization(nesterov, start_weight, steps)
    adagrad_history = run_optimization(adagrad, start_weight, steps)
    adamw_history = run_optimization(adamw, start_weight, steps)
    
    print("Optimizer Convergence History:")
    print(f"{'Step':<5} | {'SGD':<10} | {'Nesterov':<10} | {'AdaGrad':<10} | {'AdamW':<10}")
    print("-" * 55)
    for step in range(steps + 1):
        print(f"{step:<5d} | {sgd_history[step]:<10.4f} | {nesterov_history[step]:<10.4f} | {adagrad_history[step]:<10.4f} | {adamw_history[step]:<10.4f}")
        
    print("\nSummary of Concepts:")
    print("  * Nesterov Momentum: Computes the gradient *after* applying velocity lookahead. Prevents overshooting targets.")
    print("  * AdaGrad: Historically important. Scales learning rates down based on parameter frequency, but learning rate can decay to zero.")
    print("  * AdamW: Fixes the standard Adam weight decay bug. Decouples L2 regularization from gradient moments, improving generalization.")

if __name__ == "__main__":
    main()
