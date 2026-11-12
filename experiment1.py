import numpy as np
import matplotlib.pyplot as plt


# =======================================================
# 1. THE 5 OPTIMIZER CLASSES
# =======================================================
class BatchGD:
    # Stage 1 and Stage 2 use the exact same update rule on a raw mathematical function.
    def __init__(self, learning_rate=0.01):
        self.eta = learning_rate

    def step(self, w, grad):
        return w - self.eta * grad


class Momentum:
    # Stage 3: Adds velocity as a shock absorber.
    def __init__(self, learning_rate=0.01, beta1=0.9):
        self.eta = learning_rate
        self.beta1 = beta1
        self.v = None

    def step(self, w, grad):
        if self.v is None:
            self.v = np.zeros_like(w)
        self.v = self.beta1 * self.v + (1 - self.beta1) * grad
        return w - self.eta * self.v


class RMSProp:
    # Stage 4: Adds steepness scaling.
    def __init__(self, learning_rate=0.05, beta2=0.99, epsilon=1e-8):
        self.eta = learning_rate
        self.beta2 = beta2
        self.epsilon = epsilon
        self.s = None

    def step(self, w, grad):
        if self.s is None:
            self.s = np.zeros_like(w)
        self.s = self.beta2 * self.s + (1 - self.beta2) * (grad**2)
        return w - (self.eta / (np.sqrt(self.s) + self.epsilon)) * grad


class Adam:
    # Stage 5: Combines Momentum and RMSProp with Bias Correction.
    def __init__(self, learning_rate=0.1, beta1=0.9, beta2=0.99, epsilon=1e-8):
        self.eta = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.s = None
        self.t = 0

    def step(self, w, grad):
        if self.m is None:
            self.m = np.zeros_like(w)
            self.s = np.zeros_like(w)

        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.s = self.beta2 * self.s + (1 - self.beta2) * (grad**2)

        m_hat = self.m / (1 - self.beta1**self.t)
        s_hat = self.s / (1 - self.beta2**self.t)

        return w - (self.eta / (np.sqrt(s_hat) + self.epsilon)) * m_hat


# =======================================================
# 2. THE LANDSCAPES & GRADIENTS
# =======================================================
def canyon(w):
    # f(w1, w2) = 0.5w1^2 + 25w2^2
    return 0.5 * w[0] ** 2 + 25 * w[1] ** 2


def canyon_grad(w):
    return np.array([w[0], 50 * w[1]])


def rosenbrock(w):
    # f(w1, w2) = (1 - w1)^2 + 100(w2 - w1^2)^2
    return (1 - w[0]) ** 2 + 100 * (w[1] - w[0] ** 2) ** 2


def rosenbrock_grad(w):
    dw1 = -2 * (1 - w[0]) - 400 * w[0] * (w[1] - w[0] ** 2)
    dw2 = 200 * (w[1] - w[0] ** 2)
    return np.array([dw1, dw2])


# =======================================================
# 3. EXPERIMENT RUNNER
# =======================================================
def run_experiment(optimizer_class, lr, w0, func, grad_func, steps=200):
    opt = optimizer_class(learning_rate=lr)
    w = np.array(w0, dtype=float)

    path = [w.copy()]
    losses = [func(w)]

    for _ in range(steps):
        g = grad_func(w)
        w = opt.step(w, g)
        path.append(w.copy())
        losses.append(func(w))

    return np.array(path), np.array(losses)


# =======================================================
# 4. EXECUTION & UNIFIED GRAPHING
# =======================================================
if __name__ == "__main__":
    w0 = [-1.5, 2.0]
    T = 200

    # We set different learning rates because advanced optimizers handle larger steps safely.
    configs = [
        ("Vanilla GD", BatchGD, 0.005),
        ("Momentum", Momentum, 0.005),
        ("RMSProp", RMSProp, 0.05),
        ("Adam", Adam, 0.1),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Create background grids for the contour maps
    w1_c, w2_c = np.meshgrid(np.linspace(-2, 2, 100), np.linspace(-2.5, 2.5, 100))
    Z_canyon = 0.5 * w1_c**2 + 25 * w2_c**2

    w1_r, w2_r = np.meshgrid(np.linspace(-2, 2, 100), np.linspace(-1, 3, 100))
    Z_rosen = (1 - w1_r) ** 2 + 100 * (w2_r - w1_r**2) ** 2

    # Setup Contour Plots
    axes[0, 0].contour(
        w1_c, w2_c, Z_canyon, levels=np.logspace(-1, 3, 20), cmap="Blues"
    )
    axes[0, 0].set_title("Canyon: 2D Trajectories")
    axes[0, 0].plot(0, 0, "r*", markersize=12)  # Target

    axes[0, 1].contour(w1_r, w2_r, Z_rosen, levels=np.logspace(-1, 3, 20), cmap="Blues")
    axes[0, 1].set_title("Rosenbrock: 2D Trajectories")
    axes[0, 1].plot(1, 1, "r*", markersize=12)  # Target (Rosenbrock minimum is at 1,1)

    # Run and plot for each optimizer
    for name, opt_cls, lr in configs:
        # Canyon
        path_c, loss_c = run_experiment(opt_cls, lr, w0, canyon, canyon_grad, T)
        axes[0, 0].plot(path_c[:, 0], path_c[:, 1], label=name, linewidth=2, alpha=0.8)
        axes[1, 0].plot(np.log10(loss_c + 1e-10), label=name, linewidth=2)

        # Rosenbrock
        path_r, loss_r = run_experiment(opt_cls, lr, w0, rosenbrock, rosenbrock_grad, T)
        axes[0, 1].plot(path_r[:, 0], path_r[:, 1], label=name, linewidth=2, alpha=0.8)
        axes[1, 1].plot(np.log10(loss_r + 1e-10), label=name, linewidth=2)

    # Formatting Loss Graphs
    axes[1, 0].set_title("Canyon: log10(Loss) vs Iteration")
    axes[1, 0].set_xlabel("Iteration t")
    axes[1, 0].set_ylabel("log10(Loss)")

    axes[1, 1].set_title("Rosenbrock: log10(Loss) vs Iteration")
    axes[1, 1].set_xlabel("Iteration t")
    axes[1, 1].set_ylabel("log10(Loss)")

    for ax in axes.flat:
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
