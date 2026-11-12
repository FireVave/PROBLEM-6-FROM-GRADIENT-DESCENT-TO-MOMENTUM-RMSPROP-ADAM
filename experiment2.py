
import numpy as np
import matplotlib.pyplot as plt
import time
import tracemalloc


# Generate data
N = 50000
D = 50

rng = np.random.default_rng(42)

scales = np.logspace(-1, 1, D)
X = rng.normal(size=(N, D)) * scales

true_w = rng.normal(size=D)
y = X @ true_w + rng.normal(0, 0.5, N)


#loos&gradient
def loss(w, X, y):
    predictions = X @ w
    return np.mean((predictions - y) ** 2) / 2


def gradient(w, X, y):
    error = X @ w - y
    return X.T @ error / len(y)


#optimize
class BatchGD:
    def __init__(self, lr=0.01):
        self.lr = lr

    def step(self, w, grad):
        return w - self.lr * grad


class MiniBatchSGD:
    def __init__(self, lr=0.001):
        self.lr = lr

    def step(self, w, grad):
        return w - self.lr * grad


class Momentum:
    def __init__(self, lr=0.001, beta=0.9):
        self.lr = lr
        self.beta = beta
        self.m = None

    def step(self, w, grad):
        if self.m is None:
            self.m = np.zeros_like(grad)

        self.m = self.beta * self.m + (1 - self.beta) * grad
        return w - self.lr * self.m


class RMSProp:
    def __init__(self, lr=0.001, beta=0.9, eps=1e-8):
        self.lr = lr
        self.beta = beta
        self.eps = eps
        self.s = None

    def step(self, w, grad):
        if self.s is None:
            self.s = np.zeros_like(grad)

        self.s = self.beta * self.s + (1 - self.beta) * grad**2
        return w - self.lr * grad / (np.sqrt(self.s) + self.eps)


class Adam:
    def __init__(self, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

        self.m = None
        self.s = None
        self.t = 0

    def step(self, w, grad):
        if self.m is None:
            self.m = np.zeros_like(grad)
            self.s = np.zeros_like(grad)

        self.t += 1

        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.s = self.beta2 * self.s + (1 - self.beta2) * grad**2

        m_corrected = self.m / (1 - self.beta1**self.t)
        s_corrected = self.s / (1 - self.beta2**self.t)

        return w - self.lr * m_corrected / (np.sqrt(s_corrected) + self.eps)


#training
def train(optimizer, batch_size, epochs=10):
    w = np.zeros(D)

    times = [0]
    losses = [loss(w, X, y)]

    tracemalloc.start()
    start_time = time.perf_counter()

    for epoch in range(epochs):
        indices = rng.permutation(N)

        for i in range(0, N, batch_size):
            batch_indices = indices[i:i + batch_size]

            X_batch = X[batch_indices]
            y_batch = y[batch_indices]

            grad = gradient(w, X_batch, y_batch)
            w = optimizer.step(w, grad)

        current_loss = loss(w, X, y)
        losses.append(current_loss)

        elapsed = time.perf_counter() - start_time
        times.append(elapsed)

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    memory = peak / (1024 ** 2)

    return times, losses, memory


#comparing_optimizers
methods = {
    "Batch GD": (BatchGD(lr=0.01), N),
    "Mini-Batch SGD": (MiniBatchSGD(lr=0.001), 64),
    "Momentum": (Momentum(lr=0.001), 64),
    "RMSProp": (RMSProp(lr=0.001), 64),
    "Adam": (Adam(lr=0.01), 64)
}

results = {}

for name, (optimizer, batch_size) in methods.items():
    print("Running:", name)

    times, losses, memory = train(optimizer, batch_size)

    results[name] = (times, losses, memory)

    print("Final loss:", round(losses[-1], 6))
    print("Training time:", round(times[-1], 3), "seconds")
    print("Peak memory:", round(memory, 2), "MB")
    print()


#plotloss
plt.figure(figsize=(9, 5))

for name, data in results.items():
    times, losses, memory = data
    plt.plot(times, losses, marker="o", label=name)

plt.xlabel("Time (seconds)")
plt.ylabel("Loss")
plt.title("Loss vs Training Time")
plt.yscale("log")
plt.legend()
plt.grid()
plt.tight_layout()

plt.savefig("experiment2_loss_time.png", dpi=150)
plt.show()


#plotmemoryusag
plt.figure(figsize=(9, 5))

names = list(results.keys())
memory_values = [results[name][2] for name in names]

plt.bar(names, memory_values)

plt.ylabel("Memory (MB)")
plt.title("Memory Usage")
plt.xticks(rotation=15)

plt.tight_layout()
plt.savefig("experiment2_memory.png", dpi=150)
plt.show()
