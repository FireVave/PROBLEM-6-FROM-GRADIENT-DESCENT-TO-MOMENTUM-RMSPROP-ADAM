import numpy as np


class MomentumOptimizer:
    def __init__(self, learning_rate, beta):
        self.learning_rate = learning_rate
        self.beta = beta
        self.velocity = None

    def step(self, w, grad):
        if self.velocity is None:
            self.velocity = np.zeros_like(w, dtype=float)

        self.velocity = self.beta * self.velocity + (1 - self.beta) * grad
        return w - self.learning_rate * self.velocity