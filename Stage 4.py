import numpy as np


class RMSProp:
    def __init__(self, learning_rate=0.5, beta2=0.99, epsilon=1e-8):
        self.eta = learning_rate
        self.beta2 = beta2
        self.epsilon = epsilon
        self.s = None

    def step(self, w, grad):
        w = np.array(w)
        grad = np.array(grad)

        if self.s is None:
            self.s = np.zeros_like(w)

        self.s = self.beta2 * self.s + (1 - self.beta2) * (grad**2)
        return w - (self.eta / (np.sqrt(self.s) + self.epsilon)) * grad
