import numpy as np

class AdamOptimizer:
    def __init__(self, lr, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = None
        self.s = None
        self.t = 0

    def step(self, w, grad):
        if self.m is None:
            self.m = np.zeros_like(w) #defining m
            self.s = np.zeros_like(w)
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.s = self.beta2 * self.s + (1 - self.beta2) * grad**2
        m_hat = self.m / (1 - self.beta1**self.t)
        s_hat = self.s / (1 - self.beta2**self.t)
        return w - self.lr * m_hat / (np.sqrt(s_hat) + self.eps)