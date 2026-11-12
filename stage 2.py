import numpy as np

class MiniBatchSGD:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def step(self, w, grad):
        w = np.array(w)
        grad = np.array(grad)
        return w - (self.learning_rate * grad)
