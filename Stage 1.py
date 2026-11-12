import numpy as np

class BatchGD:
    def __init__(self, learning_rate):
        # Stage 1 requires 0 extra vectors of state memory[cite: 4].
        self.learning_rate = learning_rate

    def step(self, w, grad):
        w = np.array(w)
        grad = np.array(grad)
        
        return w - (self.learning_rate * grad)
