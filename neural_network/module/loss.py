from neural_network.module.module import Module
import numpy as np

class LossModule(Module):
    def __init__(self):
        super().__init__()
    
    def __call__(self, y, y_pred):
        return self.forward(y, y_pred)
    

class CrossEntropy(LossModule):
    def __init__(self, epsilon: float = 1e-15) -> None:
        super().__init__()
        self.name = "cross-entropy"
        self.eps = epsilon
    
    def forward(self, y, y_pred):
        return - 1 / y.shape[0] * np.sum(y * np.log( y_pred + self.eps))
    
    def grad(self, y, y_pred):
        return np.sum(y) / np.sum(y_pred) - 1 * (y / (y_pred + self.eps))


class MeanSquaredError(LossModule):
    def __init__(self):
        super().__init__()
        self.name = "mean-squared-error"

    def forward(self, y, y_pred):
        return 1 / y.shape[0] * np.sum((y - y_pred) ** 2 ) / 2
    
    def grad(self, y, y_pred):
        return (y - y_pred)


class MSE(MeanSquaredError):
    def __init__(self):
        super().__init__()

    
cross_entropy = CrossEntropy()
mse = MeanSquaredError()

loss = {
    'cross_entropy': cross_entropy,
    'mse': mse
}

# d_loss = {
#     'cross_entropy': cross_entropy.grad,
#     'mse': mse.grad
# }

# def d_ce_s(y, y_pred):
#     return - (y - np.sum(y, axis=1, keepdims=True) * y_pred)