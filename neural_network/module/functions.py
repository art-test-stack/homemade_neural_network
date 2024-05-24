from neural_network.module.module import Module
import numpy as np


class LeakyReLU(Module):
    def __init__(self, negative_slope: float | int = .2) -> None:
        super().__init__()
        self.negative_slope = negative_slope

    def forward(self, x):
        return np.where(x < 0, self.negative_slope * x, x)
    
    def grad(self, x):
        return np.where(x < 0, self.negative_slope, 1)
    

class ReLU(LeakyReLU):
    def __init__(self) -> None:
        super().__init__(negative_slope=0)


class Linear(LeakyReLU):
    def __init__(self):
        super().__init__(negative_slope = 1)
    

class Sigmoid(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return np.where(x<0, np.exp(x) / (1 + np.exp(x)), 1 / (1 + np.exp(-x)))
    
    def grad(self, x):
        return x * ( 1 - x )
    

class TanH(Module):
    def __init__(self):
        super().__init__()


    def forward(self, x):
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
    
    def grad(self, x):
        return 1 - x ** 2

sigmoid = Sigmoid()
tanh = TanH()
linear = Linear()
relu = ReLU()
leaky_relu = LeakyReLU()

act_function = {
    'sigmoid': sigmoid,
    'tanh': tanh,
    'linear': linear,
    'relu': relu,
    'leaky_relu': leaky_relu,
}

# d_act_function = {
#     'sigmoid': sigmoid.grad,
#     'tanh': tanh.grad,
#     'linear': linear.grad,
#     'relu': relu.grad,
#     'leaky_relu': leaky_relu.grad,
# }

# ------------------OUPUT ACTIVATION FUNCTION------------------

class SoftMax(Module):
    def __init__(self) -> None:
        super().__init__()


    def forward(self, x):
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / np.sum(e_x, axis=1, keepdims=True)

    def grad(self, x):
        diag = []
        product = []
        for mini_x in x:
            diag.append(np.diag(mini_x))
            mini_x = mini_x.reshape(-1, 1)
            product.append(mini_x.dot(mini_x.T))
        diag = np.array(diag)
        product = np.array(product)
        return diag - product

softmax = SoftMax()

type_function = {
    'softmax': softmax,
    None: Linear(),
    'none': Linear()
}

