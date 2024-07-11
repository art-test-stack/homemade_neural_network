import numpy as np


class Module():
    def __init__(self) -> None:
        self.no_grad = False
        self.last_x = None
    
    def __call__(self, x):
        self.last_x = x if not self.no_grad else self.last_x
        return self.forward(x)
    
    def forward(self, x):
        NotImplementedError("Implement self forward method")

    def grad(self, x):
        NotImplementedError("Implement self grad method")

    def backward(self, J):
        g = J * self.grad(self.last_x)
        if not self.no_grad:
            self.dw = (self.last_x).T.dot(g) / self.last_x.shape[0] + self.wreg * self.w_reg(self.W)
            self.db = np.mean(g, axis = 0) + self.wreg * self.w_reg(self.b)
        return g