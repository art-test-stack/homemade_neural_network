import numpy as np
from neural_network.module.utils import HiddenLayerConfig
from neural_network.module.functions import * 

class Layer:

    def __init__(self, config: HiddenLayerConfig, lrate, wrt = None, wreg = 0) -> None:
        
        for k, v in config.items():
            setattr(self, k, v)
        if 'lrate' not in config.keys():
            self.lrate = lrate

        self.size = max(1, min(1000, self.size))
        self.wrt = wrt
        self.wreg = wreg
        self.weight_hist = []

    def init_weights(self, size_input):
        if self.wr == 'glorot':
            wr2 = np.sqrt(6) / np.sqrt(size_input + self.size)
            wr1 = -wr2 
        else:
            wrstriped = self.wr.strip('()').split()
            wr1, wr2 = [float(wr) for wr in wrstriped]

        if 'br' in self.__dict__.keys():
            brstriped = self.br.strip('()').split()
            br1, br2 = [float(br) for br in brstriped]
        else:
            br1, br2 = wr1, wr2

        self.W = np.random.uniform(wr1, wr2, size=(size_input, self.size))
        self.b = np.zeros((1, self.size)) if self.wr == 'glorot' and 'br' not in self.__dict__.keys() else np.random.uniform(br1, br2, size=(1, self.size))

    def forward_pass(self, h_last):
        self.hlast = h_last.copy()
        x = h_last.dot(self.W) + self.b
        self.h = act_function[self.act](x)
        return self.h

    def backward_pass(self, J):
        g = J * d_act_function[self.act](self.h)
        self.dw = (self.hlast).T.dot(g) / self.hlast.shape[0] + self.wreg * self.w_reg(self.W)
        self.db = np.mean(g, axis = 0) + self.wreg * self.w_reg(self.b)
        return g.dot(self.W.T)
    
    def w_reg(self, x):
        return linear(x) if self.wrt == 'L2' else np.sign(x) if self.wrt == 'L1' else 0

    def update_weights(self):
        self.weight_hist.append(self.W.copy())

        self.W -= self.lrate * self.dw
        self.b -= self.lrate * self.db 