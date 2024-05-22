from neural_network.module.functions import * 
from neural_network.module.loss import * 
from neural_network.module.layer import Layer
from neural_network.module.utils import GlobalConfig, LayersConfig, read_config, open_config

import numpy as np
from matplotlib import pyplot as plt


def has_nan(np_tab, name, ep=''):
    if np.isnan(np_tab).all():
        assert False, f'{name} has nan: {ep}'


def print_empty_tab_line(width=40):
    print(f"|{'-'* width}", end='|\n')

def print_tab_line(row1, row2 = '', width = 40):
    print(f"| {row1}{' '*(9 - len(row1) - 1)} | {row2}{' '*(width - len(row2) - 12)}", end='|\n')

def print_epoch_scores(epoch, loss, val_loss=None):
    epoch, loss = str(epoch), str(loss)
    val_loss_print = f" val loss: {str(val_loss)}" if val_loss else ""

    print(f"Epoch: {epoch},{' '*(5 - len(epoch))} loss: {loss},{' '*(18 - len(loss))}{val_loss_print}")

def wr_function(wrt, w):
    if wrt == 'L1':
        return np.abs(w)
    if wrt == 'L2':
        return w ** 2
    else:
        return 0

class Network:
    def __init__(self,
        global_config: GlobalConfig,
        layers_config: LayersConfig,
        show_config: bool = False) -> None:
        
        for k, v in global_config.__dict__.items():
            setattr(self, k, v)

        self.input = layers_config.input
        self.type = layers_config.type

        self.layers = []
        for c_layer in layers_config.hidden_layers:
            self.layers.append(Layer(c_layer, self.lrate, self.wrt, self.wreg))
            
        self.global_config = global_config
        self.layers_config = layers_config

        self.grad_custom = False

        self.init_weights()

        if show_config: 
            self.print_config()

    def forward_pass(self, X_train):
        h = X_train

        for layer in self.layers:
            h = layer.forward_pass(h)

        return type_function[self.type](h)
    
    def backward_pass(self, y_train, y_pred):
        dL = d_loss[self.loss](y_train, y_pred)
        dy = softmax.grad(y_pred) if self.type == 'softmax' else 1

        J = np.einsum('ik,ikj->ij', dL, dy) if self.type == 'softmax' else dL

        # J = d_ce_s(y_train, y_pred) if self.grad_custom else J
        # if self.type == 'softmax' and self.loss == 'cross_entropy' else J

        for layer in reversed(self.layers):
            J = layer.backward_pass(J)
    
    def init_weights(self): 
        input_size = self.input ** 2
        for layer in self.layers:
            layer.init_weights(input_size)
            input_size = layer.size

    def update_weights(self):
        for layer in self.layers:
            layer.update_weights()

    def regularization(self):
        penalty = 0
        for layer in self.layers:
            penalty += np.sum(wr_function(self.wrt, layer.W))
        return self.wreg * penalty

    def minibatch(self, X_train, y_train, size_minibatch):
        index = []

        if size_minibatch >= 1:
            return X_train, y_train
        
        while len(index) < len(X_train) * size_minibatch:
            idx = np.random.randint(0, len(X_train))
            if idx not in index: index.append(idx)
        index.sort()
        return X_train[index], y_train[index]


    def fit(self, 
        X_train, y_train, 
        X_val = [], y_val = [],
        X_test = [], y_test =[], 
        epoch: int = 100, size_minibatch: float = 1,
        verbose = False):
        
        X_train = X_train.reshape(-1, self.input ** 2) if len(X_train.shape)==3 else X_train

        loss_train = []
        loss_val = []

        acc_train = []
        acc_val = []

        val = False
        test = False

        if not ((len(X_val) == 0) or (len(y_val) == 0)):
            X_val = X_val.reshape(-1, self.input ** 2)
            val = True

        if not ((len(X_test) == 0) or (len(y_test) == 0)):
            X_test = X_test.reshape(-1, self.input ** 2)
            test = True
            
        for ep in range(epoch):

            X_batch, y_batch = self.minibatch(X_train, y_train, size_minibatch) if size_minibatch < 1 else (X_train, y_train)

            if val:
                ypred_val = self.forward_pass(X_val)
                loss_val_ep = loss[self.loss](y_val, ypred_val) + self.regularization()
                loss_val.append(loss_val_ep)
                acc_val.append(np.mean(np.argmax(y_val, axis=1) == np.argmax(ypred_val, axis=1)))

            y_pred = self.forward_pass(X_batch)
            loss_train_ep = loss[self.loss](y_batch, y_pred) + self.regularization()

            if verbose:
                index_to_show = np.random.randint(0, len(X_batch) - 1)
                print("Index plotted:", index_to_show)
                print("Network inputs:", X_batch[index_to_show])
                print("Network outputs:", y_pred[index_to_show])
                print("Target values:", y_batch[index_to_show])
                print(f"Loss ({self.loss}) of the random element:", loss[self.loss](y_batch[index_to_show], y_pred[index_to_show]) + self.regularization())
                print(f"Loss ({self.loss}) of the batch:", loss_train_ep)

            self.backward_pass(y_batch, y_pred)
            self.update_weights()

            print_epoch_scores(ep+1, loss_train_ep, loss_val_ep) if val else print_epoch_scores(ep+1, loss_train_ep)
            loss_train.append(loss_train_ep)
            acc_train.append(np.mean(np.argmax(y_batch, axis=1) == np.argmax(y_pred, axis=1)))

        plt_title = "Train and val loss by epoch" if val else "Train loss by epoch"
        plt.plot(range(1, epoch + 1), loss_train, c='b', label='train loss')
        if val: plt.plot(range(1, epoch + 1), loss_val, c='r', label='val loss')
        plt.title(plt_title)
        plt.legend()
        plt.show()

        plt_title = "Train and val accuracy by epoch" if val else "Train accuracy by epoch"
        plt.plot(range(1, epoch + 1), acc_train, c='b', label='train acc')
        if val: plt.plot(range(1, epoch + 1), acc_val, c='r', label='val acc')
        plt.title(plt_title)
        plt.legend()
        plt.show()

        if test:
            ypred_test = self.forward_pass(X_test)
            print('Test loss:', loss[self.loss](y_test, ypred_test) + self.regularization())

        
    def predict(self, X):
        X = X.reshape(-1, self.input ** 2) if len(X.shape)==3 else X
        return self.forward_pass(X)

    def summary(self, width = 40):
        print_empty_tab_line(width)
        print_tab_line('GLOBAL', width=width)
        print_empty_tab_line(width)
        for k, v in self.global_config.__dict__.items():
            if not k in ['input', 'layers', 'type']:
                print_tab_line(k, str(v), width)
        print_empty_tab_line(width)
        print_tab_line('LAYERS', width=width)
        print_empty_tab_line(width)

        print_tab_line('input', str(self.layers_config.input), width=width)
        k = 1
        for layer in self.layers_config.hidden_layers:
            print_tab_line(f"layer {k}", width=width)
            k += 1
            # print_tab_line('', f"Nb params: {layer.W.shape[0]*layer.W.shape[1]}", width=width)
            for key, v in layer.items():
                print_tab_line('', f"{key}: {str(v)}", width=width)
        print_tab_line('type', str(self.layers_config.type), width=width)
        print_empty_tab_line(width)