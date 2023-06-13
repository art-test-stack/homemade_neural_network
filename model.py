from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score


def loss_function(A, y):
    eps = 1e-15
    loss = - 1 / y.shape[0] * np.sum( y * np.log( A + eps ) + ( 1 - y ) * np.log( 1 - A + eps ))
    return loss


class Model :

    def __init__(self, input_size, output_size, hidden_layers = (32, 32, 32)) -> None:

        np.random.seed(0)
        self.dimensions = [hidden_layers] if type(hidden_layers) == int else list(hidden_layers)
        self.dimensions.insert(0, input_size)
        self.dimensions.append(output_size)

        self.C = len(self.dimensions) - 1
        self.W, self.b = {}, {}
        for c in range(1, self.C + 1):
            self.W[str(c)] = np.random.randn(self.dimensions[c], self.dimensions[c-1])
            self.b[str(c)] = np.random.randn(self.dimensions[c], 1)
            
        self.train_loss, self.train_acc, self.test_loss, self.test_acc = [], [], [], []
        
    def forward_propagation(self, X):

        activations = { 'A0': X }
        Z = {}

        for c in range(1, self.C + 1):
            Z[str(c)] = self.W[str(c)].dot(activations['A' + str(c - 1)]) + self.b[str(c)]
            activations['A' + str(c)] = 1 / ( 1 + np.exp( -Z[str(c)] ))

        return activations

    def back_propagation(self, X, y, activations):

        m = y.shape[1]

        dZ = activations['A' + str(self.C)] - y
        grad = {}

        for c in reversed(range (1, self.C + 1 )):
            grad['dW' + str(c)] = 1 / m * dZ.dot(activations['A' + str(c - 1)].T)
            grad['db' + str(c)] = 1 / m * np.sum(dZ, axis=1, keepdims=True)
            if c > 1 :
                dZ = np.dot(self.W[str(c)].T, dZ) * activations['A'+str(c-1)] * (1 - activations['A'+str(c-1)])
        
        return grad

    def update(self, grad, lr):
        for c in range(1, self.C + 1) : 
            self.W[str(c)] = self.W[str(c)] - lr * grad['dW' + str(c)]
            self.b[str(c)] = self.b[str(c)] - lr * grad['db' + str(c)]

    def predict(self, X):
        activations = self.forward_propagation(X)
        AC = activations['A' + str(self.C)]
        return AC >= 0.5

    def plot_loss_and_acc(self):
        nb_column = 2 if self.test_loss == [] else 3

        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.plot(self.train_loss, label='trainset loss')
        plt.legend()
        if self.test_loss != [] :
            plt.subplot(1, nb_column, 2)
            plt.plot(self.test_loss, label='testset loss')
            plt.legend()
        plt.subplot(1, nb_column, nb_column)
        plt.plot(self.train_acc, label='trainset accuracy')
        plt.plot(self.test_acc, label='testset accuracy')
        plt.legend()
        plt.show()

    def train_model(
            self,
            X_train: np.array,
            y_train: np.array,
            X_test = np.zeros(shape=0), 
            y_test = np.zeros(shape=0),
            lr = 0.1, 
            n_iter = 100, 
        ):

        for epoch in tqdm(range(n_iter)) :

            activations = self.forward_propagation(X_train)
            grad = self.back_propagation(X_train, y_train, activations)
            self.update(grad, lr)

            if epoch%1 == 0 :
                self.train_loss.append(loss_function(activations['A' + str(self.C)], y_train))
                y_pred = self.predict(X_train)
                self.train_acc.append(accuracy_score(y_train.flatten(), y_pred.flatten()))

                if X_test.size != 0 and y_test.size != 0:
                    activations_test =  self.forward_propagation(X_test)
                    self.test_loss.append(loss_function(activations_test['A' + str(self.C)], y_test))
                    y_pred = self.predict(X_test)
                    self.test_acc.append(accuracy_score(y_test.flatten(), y_pred.flatten()))
        
        self.plot_loss_and_acc()
