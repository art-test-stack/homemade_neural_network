import numpy as np


def loss_function(A, y):
    eps = 1e-15
    loss = - 1 / y.shape[0] * np.sum( y * np.log( A + eps ) + ( 1 - y ) * np.log( 1 - A + eps ))
    return loss

def normalize_images(X_train, y_train, X_test, y_test):
    max = X_test.max() if X_test.max() > X_train.max() else X_train.max()
    X_train = X_train.reshape( -1, X_train.shape[0]) / max
    X_test = X_test.reshape(-1, X_test.shape[0]) / max
    y_train = y_train.reshape(-1, y_train.shape[0])
    y_test = y_test.reshape(-1, y_test.shape[0])
    return X_train, y_train, X_test, y_test