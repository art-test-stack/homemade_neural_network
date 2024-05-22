from neural_network.module.utils import open_config
from neural_network.module.network import Network
from datasets.doodle.doodle import Doodle

if __name__ == '__main__':
    global_config, layers_config = open_config()
    dataset = Doodle()

    X_train, y_train, _ = dataset.train_set
    X_val, y_val, _ = dataset.val_set
    X_test, y_test, _ = dataset.test_set

    model = Network(global_config, layers_config)

    model.fit(X_train, y_train, X_val, y_val, X_test, y_test)
    
