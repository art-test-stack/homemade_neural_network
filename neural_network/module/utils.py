from configs.settings import *

import yaml
from enum import Enum

# -------------------------------------------
#           NEURAL NETWORK PARAMETERS
# -------------------------------------------


class Loss(Enum):
    cross_entropy: str = 'cross_entropy'
    mse: str = 'mse'

class Regularizator(Enum):
    L1: str = 'L1'
    L2: str = 'L2'
    none = None

class ActivationFunction(Enum):
    sigmoid: str = 'sigmoid'
    tanh: str = 'tanh'
    relu: str = 'relu'
    linear: str = 'linear'
    
class OutputActFunction(Enum):
    softmax: str = 'softmax'

class HiddenLayerConfig:
    size: int = 10
    act: str = 'relu'
    wr: tuple # | str
    lrate: float = 0.01

class InputLayerConfig:
    input: int = 20

class OutputLayerConfig:
    type: OutputActFunction = 'softmax'


class GlobalConfig:

    def __init__(self, **params):
        self.loss: Loss = Loss.cross_entropy.value
        self.lrate: float = 0.1
        self.wreg: float = 0
        self.wrt: Regularizator = None
        for k, v in params.items():
            setattr(self, k, v)

class LayersConfig:
    input: InputLayerConfig.input
    layers: list[HiddenLayerConfig]
    type: OutputLayerConfig.type

    def __init__(self, **params):
        for k, v in params.items():
            setattr(self, k, v)

# class DatasetConfig: 
#     def __init__(self, **params):
#         self.load = True
#         self.name = 'dataset_2024-02-13_750_9_50'

#         for k, v in params.items():
#             setattr(self, k, v)


def read_config(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

basic_config = read_config(YAML_CONFIG)

def open_config(file_path = YAML_CONFIG):
    config = read_config(file_path)

    global_config = GlobalConfig(**config['GLOBAL']) if 'GLOBAL' in config.keys() and config['GLOBAL'] is not None else GlobalConfig(**basic_config['GLOBAL'])
    layers_config = LayersConfig(**config['LAYERS']) if 'LAYERS' in config.keys() and config['LAYERS'] is not None else GlobalConfig(**basic_config['LAYERS'])
    # dataset_config = DatasetConfig(**config['DATASET']) if 'DATASET' in config.keys() and config['DATASET'] is not None else GlobalConfig(**basic_config['DATASET'])
    return global_config, layers_config #, dataset_config