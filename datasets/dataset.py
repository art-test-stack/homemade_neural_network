from datasets.utils import *
from configs.settings import * 

import numpy as np
from pathlib import Path
from datetime import datetime


class Dataset():


    def __init__(self, name: str, data_type: str = "images", address: str = None, config: dict = {}) -> None:

        assert data_type == 'images', f"The kind of data '{data_type}' is not yet implemented for that class"

        self.name = name
        self.address = address
        self.config = config
        self._load_dataset()
        self.data_type = data_type


    def _generate_dataset_folder_name(self, size, nb_classes, n):
        name_folder = f"{self.name}_{str(datetime.now().date())}_{size}_{nb_classes}_{n}"

        path = DATA_FOLDER / name_folder
        if not path.exists():
            path.mkdir()
            self.address = name_folder
        
        k = 1
        folder = name_folder + f"_v{k}"
        path = DATA_FOLDER / folder
        while path.exists():
            k += 1
            folder = name_folder + f"_v{k}"
            path = DATA_FOLDER / folder
        
        self.address = folder
        path.mkdir()

    def _split_dataset(self, dataset):
        assert VAL_SET_PROP + TEST_SET_PROP < 100, "Proportions do not make 100%"

        self.train_size = int(TRAIN_SET_PROP * len(dataset[0]) / 100)
        self.test_size = int(TEST_SET_PROP * len(dataset[0]) / 100)
        self.val_size = len(dataset[0]) - self.train_size - self.test_size 

        inputs, targets, labels, _, _ = dataset

        X_train = inputs[:self.train_size]
        Y_train = targets[:self.train_size]
        labels_train = labels[:self.train_size]

        X_test = inputs[self.train_size:self.train_size + self.test_size]
        Y_test = targets[self.train_size:self.train_size + self.test_size]
        labels_test = labels[self.train_size:self.train_size + self.test_size]

        X_val = inputs[self.train_size + self.test_size:]
        Y_val = targets[self.train_size + self.test_size:]
        labels_val = labels[self.train_size + self.test_size:]
        
        self.train_set = (X_train, Y_train, labels_train)
        self.test_set = (X_test, Y_test, labels_test)
        self.val_set = (X_val, Y_val, labels_val)


    def _load_dataset_from_folder_name(self): 

        address = DATA_FOLDER / self.address

        self.train_set = load_subsets(address, 'train')
        self.test_set = load_subsets(address, 'test')
        self.val_set = load_subsets(address, 'val')

        self.train_size = len(self.train_set)
        self.test_size = len(self.test_set)
        self.val_size = len(self.val_set)


    def _create_dataset(self):
        NotImplementedError()


    def _load_dataset(self):

        if self.address == None or not DATA_FOLDER.joinpath(str(self.address)).exists():
            self._create_dataset()

        self._load_dataset_from_folder_name()


    def _save_dataset(self):
        print("address:",  self.address)
        folder_path = DATA_FOLDER / self.address
        
        X_train, Y_train, labels_train = self.train_set
        X_test, Y_test, labels_test = self.test_set
        X_val, Y_val, labels_val = self.val_set

        save_data_in_folder(folder_path, 'train', X_train, Y_train, labels_train)
        save_data_in_folder(folder_path, 'test', X_test, Y_test, labels_test)
        save_data_in_folder(folder_path, 'val', X_val, Y_val, labels_val)
            

    def show_img_cases(self, nb_cases = 10):
        NotImplementedError()
