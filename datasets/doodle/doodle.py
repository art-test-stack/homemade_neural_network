from datasets.dataset import Dataset
from datasets.utils import *
import datasets.doodle.doodler_forall as dd

from configs.settings import * 

import numpy as np


class Doodle(Dataset):

    def __init__(self, address: str = None) -> None:
        super().__init__(name="doodle", address=address)


    def _create_dataset(self,
                    dataset_size = DATASET_SIZE, image_size = DOODLE_IMAGE_SIZE, noise = DOODLE_NOISE, wr=[0.2,0.4],hr=[0.2,0.4],
                    flattening = False, centered = DOODLE_IS_CENTERED, image_types = dd._doodle_image_types_, show_images = False):
        
        self._generate_dataset_folder_name(dataset_size, len(image_types), image_size)
        
        dataset = dd.gen_doodle_cases(count=dataset_size,rows=image_size,cols=image_size,imt=image_types, 
                    hr=hr,wr=wr, nbias=noise,cent=centered, show=False,
                    one_hots=True,auto=False, flat=flattening,bkg=0, d4=False, fog=1, fillfrac=None, fc=(1,1),
                    gap=1,multi=False,mono=True, dbias=0.7,poly=(4,5,6))
        
        self._split_dataset(dataset)
        self._save_dataset()
        if show_images: self.show_img_cases()

    def show_img_cases(self, nb_cases = 10):
        subset = self.test_set
        nb_cases = np.minimum(nb_cases, len(subset[0][0]))
        
        images = []
        labels = []

        index = []
        for _ in range(np.minimum(nb_cases, 10)):
            subset = subset[np.random.randint(0, 2)]
            rd_img_index = np.random.randint(0, subset[0].shape[0] - 1)
            while rd_img_index in index:
                rd_img_index = np.random.randint(0, subset[0].shape[0] - 1)
            index.append(rd_img_index)
            images.append(subset[0][rd_img_index])
            labels.append(subset[2][rd_img_index])

        cases = (images, '', labels, '', '')
        dd.show_doodle_cases(cases)