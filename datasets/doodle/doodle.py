from datasets.dataset import Dataset
from datasets.utils import *
import datasets.doodle.doodler_forall as dd

from configs.settings import * 

import numpy as np


class Doodle(Dataset):

    def __init__(self, address: str = None) -> None:
        super().__init__(name="doodle", address=address)


    def _create_dataset(self,
                    dataset_size = DATASET_SIZE, image_size = 30, noise = .1, wr=[0.2,0.4],hr=[0.2,0.4],
                    flattening = False, centered = False, image_types = dd._doodle_image_types_, show_images = False):
        
        self._generate_dataset_folder_name(dataset_size, len(image_types), image_size)
        
        dataset = dd.gen_doodle_cases(count=dataset_size,rows=image_size,cols=image_size,imt=image_types, 
                    hr=hr,wr=wr, nbias=noise,cent=centered, show=False,
                    one_hots=True,auto=False, flat=flattening,bkg=0, d4=False, fog=1, fillfrac=None, fc=(1,1),
                    gap=1,multi=False,mono=True, dbias=0.7,poly=(4,5,6))
        
        self._split_dataset(dataset)
        self._save_dataset()
        if show_images: self.show_img_cases()
    
