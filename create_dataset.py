from datasets.utils import *
from datasets.doodle.doodler_forall import _doodle_image_types_
from pathlib import Path

_doodle_image_types = _doodle_image_types_

if __name__ == '__main__':
    dataset_name = create_dataset(
        dataset_size=1000, 
        image_size=30, 
        image_types=_doodle_image_types, 
        proportions=(50, 30, 20),
        noise=0,
        centered=True
    )
    print("dataset name:", dataset_name)