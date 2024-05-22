from datasets.utils import *
from datasets.doodle.doodle import Doodle
from pathlib import Path


if __name__ == '__main__':
    dataset = Doodle()
    dataset.show_img_cases()