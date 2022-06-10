""" Content of this file is specific for different tasks.
    Therefore, only standard preprocessing functions are implemented.
    Follow the structure of this file to keep uniformity across all -
    the scripts in this repository.

    Note: At the end of preprocessing training pairs should be divided to 4 folder
    /training_samples
    /training_targets
    /validation_samples
    /validation_targets
    This will allow user to easily use predefined configuration files
"""
import numpy as np

from src.utils import parse_config
from src.helper_functions import one_hot_encode, standardization, normalization, min_max_scale



@parse_config
def main(config):
    pass

if __name__ == '__main__':
    main()