# libraries
import nibabel as nib
import os
import numpy as np
import tensorflow.keras.backend as K
import tensorflow as tf
from itertools import cycle
from typing import Optional, List
from src.helper_functions import standardization, normalization
import logging
from random import shuffle

class DataGenerator:
    def __init__(self, path: bytes, target_pattern: bytes, sample_patterns: Optional[List[bytes]]=None, class_idxs: Optional[List[int]]=None, logging: bool=False):
        self.logging = logging
        self.dir_cycle = self.create_dir_cyle(path)
        self.x_patterns = [sample_patterns] if isinstance(sample_patterns, str) else list(sample_patterns)
        self.y_pattern = target_pattern
        if isinstance(class_idxs, type(None)):
            self.class_idxs = [0,1,2,3]
        else:
            self.class_idxs = class_idxs if isinstance(class_idxs, list) else list(class_idxs)

    def __iter__(self):
        return self
    
    def __next__(self):
        return self.next()
    
    def next(self):
        """ Defined pipeline """
        dir_ = next(self.dir_cycle)
        x_paths, y_path = self.sort_filenames(dir_)
        x_paths = list(map(lambda x: os.path.join(dir_,x),x_paths))
        y_path = os.path.join(dir_,y_path)
        x,y = self.create_training_pair(x_paths, y_path)
        return x,y
    

    def create_dir_cyle(self, path: bytes) -> cycle:
        """ Creates cycle with directories where samples are saved """
        path = os.path.abspath(path)
        dirs = list(filter(lambda x: os.path.isdir(os.path.join(path,x)), os.listdir(path)))
        dirs = [os.path.join(path, d) for d in dirs]
        dirs = sorted(dirs)
        if self.logging:
            logging.info(f"Directory cycle consists of {len(dirs)}")
        return cycle(dirs)
    
    def sort_filenames(self, dir):
        """ Sort filenames in the given directory and
            returns x_paths and y_paths
        """
        
        channels = []
        target = None

        x_patterns = self.x_patterns.copy()
        # sample data
        files = os.listdir(dir)
        for i, file in enumerate(files):
            for j, pattern in enumerate(x_patterns):
                if file.find(pattern) > -1:
                    channels.append(file)
        
        # target
        for file in files:
            if file.find(self.y_pattern) > -1:
                target = file
        
        return channels, target

    def create_training_pair(self, x_paths, y_path):
        """ Creates training pair """
        x_volumes = [self.resize_if_specified(self.load_data(path), True) for path in x_paths]
        if len(x_volumes) == 1:
            x_stack = x_volumes[0]
        else:
            x_stack = self.stack_volumes(x_volumes)
        del x_volumes

        y_data = self.load_data(y_path)
        y_data = self.resize_if_specified(y_data, True)
        encoded_y = self.one_hot_encode(y_data)     
        del y_data

        x,y = self.preprocessing(x_stack, encoded_y)
        return x,y

    def load_data(self, path):
        """ Loads data """
        if isinstance(path, bytes):
            path = path.decode()
        data = nib.load(path)
        data = data.get_fdata()
        return data

    def stack_volumes(self, volumes):
        """ Stacks volumes along channels axis """
        stack = np.stack(volumes, axis=-1)
        return stack
    
    def one_hot_encode(self, y):
        encoded = np.zeros((128,128,128,len(self.class_idxs)))
        for i, cls in enumerate(self.class_idxs):
            encoded[..., i] = (y == cls)

        return encoded

    def preprocessing(self,x,y):
        """ Preprocessing """

        x = K.cast(x, K.floatx())
        y = K.cast(y, K.floatx())

        x = normalization(x, keepdims=True)
        ### preprocessing
        
        return x,y
    
    def resize_if_specified(self,x, resize=False):
        """ Resize tensor """
        if resize:
            x = x[56:184, 56:184, 13:141]
        return x

class LoadFromFolder:
    def __init__(self, X_folder, y_folder, shuffle_=False):
        X_paths = sorted(list(map(lambda x: os.path.join(X_folder, x), os.listdir(X_folder))))
        y_paths = sorted(list(map(lambda x: os.path.join(y_folder, x), os.listdir(y_folder))))
        assert len(X_paths) == len(y_paths), "Inequivalent number of samples and targers."
        self.pairs = [(x,y) for x,y in zip(X_paths, y_paths)]
        del X_paths
        del y_paths
        shuffle(self.pairs)
        self.n_samples = len(self.pairs)
        self.pair_cycle = cycle(self.pairs)
        self.iteration = 0
        self.shuffle = shuffle_

    def __iter__(self):
        return self
    
    def __next__(self):
        return self.next()
    
    def shuffle_cycle(self):
        shuffle(self.pairs)
        self.pair_cycle = cycle(self.pairs)
        self.iteration = 0
    
    def next(self):
        if self.shuffle:
            if self.iteration == self.n_samples:
                self.shuffle_cycle()

        X_path, y_path = next(self.pair_cycle)
        X = np.load(X_path)
        y = np.load(y_path)
        self.iteration += 1
        return X,y