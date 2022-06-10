# libraries
import os
import numpy as np
from itertools import cycle
from random import shuffle

class LoadFromFolder:
    def __init__(self, X_folder, y_folder, shuffle_=False):
        X_paths = sorted(list(map(lambda x: os.path.join(X_folder, x), os.listdir(X_folder))))
        y_paths = sorted(list(map(lambda x: os.path.join(y_folder, x), os.listdir(y_folder))))
        assert len(X_paths) == len(y_paths), "Inequivalent number of samples and targers."
        self.pairs = [(x,y) for x,y in zip(X_paths, y_paths)]
        del X_paths
        del y_paths
        if shuffle_:
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
            if (self.iteration % self.n_samples) == 0:
                self.shuffle_cycle()

        X_path, y_path = next(self.pair_cycle)
        X = np.load(X_path)
        y = np.load(y_path)
        self.iteration += 1
        return X,y
