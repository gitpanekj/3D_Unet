import numpy as np
import matplotlib.pyplot as plt

import nibabel as nib

from tqdm import tqdm
import os
import logging



def return_channels(path, channels):
    files = os.listdir(path)
    samples = []
    for file_ in files:
        for i,c in enumerate(channels):
            if file_.endswith(f'{c}.nii'):  
                samples.append(file_)
                channels.pop(i)
                break
    samples = list(map(lambda x: os.path.join(path,x), samples))
    return samples


def return_target(path):
    files = os.listdir(path)
    for file_ in files:
        if file_.endswith('seg.nii'):   return os.path.join(path, file_)

def min_max_scale(data):
    return (data-np.min(data))/(np.max(data)-np.min(data))

def one_hot_encode(data):
    encoded = np.ndarray((128,128,128,4), dtype=np.uint8)
    labels = [[1,0,0,0],
              [0,1,0,0],
              [0,0,1,0],
              [0,0,0,1]]

    for i in range(4):
        encoded[data == i] = labels[i]
    del data
    return encoded

def processing_pipeline(x_paths,y_path):
    x = sorted(x_paths)
    y = y_path
    
    # y_data
    valuable = None
    Y = nib.load(y).get_fdata().astype(np.uint8)
    Y = Y[56:184,56:184,13:141]
    val, counts = np.unique(Y, return_counts=True)
    if (1 - counts[0]/np.sum(counts)) > 0.01:
        valuable = True
        Y[Y == 4] = 3
        Y = one_hot_encode(Y)
    else:
        valuable = False
        logging.info(f"sample_{INDEX}")
        return False

    if valuable:
        # x_data
        stacks = []
        for x_path in x_paths:
            X = nib.load(x_path).get_fdata()
            X = min_max_scale(X)
            X = X[56:184,56:184,13:141].astype(np.float32)
            stacks.append(X)
            del X
        X = np.stack(stacks, axis=-1)
        del stacks

        # save
        np.save(save_path + '/samples' + f'/sample_{INDEX}.npy', X)
        np.save(save_path + '/targets' + f'/target_{INDEX}.npy', Y)
        return True


if __name__ == '__main__':
    
    logging.basicConfig(filename='loggs.log', level=logging.INFO)
    

    base = "/OV-data/JP/BraTS_2020/MICCAI_BraTS2020_TrainingData"
    save_path = "/OV-data/JP/BraTS_2020/training"

    base = os.path.abspath(base)
    samples = list(map(lambda x: os.path.join(base,x), os.listdir(base)))
    samples = list(filter(lambda x: os.path.isdir(x), samples))


    for INDEX,sample in tqdm(enumerate(samples)):
        target = return_target(sample)
        channels = return_channels(sample, ['t1ce','t2','flair'])
        processed = processing_pipeline(channels, target)

