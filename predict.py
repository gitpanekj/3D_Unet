"""
The config for this file is specified via cmd when running the scripts.
Config files are stored in the config folder.
wrtie to console: python predict.py --config path_to_your_config
"""
import numpy as np
import tensorflow as tf

import os
from tqdm import tqdm

from src.utils import parse_config
from src.models.unet import Unet

@parse_config
def main(config) -> None:
    data_path = config['sample_path']
    n_preds = config['n_preds']
    seg_path = config["seg_path"]
    
    if not os.path.isdir(seg_path):
        os.mkdir(seg_path)

    unet = Unet()
    unet.load_model(config['model_graph'], config['model_weights'])
    validation = sorted(os.listdir(data_path))


    iter = 0


    for sample in tqdm(validation):
        x = np.load(os.path.join(data_path,sample))[np.newaxis, ...]
        pred = unet.model.predict(x)
        pred=tf.argmax(pred, axis=-1)
        pred = np.squeeze(pred)
        np.save(os.path.join(seg_path, sample.replace("sample", "pred")), pred.astype(np.int8))

        del pred
        iter += 1
        if iter == n_preds:
            break


if __name__ ==  '__main__':
    main()
