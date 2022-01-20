import numpy as np
import tensorflow as tf

import pandas as pd
import tifffile

import os
import argparse
import yaml
import logging
from typing import Optional, Tuple
import json

from src.data_handling.generator import LoadFromFolder
from src.models.internals.metrics import JaccardIndex
from src.utils import parse_config
from src.models.unet import Unet

@parse_config
def main(config) -> None:
    """ Evaluate model in WT_3 data """

    data_path = config['data_path']
    n_preds = config['n_preds']
    seg_path = config["seg_path"]
    predictions = np.zeros((n_preds,128,128,128)).astype('uint8')
    targets =  np.zeros((n_preds,128,128,128)).astype('uint8')
 
    os.mkdir(seg_path)

    unet = Unet()
    unet.load_model(config['model_graph'], config['model_weights'])

    metric = JaccardIndex()

    history = {"frame":[],f"{metric.name}":[]}
    validation_args = [data_path + '/validation/samples', data_path + '/validation/targets']
    validation = tf.data.Dataset.from_generator(LoadFromFolder,
                                                args=validation_args,
                                                output_types=((tf.float32), (tf.float32)),
                                                output_shapes=((128, 128, 128, 3), ((128, 128, 128, 4)))
                                                ).batch(1)
    iter = 0
    for x,y in validation:
        pred = unet.model.predict(x)
        metric.update_state(y, pred)
        history['frame'].append(iter)
        history[f"{metric.name}"].append(metric.result())
        predictions[iter] = tf.argmax(pred, axis=-1)
        targets[iter] = tf.argmax(pred, axis=-1)
        del pred
        iter += 1
        if iter == n_preds:
            break

    pd.DataFrame.from_dict(history).to_csv(config['base_path'] + '/' + 'iou_per_frame.csv')
    # save segmentations
    metadata = {'hyperstack': True, 'axes': 'TZYX'} # format data
    tifffile.imwrite(file=seg_path + f"/segmentations.tif",
                     data=predictions,
                     imagej=True,
                     metadata=metadata)
    tifffile.imwrite(file=seg_path + f"/targets.tif",
                     data=targets,
                     imagej=True,
                     metadata=metadata)

if __name__ ==  '__main__':
    main()
