"""
The config for this file is specified via cmd when running the scripts.
Config files are stored in the config folder.
To run this script type  'python training.py --config path_to_your_config'  to you console
"""
# src imports
from src.models.unet import Unet
from src.data_handling.generator import LoadFromFolder
from src.models.internals.losses import DiceLoss, CategoricalFocalLoss
from src.models.internals.metrics import DiceScore, JaccardIndex
from src.models.internals.callbacks import AGCallback
from src.utils import parse_config, LoggingConfig
# libraries
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
import numpy as np
import pandas as pd


import logging
import yaml
from os import join, mkdir
from datetime import timedelta
from time import time


@parse_config
def main(config: dict) -> None:
    """ Load data
        Build and compile network
        Optimize network
        Save models and training history
    """


    folder_path = config["base_path"]

    try:
        mkdir(folder_path)
    except FileExistsError:
        raise FileExistsError(f"Directory {folder_path} already exists\n" +  
                                "choose different value for experiment_folder parameter")
    mkdir(join(folder_path, "model"))
    with open(config["base_path"] + '/config.yaml', 'w') as file:
        yaml.dump(config, file, default_flow_style=False)

    LoggingConfig(folder_path)

    logging.info("Starting experiment")


    ## BUILDING NETWORK GRAPH ##
    unet = Unet()
    unet.build(**config['unet']['build'])    
    unet.save_model_graph(filename=join(folder_path,*("model","network_graph.json")))


    ## COMPILATION OF NETWORK GRAPH ##
    logging.info("Compiling network graph")
    # metrics
    f_score = DiceScore(threshold=0.5)
    metrics = [f_score]
    # losses
    focal = CategoricalFocalLoss(gamma=2)
    dice = DiceLoss()
    # compile
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    unet.compile(optimizer = optimizer,
                 loss = lambda y_true,y_pred: focal(y_true, y_pred) + dice(y_true,y_pred),
                 metrics = metrics)


    ## CALLBACKS ##
    logging.info("Defining callbacks") 
    checkpoint = ModelCheckpoint(join(folder_path, *("model","checkpoints","cp.ckpt")), save_weights_only=True, save_best_only=True)
    callbacks=[checkpoint]

    # ignore lines 76-78 unless an attention gate structure is used, for more info see README.md
    if  config['unet']['fit'].pop('track_AG') and config['AG_sample'] and config['unet']['build']['use_attention']:
        sample = np.load(config['AG_sample'])
        callbacks.append(AGCallback(folder_path, sample))


    ## DEFINING DATA GENERATORS ##
    logging.info("Defininf data generators") 
    training_args = [config['training_sample_path'], config['training_target_path']]
    validation_args = [config['validation_sample_path'], config['validation_target_path']]
    sample_shape, target_shape = config['generator']['sample_shape'], config['generator']['target_shape']

    training = tf.data.Dataset.from_generator(LoadFromFolder,
                                              args=training_args,
                                              output_types=((tf.float32),(tf.float32)),
                                              output_shapes=(sample_shape, target_shape)
                                              ).batch(config['unet']['fit'].pop('batch_size'))
    validation = tf.data.Dataset.from_generator(LoadFromFolder,
                                                args=validation_args,
                                                output_types=((tf.float32),(tf.float32)),
                                                output_shapes=(sample_shape, target_shape)
                                                ).batch(1)


    ## TRAINING ##
    logging.info("Training started") 
    unet.train(training, callbacks=callbacks, validation_dataset=validation, **config['unet']['fit'])

    ## SAVING TRAINING HISTORY
    logging.info("Saving training history")
    unet.save_training_history(join(folder_path, 'history.csv'))


if __name__ == '__main__':
    start = time()
    main()
    end = time()
    delta = end - start
    training_time = str(timedelta(seconds=delta))
    logging.info(f"Training time: {training_time}")
