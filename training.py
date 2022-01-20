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
import os
from datetime import timedelta
from time import time


@parse_config
def main(config: dict) -> None:
    """ Load data
        Build and compile network
        Optimize network
        Save models and training history
    """


    data_path = config['data_path']
    folder_path = config["base_path"]

    try:
        os.mkdir(folder_path)
    except FileExistsError:
        raise FileExistsError(f"Directory {folder_path} already exists\n" +  
                                "choose different value for experiment_folder parameter")
    os.mkdir(os.path.join(folder_path, "model"))
    with open(config["base_path"] + '/config.yaml', 'w') as file:
        yaml.dump(config, file, default_flow_style=False)

    LoggingConfig(folder_path)

    logging.info("Starting experiment")

    # build network graph
    unet = Unet()
    unet.build(**config['unet']['build'])    
    unet.save_model_graph(filename=folder_path+"/model/network_graph.json")

    # Compile
    # -> Loss and Metrics are hard typed
    # metrics
    f_score = DiceScore(threshold=0.5)
    iou_score_0 = JaccardIndex(threshold=0.5, class_indexes=[0], name="BackGround")
    iou_score_1 = JaccardIndex(threshold=0.5, class_indexes=[1], name="Core")
    iou_score_2 = JaccardIndex(threshold=0.5, class_indexes=[2], name="Edema")
    iou_score_3 = JaccardIndex(threshold=0.5, class_indexes=[3], name="Enhancing")
    
    metrics = [f_score,iou_score_0,iou_score_1,iou_score_2,iou_score_3]
    # losses
    focal = CategoricalFocalLoss(gamma=2)
    #dice = DiceLoss()
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    unet.compile(optimizer = optimizer,
                 loss = focal,
                 metrics = metrics)


    # Callbacks
    checkpoint = ModelCheckpoint(folder_path + "/model/checkpoints/cp.ckpt", save_weights_only=True, save_best_only=True)
    tensorboard = TensorBoard(log_dir=folder_path, histogram_freq=1,write_images=True,)
    callbacks=[checkpoint, tensorboard]
    if  config['unet']['fit'].pop('track_AG') and config['sample_path'] and config['unet']['build']['use_attention']:
        sample = np.load(config['sample_path'])
        callbacks.append(AGCallback(folder_path, sample))



    # predefined datasets combining WT_1 and KO_21 data for training and WT_3 for validation
    # args = [base_path, sample_patterns, target_patterns, subdirs, data_format, logging]
    training_args = [data_path + '/training/samples', data_path + '/training/targets', True]
    validation_args = [data_path + '/validation/samples', data_path + '/validation/targets']
    training = tf.data.Dataset.from_generator(LoadFromFolder,
                                              args=training_args,
                                              output_types=((tf.float32),(tf.float32)),
                                              output_shapes=((128,128,128,3),((128,128,128,4)))
                                              ).batch(config['unet']['fit'].pop('batch_size'))
    validation = tf.data.Dataset.from_generator(LoadFromFolder,
                                                args=validation_args,
                                                output_types=((tf.float32),(tf.float32)),
                                                output_shapes=((128,128,128,3),((128,128,128,4)))
                                                ).batch(1)

    unet.train(training, callbacks=callbacks, validation_dataset=validation, **config['unet']['fit'])

    pd.DataFrame.from_dict(unet.model.history.history).to_csv(folder_path + '/history.csv')


if __name__ == '__main__':
    start = time()
    main()
    end = time()
    delta = end - start
    training_time = str(timedelta(seconds=delta))
    logging.info(f"Training time: {training_time}")
