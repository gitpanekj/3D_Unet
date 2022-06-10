"""
The config for this file is specified via cmd when running the scripts.
Config files are stored in the config folder.
to run this script type 'python evaluate.py --config path_to_your_config'  to your console
"""
import numpy as np
from tqdm import tqdm
import pandas as pd
pd.set_option("display.float_format", '{:.2f}'.format)
import os
import re

from src.models.internals.metrics import JaccardIndex, Recall, Precision, DiceScore
from preprocessing import one_hot_encode
from src.utils import parse_config


@parse_config
def main(config) -> None:

    ## DEFINE METRICS ##
    jaccard_case1 = JaccardIndex(class_indexes=[0])
    jaccard_case2 = JaccardIndex(class_indexes=[1])
    recall_case1 = Recall()
    
    # metric_list should containt all the metrics to be used
    metric_list = [jaccard_case1, jaccard_case2, recall_case1]
    # coulmns should contain names for all the defined metrics
    columns = ["name"] + ["jaccard_case1", "jaccard_case2", "recall_case1"]
    
    results = pd.DataFrame(columns=columns)
    # prepare data
    
    preds = sorted(os.listdir(config['seg_path']))
    targets = sorted(os.listdir(config['target_path']))


    for i in tqdm(range(len(preds))):
        # loading data
        y = np.load(os.path.join(config["target_path"], targets[i]))
        x = one_hot_encode(np.load(os.path.join(config['seg_path'], preds[i])), classes=config['data']['n_classes'])

        ## Following lines merge n_classes to 1
        #location = np.sum(y[...,1:], axis=-1, keepdims=True) # merge channels
        #pred_flat = np.sum(x[...,1:], axis=-1, keepdims=True) # merge channels

        # save data to Pandas table
        result_dct = {"name":preds[i]}
        for name, metric in zip(columns, metric_list(y,x)):
            result_dct.update({name:metric})
        results.loc[i] = {}

    results.to_csv(config["result_path"])

if __name__ == '__main__':
    main()

