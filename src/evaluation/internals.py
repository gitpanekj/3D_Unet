from src.helper_functions import _gather_channels, calc_average, threshold_if_specified, get_reduce_axes
import numpy as np
import tensorflow as tf
from typing import Optional, List
# ------------------
# Eval Functions
# ------------------

def confusion_matrix(y_true: np.array, y_pred: np.array, class_indexes: Optional[List[int]]=None, threshold: Optional[float]=None) -> None:
    y_true = _gather_channels(y_true, indexes=class_indexes)
    y_pred = _gather_channels(y_pred, indexes=class_indexes)

    y_pred = threshold_if_specified(y_pred, threshold=threshold)

    tp = []
    fp = []
    tn = []
    fn = []
    for index in class_indexes:
        cls_tp = tf.keras.backend.sum(y_pred*y_true)
        cls_fp = tf.keras.backend.sum(y_pred) - cls_tp
        cls_tn = tf.keras.backend.sum(not(bool(y_true))*y_pred)
        cls_fn = tf.keras.backend.sum(y_true) - cls_tp
        tp.append(cls_tp)
        fp.append(cls_fp)
        tn.append(cls_tn)
        fn.append(cls_fn)

    return (tp, fp, tn, fn)