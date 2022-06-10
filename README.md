# 3D_UNET for semantic segmentation
    
* This repository offers configurable U-Net-like network for volumetric data segmentation writen in Python. 
* More complex network blocks such as attention gating are included as well as variety of loss functions and matrices for evaluation.
* For simplicity, all the scripts are configured via. .yaml. Few exceptions must be configured directly in script by user.


## Setup environment
* Compatible with Python version 3.+
``` bash
python -m venv name_of_your_venv
```
On Windows
``` bash
./name_of_your_venv/activate.bat
```
on Linux
``` bash
./name_of_your_venv/activate
``` 
then
``` bash
python -m pip install -r requirements.txt
```

## How to use created scripts 

Everything is controlled by 5 main scripts configured via. <b>. yaml</b> in <b>./config </b> these are:
* preprocessing.py
* training.py
* predict.py
* evaluate.py
* fmap_extract.py

    These file are then executed via command line
    ``` bash
    python scripts_name.py --config ./config/config_name.py
    ```

    <b>Note:</b> It is important to follow conventions of preprocessing and storing data to be able to exploit all the functionality.
 
### Preprocessing.py
* since preprocessing is specific for different task, only base structure of this script has been preserved.
    
    <b>Note:</b> For compatibility with other scripts, at the end of preprocessing training pairs should be divided to 4 separate folders

        /training_samples
        /training_targets
        /validation_samples
        /validation_targets
    <b>Note:</b> Training data pairs should be named according to the following convention
        
        ./training_samples
            sample_001.npy
            sample_002.npy
            sample_003.npy
        ./training_targets
            target_001.npy
            target_002.npy
            target_003.npy
    
### training.py
* this script build, compiles and trains network. Network graph and weights are saved via. Checkpoint callback, unless otherwise defined
* majority of parameters is configured via training.yaml with few exceptions

    <b>Note:</b> Loss functions, metrics, optimizer and callbacks must be defined manually in the script. (lines 58-81)
* Specialized loss function and metrics can be imported from 

        ./src/models/internals/losses.py
        ./src/models/internals/metrics.py
    <b>Losses:</b> JaccardLoss (IoULoss), DiceLoss, BinaryFocalCrossentropy, CategoricalFocalcrossentropy

    <b>Metrics:</b> JaccardScore (IoUScore), DiceScore, Recall, Precision
* see <b>training.yaml</b> and <b>src/models/unet_core docs.</b> to exploit all the posibilities of U-Net configuration
``` yaml
# folder path where all the result will be saved
base_path: ""
training_sample_path:  ""
training_target_path:  ""
validation_sample_path:  ""
validation_target_path:  ""
AG_sample: False  # data sample which will be eventualy used in the AG callback

generator:
  sample_shape: !!python/tuple [~,~,~,~]  # XYZC
  target_shape: !!python/tuple [~,~,~,~]  # XYZC

# see .src/models/unet_core for more information about individual parameters
unet:
  build:
    input_shape_: !!python/tuple [~,~,~,~] # XYZC
    n_depth: ~
    z_depth: ~
    use_attention: ~
    use_transconv: ~
    normalization: ~
    norm_kwargs: ~
    output_probabilities: ~
    output_channels: ~
    last_activation: ~
  fit:
    epochs: ~
    batch_size: ~
    steps_per_epoch: ~
    track_AG: ~       # if AG callbacks should be used
    validation_steps: ~
```

### predict.py
* this scripts creates predictions of saved model
* specify .yaml and run scripts as shown
``` yaml
sample_path: "" # direct folder with preprocessed validation data
n_preds: ~

model_graph:  # path to model graph
model_weights:  # path to model weights, cp.ckpt
seg_path:  # where preds will be store
```
### evaluate.py
* evaluation of model performance
* To configure this scripts follow these instructions
1. define metrices and add metrices´ objects to metric_list
2. for each metric object add corresponding metric name to columns
3. specify parameters in cofiguration file
``` yaml
seg_path: "" # directory with predictions
target_path: "" # directory with corresponding targets
result_path: "" # where .csv and other files will be saved
data:
  n_classes: ~ # number of classes for one_hot_encoding
```

### fmap_extraction.py
* this scripts offers extraction of feature maps from various network layers
* specify following in a .yaml file

feature_amp_extraction.yml
``` yaml
    sample_path: "" # path to input for network
    model_graph_path: "" # path to model graph
    model_weights_path: "" # path to model weights
    results: ""  # directory where fmap will be saved
    layer_names: [] # names of layers which fmaps are required
```