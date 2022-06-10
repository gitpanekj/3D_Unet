import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model

from src.models.unet import Unet
from src.utils import parse_config

@parse_config
def main(config) -> None:
    """ Extract feature maps from network """

    sample_path  = config['sample_path']
    model_graph_path = config['model_graph_path']
    model_weights_path = config['model_weights_path']
    results = config['results']
    

    unet = Unet()
    unet.load_model(model_graph_path, model_weights_path)
    model = unet.model
    del unet
    
    
    data = np.load(sample_path)
    # extension of tensor dimension is required for capability with network input_shape
    data = data[np.newaxis, ...]
    

    #this list contains names of layers from which should be extracted feature maps
    layer_names = ['tf.nn.relu_12',
                   'tf.nn.relu_15',
                   'tf.nn.relu_18',
                   'tf.nn.relu_21']

    # extract and save
    for name in layer_names:
        for layer in model.layers:
            if layer.name == name:
                partial_model = Model(inputs=[model.inputs], outputs=[layer.output])
                continue
        pred = partial_model.predict(data)
        del partial_model
        np.save(results+f'/{name}', pred)
        del pred

if __name__ ==  '__main__':
    main()
