from tensorflow.keras.models import load_model
from tensorflow_addons.layers import InstanceNormalization, SpectralNormalization
import numpy as np


model = None


def load(model_path):
    global model
    model = load_model(model_path, custom_objects={'InstanceNormalization': InstanceNormalization, 'SpectralNormalization': SpectralNormalization})
    model.summary()


def inference(img):

    upsampled = model.predict(np.expand_dims(img, axis=0))
    if type(upsampled) is list:
        for i in range(len(upsampled)):
            upsampled[i] = ((upsampled[i] + 1) * 127.5).astype(np.uint8)
    else:
        upsampled = ((upsampled + 1) * 127.5).astype(np.uint8)
    return upsampled
