from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Conv2D, Flatten
from tensorflow.keras.callbacks import EarlyStopping
import os
from tensorflow import keras
import numpy as np


def get_ae_model(model_type='nn', input_dim=(8, 14, 3), encoding_dim=24 * 7):
    if model_type == 'nn':
        
        input_layer = Input(shape=(input_dim,))
        encoded = Dense(15, activation='relu')(input_layer)
        encoded = Dense(10, activation='relu')(encoded)
        encoded = Dense(encoding_dim, activation='relu', name='latent')(encoded)
        decoded = Dense(10, activation='relu')(encoded)
        decoded = Dense(15, activation='relu')(decoded)
        decoded = Dense(input_dim, activation='sigmoid')(decoded)
        encoder = Model(input_layer, encoded)
        autoencoder = Model(input_layer, decoded)
    elif model_type == 'cnn':
        input_layer = Input(shape=input_dim)
        encoded = Conv2D(4, (3, 3), activation='relu', padding='same')(input_layer)
        encoded = Conv2D(3, (3, 3), activation='relu', padding='same')(encoded)
        latent = Flatten(name='latent')(encoded)
        decoded = Conv2D(4, (3, 3), activation='relu', padding='same')(encoded)
        decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(decoded)
        encoder = Model(input_layer, latent)
        autoencoder = Model(input_layer, decoded)
    else:
        raise NotImplementedError()
    autoencoder.compile(optimizer='adam', loss='mse')
    return encoder, autoencoder


class Extractor:
    def __init__(self, save_path=None):
        self.callbacks = [EarlyStopping(monitor='loss', patience=3)]
        self.epochs = 100
        self.batch_size = 512
        self.existing_models = {}
        self.save_path = save_path

    def train(self, data, model_type, normalisation=None, input_dim=(8, 14, 3), encoding_dim=24 * 7):
        if self.save_path is not None:
            try:
                encoder = keras.models.load_model(os.path.join(self.save_path, model_type + str(normalisation)) + '.h5')
                self.existing_models[model_type + str(normalisation)] = encoder
            except (OSError, FileNotFoundError):
                pass
        if model_type + str(normalisation) in self.existing_models:
            return
        print('Train Extractor: ', model_type, normalisation)
        encoder, model = get_ae_model(model_type, input_dim, encoding_dim)
        # data = data.reshape(data.shape[0], -1)
        # data = normalise(data, normalisation)
        if model_type == 'nn':
            data = data.reshape(data.shape[0], -1)
        else:
            # data = data.reshape([data.shape[0], 8, 14, 3])
            data = data.transpose(0, 2, 3, 1)
        model.fit(data, data, epochs=self.epochs, batch_size=self.batch_size, callbacks=self.callbacks, verbose=0)
        self.existing_models[model_type + str(normalisation)] = encoder
        encoder.save(os.path.join(self.save_path, model_type + str(normalisation)) + '.h5')

    def transform(self, data, model_type, normalisation=None):
        if model_type == 'nn':
            data = data.reshape(data.shape[0], -1)
        else:
            # data = data.reshape([data.shape[0], 8, 14, 3])
            data = data.transpose(0, 2, 3, 1)
        return self.existing_models[model_type + str(normalisation)].predict(data)
