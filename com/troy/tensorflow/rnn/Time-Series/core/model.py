import os
import math
import numpy as np
import datetime as dt
from keras.layers import Dense, Activation, Dropout, LSTM
from keras.models import Sequential, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint


class Model():
    def __init__(self):
        self.model = Sequential()

    def load_model(self, filepath):
        print('[model] loading model form file %s' % filepath)
        self.model = load_model(filepath)

    def build_model(self, configs):
        for layer in configs['model']['layers']:
            neurons = layer['neurons'] if 'neurons' in layer else None
            dropout_rate = layer['rate'] if 'rate' in layer else None
            activation = layer['activation'] if 'activation' in layer else None
            return_seq = layer['return_seq'] if 'return_seq' in layer else None
            input_timesteps = layer['input_timesteps'] if 'input_timesteps' in layer else None
            input_dim = layer['input_dim'] if 'input_dim' in layer else None

            if layer['type'] == 'dense':
                self.model.add(Dense(neurons, activation=activation))
            if layer['type'] == 'lstm':
                self.model.add(LSTM(neurons, input_shape=(input_timesteps, input_dim), return_sequences=return_seq))
            if layer['type'] == 'dropout':
                self.model.add(Dropout(dropout_rate))

        self.model.compile(loss=configs['model']['loss'], optimizer=configs['model']['optimizer'])
        print('[ModeL] Model Compiled')

        return self.model

    def train(self, x, y, epochs, batch_size, save_dir):
        print('model training started')
        print('model % epochs, %s batch size' %(epochs, batch_size))
        save_file_name = os.path.join(save_dir, '%s-e%s.h5' % (dt.datetime.now().strftime('%Y%m%d-%H%M%S'), str(epochs)))

        callbacks = [
            EarlyStopping(monitor='loss', patience=2),
            ModelCheckpoint(filepath=save_file_name, monitor='loss', save_best_only=True)
        ]
        self.model.fit(
            x,
            y,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks
        )
        self.model.save(save_file_name)
        print("model training completed,model save file %s", save_file_name)

