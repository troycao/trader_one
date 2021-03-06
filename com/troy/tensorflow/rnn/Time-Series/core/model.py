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

    def predict_point_by_point(self,data):
        print("model predicting point by point")
        prdiceted = self.model.predict(data)
        prdiceted = np.reshape(prdiceted, (prdiceted.size,))
        return prdiceted

    def predict_sequences_multiplt(self, data, window_size, prediction_len, debug=False):
        if debug == False:
            print("model predicting sequences multiple...")
            prediction_seqs = []
            for i in range(int(len(data)/prediction_len)):
                curr_frame = data[i*prediction_len]
                predicted = []
                for j in range(prediction_len):
                    predicted.append(self.model.predict(curr_frame[np.newaxis, :, :])[0, 0])
                    curr_frame =curr_frame[1:]
                    curr_frame = np.insert(curr_frame, [window_size-2], predicted[-1], axis=0)
                prediction_seqs.append(predicted)
            return prediction_seqs
        else:
            print("model predicting sequences multiple...")
            prediction_seqs = []
            for i in range(int(len(data)/prediction_len)):
                print(data.shape)
                curr_frame = data[i * prediction_len]
                predicted = []
                for j in range(prediction_len):
                    predict_result = self.model.predict(curr_frame[np.newaxis, :, :])
                    print(predict_result)
                    final_result = predict_result[0,0]
                    predicted.append(final_result)
                    curr_frame = curr_frame[1:]
                    curr_frame = np.insert(curr_frame, [window_size-2], predicted[-1], axis=0)
                prediction_seqs.append(predicted)
            return  prediction_seqs

    def predict_sequences_full(self, data, window_size):
        print("model predicting seq full...")
        curr_frame = data[0]
        predicted = []
        for i in range(len(data)):
            predicted.append(self.model.predict(curr_frame[np.newaxis,:,:])[0,0])
            curr_frame = curr_frame[1:]
            curr_frame = np.insert(curr_frame, [window_size-2], predicted[-1], axis=0)
        return predicted

    def predict_point_by_point(self, data, debug=False):
        print("model predicting seq potin...")
        predicted = self.model.predict(data)
        print('predicted shape:', np.array(predicted).shape)  # (412L,1L)
        predicted = np.reshape(predicted, (predicted.size,))
        return predicted




