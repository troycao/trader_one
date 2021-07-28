# coding:utf-8
import json
import os
import time
import math
import matplotlib.pyplot as plt
import numpy as np
from core.data_processor import DataLoader
from core.model import Model
from keras.utils import plot_model

def main():
    configs = json.load(open('config.json', 'r'))
    if not os.path.exists(configs['model']['save_dir']):os.makedirs(configs['model']['save_dir'])
    data = DataLoader(
        os.path.join('data', configs['data']['filename']),
        configs['data']['train_test_split'],
        configs['data']['columns']
    )
    model = Model()
    model.build_model(configs)

    x,y = data.get_train_data(
        configs['data']['sequence_length'],
        configs['data']['normalise']
    )

    print(x.shape)
    print(y.shape)

    print(configs['training']['batch_size'])
    print(configs['model']['save_dir'])
    model.train(x,
                y,
                configs['training']['epochs'],
                configs['training']['batch_size'],
                configs['model']['save_dir']
                )


if __name__ == '__main__':
    main()