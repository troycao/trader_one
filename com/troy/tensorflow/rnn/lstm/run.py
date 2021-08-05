# coding:utf-8
import json
import os
import matplotlib.pyplot as plt
from core import data_processor
from core.model import Model
from keras.utils import plot_model

def plot_results(predicted_data, true_data):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label="accuracy_data")
    plt.plot(predicted_data, label="predicted_data")
    plt.legend()
    plt.show()

def plot_results_multiple(predicted_data, true_data, predicted_len):
    fig = plt.figure(facecolor="white")
    ax = fig.add_subplot(111)
    ax.plot(true_data, label="true_data")
    plt.legend()
    for i,data in enumerate(predicted_data):
        padding = [None for p in range(i * predicted_len)]
        plt.plot(padding + data, label="prediction")
    plt.show()

def main():
    configs = json.load(open('config.json', 'r'))
    if not os.path.exists(configs['model']['save_dir']):os.makedirs(configs['model']['save_dir'])

    model = Model()
    my_model = model.build_model(configs)

    plot_model(my_model, to_file='output\model.png', show_shapes=True)

    x, y, x_test, y_test = data_processor.get_multi_train_data(
        os.path.join('data', configs['data']['base_path']),
        configs['data']['train_test_split'],
        configs['data']['columns'],
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

    # predictions = model.predict_sequences_multiplt(x_test, configs['data']['sequence_length'], configs['data']['sequence_length'])
    # predictions = model.predict_sequences_full(x_test, configs['data']['sequence_length'])
    prediction_point = model.predict_point_by_point(x_test)

    # print(prediction_point)
    # print(np.array(predictions).shape)

    # plot_results_multiple(predictions, y_test, configs['data']['sequence_length'])
    plot_results(prediction_point, y_test)

if __name__ == '__main__':
    main()