import math
import numpy as np
import pandas as pd
import os

'''
    数据做归一化处理
'''
def normalise_windows(window_data, single_window=False):
    normalise_data = []
    window_data = [window_data] if single_window else window_data
    for window in window_data:
        normalise_window = []
        for col_i in range(window.shape[1]):
            normalise_col = [((float(p) / float(window[0, col_i])) -1) for p in window[:, col_i]]
            normalise_window.append(normalise_col)
        normalise_window = np.array(normalise_window).T
        normalise_data.append(normalise_window)

    return np.array(normalise_data)

def get_multi_train_data(filepath, split, clos, seq_len, normalise):
    data_x = []
    data_y = []

    file_list = os.listdir(filepath)
    for file_name in file_list:
        file = os.path.join(filepath, file_name)
        if os.path.isfile(file):
            dataframe = pd.read_csv(file)
            i_split = int(len(dataframe) * split)
            data_train = dataframe.get(clos).values[:i_split]
            data_test = dataframe.get(clos).values[i_split:]
            len_train = len(data_train)
            len_test = len(data_test)
            for i in range(len_train - seq_len):
                window = data_train[i:i + seq_len]
                window = normalise_windows(window, single_window=True)[0] if normalise else window

                x = window[:-1]
                y = window[-1, [0]]

                data_x.append(x)
                data_y.append(y)

            data_windows = []
            for i in range(len_test - seq_len):
                data_windows.append(data_test[i:i + seq_len])

            data_windows = np.array(data_windows).astype(float)
            data_windows = normalise_windows(data_windows, single_window=False) if normalise else data_windows

            test_x = data_windows[:, :-1]
            test_y = data_windows[:, -1, [0]]

    return np.array(data_x), np.array(data_y), np.array(test_x), np.array(test_y)
