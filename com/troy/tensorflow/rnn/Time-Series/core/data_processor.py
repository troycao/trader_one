import math
import numpy as np
import pandas as pd

class DataLoader():
    def __init__(self, filename, split, clos):
        dataframe = pd.read_csv(filename)
        i_split = int(len(dataframe) * split)
        self.data_train = dataframe.get(clos).values[:i_split]
        self.data_test = dataframe.get(clos).values[i_split:]
        self.len_train = len(self.data_train)
        self.len_test = len(self.data_test)
        self.len_train_windows = None

    def get_test_data(self, seq_len, normalise):
        data_windows = []
        for i in range(self.len_test - seq_len):
            data_windows.append(self.data_test[i:i+seq_len])

        data_windows = np.array(data_windows).astype(float)
        data_windows = self.normalise_windows(data_windows, single_window=False) if normalise else data_windows

        x = data_windows[:, :-1]
        y = data_windows[:, -1, [0]]
        return x,y

    def get_train_data(self, seq_len, normalise):
        data_x = []
        data_y = []
        for i in range(self.len_train - seq_len):
            x,y = self._next_windows(i, seq_len, normalise)
            data_x.append(x)
            data_y.append(y)

        return np.array(data_x),np.array(data_y)

    def _next_windows(self, i, seq_len, normalise):
        window = self.data_train[i:i+seq_len]
        window = self.normalise_windows(window, single_window=True)[0] if normalise else window

        x = window[:-1]
        y = window[-1, [0]]
        return x, y

    '''
        数据做归一化处理
    '''
    def normalise_windows(self, window_data, single_window=False):
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

