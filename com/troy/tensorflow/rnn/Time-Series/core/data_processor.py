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

