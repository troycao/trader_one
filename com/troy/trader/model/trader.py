# coding:utf-8

import tensorflow as tf
from tensorflow import keras
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob

# 定义数据
# print(tf.__version__)

# data_path = glob.glob('G:/data/stock/20210418/*.txt')
# print(data_path[:5])
# data = np.loadtxt('G:/data/stock/20210418/SH#600000.txt')
# data = np.genfromtxt('../data/SH#600000.txt', dtype=None)
# data = np.array(data)
# print(data)

data = pd.read_csv('../data/SH#600000.csv')
print(type(data))
# print(data[:, ['close']])
print(data.head())
print(data.index)
print(data.columns)
print(data['tradeDate'])
print(data.loc[:, ['open','close']])
print(data.iloc[3])
print(data[data.close > 60])
print(data.mean())
upDown = data['open'].apply(lambda x: x>0)
data['upDown'] = upDown
print(data)
# def getTraderSignal(data):
#     print(data[0:3])