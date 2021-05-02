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

# 准备数据
data = pd.read_csv('../data/SH#600000.csv')
# print(type(data))
# print(data[:, ['close']])
# print(data.head())
# print(data.index)
# print(data.columns)
# print(data['tradeDate'])
# print(data.loc[:, ['open','close']])
# print(data.iloc[3])
# print(data[data.close > 60])
# print(data.mean())
data = pd.DataFrame(data)
print(data)
upDown = np.where(data.close > data.open, 1, 0)
data['upDown'] = upDown
print(data)
data = data.drop(['tradeDate'], axis=1)
print(data)

# 训练模型
# 切分训练数据和测试数据
data = tf.constant(data, dtype=tf.float32)
# print(data)
x_train = data[:,0:-1]
y_train = data[:,-1]
print(x_train)
print(y_train)

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(1, input_shape=(6,)))
model.add(tf.keras.layers.Dense(1, activation='relu'))
model.add(tf.keras.layers.Dense(1,activation='softmax'))

# print(model.summary())
model.compile(optimizer='adam',
              loss=tf.keras.losses.binary_crossentropy
              )
history = model.fit(x_train, y_train, epochs=1000)
print(history)
