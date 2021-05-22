# coding:utf-8

import tensorflow as tf
from tensorflow import keras
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob

# 1.准备数据
data = pd.read_csv('../data/SH#600000.csv')
print(data.head())
x_data = pd.DataFrame(data)
x_data = x_data.drop(['tradeDate'], axis=1)

def get_y_data(x_data):
    y_data = [0,0,0]
    for index in range(len(x_data['close'])):
        if index >= 3:
            three_before_close = x_data['close'][index - 3]
            current_close = x_data['close'][index]
            if current_close > three_before_close:
                y_data.append(1)
            else:
                y_data.append(0)
    return y_data

y_data = get_y_data(x_data)
print(y_data)

x_data = tf.constant(x_data, dtype=tf.float32)
y_data = tf.constant(y_data, dtype=tf.float32)
x_data = x_data[:,1:]
print(x_data)
print(y_data)

# 2.建立模型
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(1, input_shape=(5,)))
model.add(tf.keras.layers.Dense(1, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='softmax'))

# 3.编译模型
model.compile(optimizer='adam',
              loss=tf.keras.losses.binary_crossentropy,
              metrics=['acc'])

# 4.训练模型
history = model.fit(x_data,
                    y_data,
                    epochs=50,
                    batch_size=32,
                    shuffle=True)

# 5.绘图
plt.plot(history.epoch, history.history.get('acc'))
plt.xlabel('acc')
plt.show()

plt.plot(history.epoch, history.history.get('loss'))
plt.xlabel('loss')
plt.show()
