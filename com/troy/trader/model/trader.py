# coding:utf-8

import tensorflow as tf
from tensorflow import keras
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import glob

# 定义数据
print(tf.__version__)
print(tf.test.gpu_device_name)

# data_path = glob.glob('G:/data/stock/20210418/*.txt')
# print(data_path[:5])
# data = np.loadtxt('G:/data/stock/20210418/SH#600000.txt')
data = np.genfromtxt('../data/SH#600000.txt', dtype=None)
print(data)