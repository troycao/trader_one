import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# 测试
version = tf.__version__
print(version)
print(tf.test.is_gpu_available())