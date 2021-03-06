import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

''''
数字图片识别
'''
# 测试
version = tf.__version__
print(version)
print(tf.test.is_gpu_available())


# 第一步：准备数据
fashion_mnist = keras.datasets.fashion_mnist
print(fashion_mnist)
(train_images, train_labels),(test_images, test_labels) = fashion_mnist.load_data()
print(train_images.shape)
print(train_labels.shape)
print(test_images.shape)
# 扩张数据 图片分为 长*款*channal  彩色rgb=3 黑白=1
train_images = np.expand_dims(train_images, -1)
print(train_images.shape)
test_images = np.expand_dims(test_images, -1)

# 第二步：建立模型
model = tf.keras.Sequential()
# 参数 32：卷积核个数
model.add(tf.keras.layers.Conv2D(64, (3,3), input_shape=train_images.shape[1:], activation='relu',padding='same'))
model.add(tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding='same'))
model.add(tf.keras.layers.Dropout(0.5))
print(model.output_shape)
model.add(tf.keras.layers.MaxPool2D())
print(model.output_shape)
model.add(tf.keras.layers.Conv2D(128, (3,3), activation='relu'))
model.add(tf.keras.layers.Conv2D(128, (3,3), activation='relu'))
print(model.output_shape)
model.add(tf.keras.layers.GlobalAvgPool2D())
model.add(tf.keras.layers.Dropout(0.5))
print(model.output_shape)
model.add(tf.keras.layers.Dense(10, activation='softmax'))
print(model.output_shape)
model.summary()

# 第三步：编译模型
model.compile(optimizer='adam',
               loss=tf.keras.losses.sparse_categorical_crossentropy,
              metrics=['acc'])

# 第四步：训练
history = model.fit(train_images,train_labels, epochs=10,validation_data=(test_images, test_labels))
print(history.history.keys())

# 绘制正确率折线图
plt.plot(history.epoch, history.history.get('acc'), label='acc')
plt.plot(history.epoch, history.history.get('val_acc'), label='val_acc')
plt.show()

# 绘制损失函数的折线图
plt.plot(history.epoch, history.history.get('loss'), label='loss')
plt.plot(history.epoch, history.history.get('val_loss'), label='val_loss')
plt.show()
