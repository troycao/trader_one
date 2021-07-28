import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import os
import tensorflow as tf
from tensorflow import keras

# read data-sensor.csv
dataframe = pd.read_csv('SH#600000.csv')
pd_value = dataframe.values

# ========= split dataset ===================
train_size = int(len(pd_value) * 0.8)
trainlist = pd_value[:train_size]
testlist = pd_value[train_size:]

look_back = 4
features = 28
step_out = 1

# ========= numpy train ===========
def create_dataset(dataset, look_back):
#这里的look_back与timestep相同
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back)]
        dataX.append(a)
        dataY.append(dataset[i + look_back])
    return np.array(dataX),np.array(dataY)
#训练数据太少 look_back并不能过大
trainX,trainY  = create_dataset(trainlist,look_back)
testX,testY = create_dataset(testlist,look_back)

# ========== set dataset ======================
# trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], features))
# testX = np.reshape(testX, (testX.shape[0], testX.shape[1] , features))

# create and fit the LSTM network
model = tf.keras.Sequential()
model.add(tf.keras.layers.LSTM(64, activation='relu', return_sequences=True, input_shape=(look_back, features)))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.LSTM(32, activation='relu'))
model.add(tf.keras.layers.Dense(features))
#model.compile(optimizer='adam', loss='mse')
model.compile(metrics=['accuracy'], loss='mean_squared_error', optimizer='adam')

model.summary()

history = model.fit(trainX, trainY, validation_data=(testX, testY),epochs=15, verbose=1).history
model.save("lstm-model.h5")

plt.plot(history['loss'], linewidth=2, label='Train')
plt.plot(history['val_loss'], linewidth=2, label='Test')
plt.legend(loc='upper right')
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
#plt.ylim(ymin=0.70,ymax=1)
plt.show()

trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

plt.plot(trainY[:100,1])
plt.plot(trainPredict[:100,1])
plt.show()

plt.plot(testY[:100,1])
plt.plot(testPredict[:100,1])
plt.plot()

# set predict_data
predict_begin = 1
predict_num = 100
predict_result = np.zeros((predict_num+look_back,features),dtype=float)
for i in range(look_back):
    predict_result[i] = testX[-predict_begin:][0,i]

# predict
for i in range(predict_num):
    begin_data = np.reshape(predict_result[i:i+look_back,], (predict_begin, look_back, features))
    predict_data = model.predict(begin_data)
    predict_result[look_back+i] = predict_data
    buff = predict_result[i+1:i+look_back]
    predict_call_back = np.append(buff,predict_data,axis=0)

# show plot
plt.plot(predict_result[-predict_num:,5])
plt.plot()
