import matplotlib.pyplot as plt
import pandas as pd
import os
import time
import numpy as np
import tensorflow as tf
# import tensorflow as tf  # This code has been tested with TensorFlow 1.6
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.layers import LSTM
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


def getData(shareCode, featureNum=10, dayNum=500):
  # 选出与shareCode相关性大的股票relatedCodes
  # path = '../hs300/'+shareCode+'_080101_180630.csv'

  coffpd = pd.DataFrame.from_csv('../hs300cov_' + str(dayNum) + '.csv')
  shareCodes = coffpd.columns.tolist()

  coffline = coffpd[shareCode]

  relatedCodes = []
  for sc in shareCodes:
    if coffline[int(sc)] >= 0.8 and coffline[int(sc)] != 1:
      relatedCodes.append(sc)

  print('相关股票个数:', len(relatedCodes))
  # 把relatedCodes中的股票作为测试集
  data = []
  label = []

  for code in relatedCodes:
    df = pd.DataFrame.from_csv('../hs300/'+code+'_080101_180630.csv')
    close_prices = df.loc[-dayNum:, 'close'].as_matrix()
    close_prices = close_prices[-500:]
    sample_num = len(close_prices) - featureNum
    rdarray = np.random.permutation(range(sample_num))
    
    for index in rdarray:
      first = close_prices[index]
      data.append([[(x/first-1)] for x in close_prices[index: index + featureNum]])
      if close_prices[index + featureNum] >= close_prices[index + featureNum-1]:
        label.append(1)
      else:
        label.append(0)
  print('训练集大小:', len(data))

  validateData = []
  validateLabel = []
  df = pd.DataFrame.from_csv('../hs300/'+shareCode+'_080101_180630.csv')
  close_prices = df.loc[-dayNum:, 'close'].as_matrix()
  close_prices = close_prices[-500:]
  
  for index in range(len(close_prices)-featureNum):
    first = close_prices[index]
    validateData.append([[(x/first-1)] for x in close_prices[index: index + featureNum]])
    if close_prices[index + featureNum] >= close_prices[index + featureNum-1]:
      validateLabel.append(1)
    else:
      validateLabel.append(0)
  print('验证集大小:', len(validateData))

  return data, label, validateData, validateLabel, close_prices


sess = tf.Session()

featureNum = 10
# LSTM 输入是三维，[样本数, timestep, 单个时间片的维度]
data, label, validateData, validateLabel, close_prices = getData(
    '000002', featureNum, 500)

# Data = [[[(i+j)/100] for i in range(5)] for j in range(100)]
# target = [(i+10)/100 for i in range(100)]

# data = np.array(Data, dtype=float)
# target = np.array(target, dtype=float)

# x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=4)
x_train = np.array(data, dtype=float)
x_test = np.array(validateData, dtype=float)
y_train = np.array(label, dtype=float)
y_test = np.array(validateLabel, dtype=float)

model = Sequential()
model.add(LSTM((50), batch_input_shape=(None, 10, 1), return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM((100), return_sequences=False))
model.add(Dropout(0.2))

model.add(Dense(1))
model.add(Activation('linear'))
model.summary()

writer = tf.summary.FileWriter("./log", sess.graph)

start = time.time()
model.compile(loss='mae', optimizer='rmsprop')
print('compilation time : ', time.time()-start)

history = model.fit(x_train, y_train, batch_size=32, epochs=10, validation_data=(x_test, y_test))

results = model.predict(x_test)
plt.figure()
plt.subplot(211)
plt.scatter(range(len(results)), results, c='r')
plt.scatter(range(len(y_test)), y_test, c='g')
plt.subplot(212)
plt.plot(history.history['loss'])
plt.show()
