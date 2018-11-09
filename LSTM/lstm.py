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
import warnings

warnings.filterwarnings('ignore')

################
# prepare data #
################
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
    maxp = max(close_prices)

    for index in rdarray:
      first = close_prices[index]
      data.append([[(x/first-1)]
                   for x in close_prices[index: index + featureNum]])
      label.append(close_prices[index + featureNum]/first-1)
      # if close_prices[index + featureNum] >= close_prices[index + featureNum-1]:
      #   label.append(1)
      # else:
      #   label.append(0)
  print('训练集大小:', len(data))

  validateData = []
  validateLabel = []
  # 用于把label以及预测结果复原为股价
  validateFirsts = []
  df = pd.DataFrame.from_csv('../hs300/'+shareCode+'_080101_180630.csv')
  close_prices = df.loc[-dayNum:, 'close'].as_matrix()
  close_prices = close_prices[-500:]
  maxp = max(close_prices)
  # validateLabel里面是处理后的price,我需要用raw_label记录下原始的price
  # raw_label = []

  for index in range(len(close_prices)-featureNum):
    first = close_prices[index]
    validateData.append([[(x/first-1)]
                         for x in close_prices[index: index + featureNum]])
    validateLabel.append(close_prices[index + featureNum]/first-1)
    # raw_label.append(close_prices[index + featureNum])
    # if close_prices[index + featureNum] >= close_prices[index + featureNum-1]:
    #   validateLabel.append(1)
    # else:
    #   validateLabel.append(0)
    # validateFirsts.append(first)
  print('验证集大小:', len(validateData))
  # print(close_prices[0:20])
  return data, label, validateData, validateLabel, close_prices, validateFirsts

sess = tf.Session()

featureNum = 10
# LSTM 输入是三维，[样本数, timestep, 单个时间片的维度]
data, label, validateData, validateLabel, close_prices, validateFirsts = getData(
    '000625', featureNum, 500)

# Data = [[[(i+j)/100] for i in range(5)] for j in range(100)]
# target = [(i+10)/100 for i in range(100)]

# data = np.array(Data, dtype=float)
# target = np.array(target, dtype=float)

# x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=4)
x_train = np.array(data, dtype=float)
x_test = np.array(validateData, dtype=float)
y_train = np.array(label, dtype=float)
y_test = np.array(validateLabel, dtype=float)
# validateFirsts = np.array(validateFirsts, dtype=float)
###############
# build model #
###############
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
model.compile(loss='mae', optimizer='adam')
print('compilation time : ', time.time()-start)

#########
# train #
#########
history = model.fit(x_train, y_train, batch_size=32, epochs=3, validation_data=(x_test, y_test))

###########
# predict #
###########
results = model.predict(x_test)
results = results.reshape(-1)

##################
# recover prices #
##################
# raw_prices = (y_test+1)*validateFirsts # 应该与close_prices[-490:0]相同
# predicted_prices = (results+1)*validateFirsts
# predicted_prices = predicted_prices + raw_prices.mean() - predicted_prices.mean()

######################
# calculate accuracy #
######################
accracyArray = []
accracyArray.append(0)
for index in range(1,len(results)):
  if ((y_test[index] - y_test[index-1]) * (results[index] - y_test[index-1])) > 0:
    accracyArray.append(1)
  else:
    accracyArray.append(0)

accracyArray = np.array(accracyArray)
accuracy = accracyArray.sum()/len(accracyArray)
print('accuracy: ', accuracy)

#################################
# calculate profit and drawback #
#################################
def evaluate(prediction, label, close_prices, featureNum):
  capital = 100000
  oldCapital = capital

  profitRateArray = []
  trade_history = []
  # profitRateArray.append(0)
  # trade_history.append('keep out')

  isIn = 0
  for i in range(1,len(prediction)):
    # i+featureNum-1代表今天
    # 判断盈亏
    if isIn == 1:
      currentCapital = oldCapital/inPrice * close_prices[i+featureNum-1]
      profitRate = (currentCapital-capital)/capital
      profitRateArray.append(profitRate)
    else:
      if len(profitRateArray) == 0:
        profitRateArray.append(0)
      else:
        profitRateArray.append(profitRateArray[-1])

    # 买卖
    if prediction[i] > label[i-1] and isIn == 0:
      isIn = 1
      trade_history.append('in')
      inPrice = close_prices[i+featureNum-1]
    elif prediction[i] < label[i-1] and isIn == 1:
      isIn = 0
      trade_history.append('out')
      oldCapital = oldCapital/inPrice * close_prices[i+featureNum-1]
    else:
      if isIn == 0:
        trade_history.append('keep out')
      elif isIn == 1:
        trade_history.append('keep in')

  

  max_draw_back = 0
  minI = 0
  minJ = 0

  for i in range(len(profitRateArray)):
    for j in range(i+1, len(profitRateArray)):
      temp_draw_back = (profitRateArray[j] -
                        profitRateArray[i])/(profitRateArray[i]+1)
      if temp_draw_back < max_draw_back:
        max_draw_back = temp_draw_back
        minI = i
        minJ = j

  profitRate = (oldCapital-capital)/capital
  annualProfitRate = profitRate/1.9
  print('年化收益: ', annualProfitRate)
  print('最大回撤: ', max_draw_back)

  # 在label的最后一天，没有对后一天的预测，所以没有trade
  trade_history.append('none')
  profitRateArray.append(profitRateArray[-1])


  # plt.figure()
  # plt.subplot(211)
  # plt.plot(range(len(profitRateArray)), profitRateArray)
  # plt.scatter(minI, profitRateArray[minI], c='r', marker='.')
  # plt.scatter(minJ, profitRateArray[minJ], c='r', marker='.')
  # plt.subplot(212)
  # plt.plot(range(len(close_prices)), close_prices)
  # plt.show()

  return profitRateArray, minI, minJ, trade_history


profitRateArray, minI, minJ, trade_history = evaluate(
    results, y_test, close_prices, featureNum)

################
# save results #
################
df = pd.DataFrame(data=np.stack((y_test, results, accracyArray, trade_history, close_prices[-490:], profitRateArray)).T,
                  columns=['label', 'prediction', 'accuracy', 'trade_history', 'close_prices', 'profitRateArray'])
df.to_csv('prediciton.csv')


################
# show results #
################
plt.figure()
plt.subplot(411)
plt.plot(range(len(results)), results, c='r', label='prediction')
plt.plot(range(len(y_test)), y_test, c='g', label='label')
plt.legend()
plt.subplot(412)
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.legend()
plt.subplot(413)
plt.plot(range(len(profitRateArray)), profitRateArray, label='profit')
plt.scatter(minI, profitRateArray[minI], c='r', marker='.')
plt.scatter(minJ, profitRateArray[minJ], c='r', marker='.')
plt.legend()
plt.subplot(414)
plt.plot(range(len(close_prices)), close_prices, label='price')
plt.legend()
plt.show()
