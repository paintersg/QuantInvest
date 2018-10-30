import pandas as pd
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.externals import joblib
import warnings

warnings.filterwarnings('ignore')

def getData(shareCode, featureNum = 10, dayNum = 500):
  # 选出与shareCode相关性大的股票relatedCodes
  # path = '../hs300/'+shareCode+'_080101_180630.csv'

  coffpd = pd.DataFrame.from_csv('../hs300cov_'+ str(dayNum) +'.csv')
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
      data.append(close_prices[index: index + featureNum])
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
    validateData.append(close_prices[index: index + featureNum])
    if close_prices[index + featureNum] >= close_prices[index + featureNum-1]:
      validateLabel.append(1)
    else:
      validateLabel.append(0)
  print('验证集大小:', len(validateData))
  
  return data, label, validateData, validateLabel, close_prices

def getModel():
  return MLPClassifier(
      solver='adam', alpha=1e-4, learning_rate='adaptive', hidden_layer_sizes=(8, 10, 6), random_state=1)

def validateOneModel(prediction, label):
  trueCounts = 0

  for i in range(len(prediction)):
    if prediction[i] == label[i]:
      trueCounts += 1

  print("accuracy: ", trueCounts/len(prediction))

def validateMultiModel(prediction, label):
  trueCounts = 0
  for i in range(len(prediction)):
    if round(prediction[i]) == label[i]:
      trueCounts += 1
  
  print("accuracy: ", trueCounts/len(prediction))

def useOneModel(data, label, validateData, validateLabel):
  model = getModel()
  model.fit(data[:3000], label[:3000])
  prediction = model.predict(validateData)
  validateOneModel(prediction, validateLabel)
  joblib.dump(model, "model.m")


def useMultiModel(data, label, validateData, validateLabel, close_prices, featureNum, modelNum=10):
  models = []
  l = int(len(data)/modelNum)
  for i in range(modelNum):
    tempModel = getModel()
    tempModel.fit(data[i*l:(i+1)*l], label[i*l:(i+1)*l])
    models.append(tempModel)

  prediction = []
  for model in models:
    prediction.append(model.predict(validateData))
  
  prediction = np.array(prediction)
  prediction = prediction.sum(0)/modelNum

  validateMultiModel(prediction, label)
  evaluate(prediction, close_prices, featureNum)

def evaluate(prediction, close_prices, featureNum):
  capital = 100000
  oldCapital = capital

  profitRateArray = []
  isIn = 0
  for i in range(len(prediction)):
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
    if prediction[i] > 0.5 and isIn == 0:
      isIn = 1
      inPrice = close_prices[i+featureNum-1]
    elif prediction[i] <= 0.5 and isIn == 1:
      isIn = 0
      oldCapital = oldCapital/inPrice * close_prices[i+featureNum-1]

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

  plt.figure()
  plt.subplot(211)
  plt.plot(range(len(profitRateArray)), profitRateArray)
  plt.scatter(minI, profitRateArray[minI], c='r', marker='.')
  plt.scatter(minJ, profitRateArray[minJ], c='r', marker='.')
  plt.subplot(212)
  plt.plot(range(len(close_prices)), close_prices)
  plt.show()

def main():
  # 天数可以取500，1000，2000
  featureNum = 10
  data, label, validateData, validateLabel, close_prices = getData(
      '000002', featureNum, 500)
  useMultiModel(data, label, validateData, validateLabel,
                close_prices, featureNum, 10)

if __name__ == '__main__':
  main()
