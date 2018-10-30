# using diff to predict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def getData(path):
  df = pd.DataFrame.from_csv(path)
  close_prices = df.loc[:, 'close'].as_matrix()
  return close_prices

def predict(trainData, depth):
  diffMatrix = []
  diffMatrix.append(trainData)

  for i in range(depth):
    diff = []
    for j in range(i+1):
      diff.append(0)
    
    for j in range(i+1, len(trainData)):
      diff.append(diffMatrix[i][j]-diffMatrix[i][j-1])
    
    diffMatrix.append(diff)

  diffMatrix = np.array(diffMatrix)
  predictions = [0]
  predictions.extend(diffMatrix.sum(0))
  predictions.pop(-1)

  return predictions

def calTrendAccuracy(trainData, predictions):
  trendAccuracy = []
  trendAccuracy.append(1)
  for i in range(len(trainData)-1):
    if (trainData[i+1]-trainData[i]) * (predictions[i+1]-trainData[i]) >= 0:
      trendAccuracy.append(1)
    else:
      trendAccuracy.append(0)
  
  print('Trend Accuracy: ', np.sum(trendAccuracy)/len(trendAccuracy))
  return trendAccuracy


def showResult(trueData, predictions):
  plt.figure(figsize=(18, 9))
  plt.plot(range(len(trueData)), trueData, color='b', label='True')
  plt.plot(range(len(predictions)), predictions,
           color='orange', label='Prediction')
  #plt.xticks(range(0,df.shape[0],50),df['Date'].loc[::50],rotation=45)
  plt.xlabel('Date')
  plt.ylabel('Mid Price')
  plt.legend(fontsize=18)
  plt.show()


def saveResult(trueData, predicitons, trendAccuracy):
  data = np.vstack((trueData, predicitons, trendAccuracy)).T
  df = pd.DataFrame(data, columns=[
                    'TRUE', 'PREDICTION', 'TREND ACCURACY'])
  df.to_csv('./diffResult.csv')

def testAll300():
  fileNames = os.listdir('e:/myRepertories/QuantInvest/hs300')

def main():
  path = '../hs300/000538_080101_180630.csv'
  data = getData(path)
  predictions = predict(data, 10)
  trendAccuracy = calTrendAccuracy(data, predictions)
  showResult(data, predictions)
  saveResult(data, predictions, trendAccuracy)

if __name__ == '__main__':
  main()
  
