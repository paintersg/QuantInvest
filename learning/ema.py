# Exponential Moving Average
# used to predict share price
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def getData(path):
  df = pd.DataFrame.from_csv(path)
  low_prices = df.loc[:, 'low'].as_matrix()

  length = len(low_prices)
  splitPoint = round(length/6)*5

  train_data = low_prices[:2500]
  test_data = low_prices[2500:]

  return low_prices, train_data, test_data


def predict(train_data, decay, diff_decay):
  N = train_data.size

  run_avg_predictions = []
  diff_avg_predictions = []
  final_predictions = []

  mse_errors = []
  trendAccuracy = []

  running_mean = 0.0
  diff_running_mean = 0.0

  run_avg_predictions.append(running_mean)
  diff_avg_predictions.append(diff_running_mean)
  final_predictions.append(running_mean+diff_running_mean)
  mse_errors.append(0)
  trendAccuracy.append(1)

  decay = decay
  diff_decay = diff_decay

  for pred_idx in range(1, N):
    diff_running_mean = diff_running_mean * \
        diff_decay + (1.0-diff_decay)*(train_data[pred_idx-1]-running_mean)
    running_mean = running_mean*decay + (1.0-decay)*train_data[pred_idx-1]
    

    run_avg_predictions.append(running_mean)
    diff_avg_predictions.append(diff_running_mean)
    final_predictions.append(running_mean+diff_running_mean)

    mse_errors.append((final_predictions[-1]-train_data[pred_idx])**2)

    trendAcc = (final_predictions[-1]-train_data[pred_idx-1]) * \
        (train_data[pred_idx]-train_data[pred_idx-1])

    if trendAcc >= 0:
      trendAccuracy.append(1)
    else:
      trendAccuracy.append(0)

  # print(decay, diff_decay)
  print('the accuracy of the trend of prediction: ', 
        (np.sum(trendAccuracy)/len(trendAccuracy)))
  # print('')

  return run_avg_predictions, diff_avg_predictions, final_predictions, mse_errors, trendAccuracy

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


def saveResult(trueData, run_avg_predictions, diff_avg_predictions, final_predictions, mse_errors, trendAccuracy):
  print(len(run_avg_predictions))
  data = np.vstack((trueData, final_predictions, run_avg_predictions, diff_avg_predictions, mse_errors, trendAccuracy)).T
  df = pd.DataFrame(data, columns=['TRUE', 'PREDICTION','RUNNING AVG', 'DIFF RUNNING AVG', 'MSE', 'TREND ACCURACY'])
  df.to_csv('./emaResult.csv')

def findBestParameters():
  bestacc = 0
  bestdacay = 0
  bestdiff_decay = 0
  for decay in range(1, 10):
    for diff_decay in range(1, 10):
      run_avg_predictions, diff_avg_predictions, final_predictions, mse_errors, trendAccuracy = predict(
          low_prices, decay/10.0, diff_decay/10.0)
      temp = (np.sum(trendAccuracy)/len(trendAccuracy))
      if temp > bestacc:
        bestacc = temp
        bestdacay = decay/10.0
        bestdiff_decay = diff_decay/10.0

  print('best accuracy: ', bestacc)
  print('best decay: ', bestdacay)
  print('best diff decay: ', bestdiff_decay)
  return bestdacay, bestdiff_decay, bestacc

def main():
  path = '../hs300/000060_080101_180630.csv'
  low_prices, train_data, test_data = getData(path)

  # bestdacay, bestdiff_decay, bestacc = findBestParameters()

  # predict with best parameters
  run_avg_predictions, diff_avg_predictions, final_predictions, mse_errors, trendAccuracy = predict(
      low_prices, 0.0, 0.0)
  showResult(low_prices, final_predictions)
  saveResult(low_prices, run_avg_predictions, diff_avg_predictions,
             final_predictions, mse_errors, trendAccuracy)

if __name__ == '__main__':
  main()
