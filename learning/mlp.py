import pandas as pd
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPClassifier


class mlpClassifier(object):
  def __init__(self, featureNum, dataPath):
    super(mlpClassifier, self).__init__()
    self.classifier = MLPClassifier(
        solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(10, 10, 6), random_state=1)
    self.data, self.label, self.validateData, self.validateLabel = self.getData(dataPath, featureNum)

  def getData(self, dataPath, featureNum):

    # validate数量需要手动改
    validateNum = 200

    df = pd.DataFrame.from_csv(dataPath)
    close_prices = df.loc[:, 'close'].as_matrix()

    sample_num = len(close_prices) - featureNum - validateNum

    rdarray = np.random.permutation(range(sample_num))

    data = []
    label = []

    for index in rdarray:
      data.append(close_prices[index : index + featureNum])
      if close_prices[index + featureNum] >= close_prices[index + featureNum-1]:
        label.append(1)
      else:
        label.append(0)
    
    validateData = []
    validateLabel = []

    for index in range(len(close_prices)-validateNum, len(close_prices)-featureNum):
      validateData.append(close_prices[index: index + featureNum])
      if close_prices[index + featureNum] >= close_prices[index + featureNum-1]:
        validateLabel.append(1)
      else:
        validateLabel.append(0)

    return np.array(data), np.array(label), np.array(validateData), np.array(validateLabel)

  def train(self):
    self.classifier.fit(self.data, self.label)
  
  def validate(self):
    predictions = self.classifier.predict(self.validateData)

    trueCounts = 0

    for i in range(len(predictions)):
      if predictions[i] == self.validateLabel[i]:
        trueCounts += 1

    print("accuracy: ", trueCounts/len(predictions))




def main():
  path = '../hs300/000538_080101_180630.csv'
  classifier = mlpClassifier(featureNum=10, dataPath=path)
  classifier.train()
  classifier.validate()

if __name__ == '__main__':
  main()
