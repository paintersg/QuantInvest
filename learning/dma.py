# double moving average
import pandas as pd
import numpy as np

# index与日期对应关系
# 68   <=>  2013/04/19
# 237  <=>  2013/12/31
# 1333 <=>  2018/06/29

class dma(object):
  def __init__(self, startDate=None, endDate=None):
    super(dma, self).__init__()
    self.startDate = startDate
    self.endDate = endDate
    # 数据表
    self.dataframe = None
    # startDate的index
    self.startIndex = None
    # 本金
    # self.capital = None
    # 短期均线中，短期的可选长度
    self.shortTermLengths = [1, 2, 3, 5, 8, 13, 21, 34]
    # 长期均线中，长期的可选长度
    self.longTermLengths = [2, 3, 5, 8, 13, 21, 34, 55, 89, 144]
    # 所有短期，长期长度组合的年化利率表
    self.profitTable = pd.DataFrame(
        index=self.longTermLengths, columns=self.shortTermLengths)
  
  def readData(self, path):
    self.dataframe = pd.DataFrame.from_csv(path)
    self.startIndex = self.dataframe[(
        self.dataframe.date == self.startDate)].index.tolist()[0]
  
  def calProfitTable(self):
    for l in self.longTermLengths:
      for s in self.shortTermLengths:
        if s >= l:
          break
        else:
          annualProfitRate = self.calAnnualProfit(l, s)
          self.profitTable.set_value(l, s, annualProfitRate)
  
  def calAnnualProfit(self, l, s):
    # shortTermAvg和longTermAvg是从2013-12-31到2018-06-30的数组
    shortTermAvg = self.calAvgArray(s)
    longTermAvg = self.calAvgArray(l)

    profitRate = 0
    tempInprice = -1
    
    # 根据双均线进行交易，计算总收益率profitRate
    # 这里假设卖出股票后获得的盈利不再用于购买股票，本金始终不变
    for day in range(len(shortTermAvg)):
      if day == 0:
        continue
      else:
        if shortTermAvg[day] > longTermAvg[day] and shortTermAvg[day-1] <= longTermAvg[day-1] and tempInprice == -1:
          tempInprice = self.dataframe.at[day+self.startIndex, 'close']
        elif shortTermAvg[day] <= longTermAvg[day] and shortTermAvg[day-1] > longTermAvg[day-1] and tempInprice != -1:
          profitRate = profitRate + (self.dataframe.at[day+self.startIndex, 'close'] - tempInprice)/tempInprice
          tempInprice = -1
        else:
          continue
    
    # 如果到endDate还没有卖出，不需要卖出全部股票
    # 因为此时卖出导致的盈亏并非双均线的结果
    
    # 4.5年
    annualProfitRate = profitRate/4.5
    return annualProfitRate

  def calAvgArray(self, days):
    length = self.dataframe.index.tolist()[-1]-self.startIndex+1

    avgArray = [None]*length
    # print('!!!!!!!!',self.startIndex)
    priceSum = 0
    for i in range(days):
      priceSum = priceSum + self.dataframe.at[self.startIndex-i-2, 'close']

    for day in range(length):
      priceSum = priceSum - self.dataframe.at[self.startIndex+day-days-1, 'close'] + self.dataframe.at[self.startIndex+day-1, 'close']
      avgArray[day] = priceSum/days
    
    return avgArray


if __name__ == '__main__':
  # 2013-04-19到2013-12-30的是备用数据，用于计算双均线
  model = dma(startDate='2013/12/31', endDate='2018/6/29')
  model.readData('./data/399300_130419_180630.csv')
  model.calProfitTable()
  print(model.profitTable)
  print('finish')
