import os
import numpy as np
import pandas as pd

# path是存放股票数据的文件夹
# minLength是单只股票数据最小长度，有些股票的数据可能不满10年
def getData(path, minLength = 1000):
  fileNames = os.listdir(path)

  data = []
  shareList = []
  

  for fileName in fileNames:
    df = pd.DataFrame.from_csv(path + '/' + fileName)

    # !!!!!!这里的日期需要手动修改
    # 有些股票在2018-06-29之前就没了，需要过滤掉这些股票
    if df.iloc[-1].date == '2018-06-29':
      closePriceArray = df.close.tolist()
    
      if len(closePriceArray) >= minLength:
        data.append(closePriceArray[-minLength:])
        shareList.append(fileName[:6])
  
  # data的每一行是一只股票的收盘价序列，长度为minLength
  # shareList是没有被过滤掉的股票列表
  return np.array(data), shareList


def main():
  minLength = 2000
  data, shareList = getData('e:/myRepertories/QuantInvest/hs300', minLength)
  
  covMatrix = np.cov(data)

  covDF = pd.DataFrame(data=covMatrix, index=shareList, columns=shareList)
  print(covDF)
  covDF.to_csv('../hs300cov_'+str(minLength)+'.csv')

if __name__ == '__main__':
  main()
