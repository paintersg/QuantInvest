import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np

def compareKLine(df1, df2, name1, name2):
  plt.figure(figsize=(18, 9))
  plt.plot(range(df1.shape[0]), (df1['low']+df1['high'])/2.0, label=name1)
  plt.plot(range(df1.shape[0]-df2.shape[0],
                 df1.shape[0]), (df2['low']+df2['high'])/2.0, label=name2)
  plt.legend()
  plt.xticks(range(0, df1.shape[0], 100),
             df1['date'].loc[::100], rotation=45, fontsize=5)
  plt.xlabel('Date', fontsize=18)
  plt.ylabel('Mid Price', fontsize=18)
  
  plt.show()

def main():
  share1 = '002304'
  share2 = '600436'
  df1 = pd.DataFrame.from_csv(
      'e:/myRepertories/QuantInvest/hs300/'+share1+'_080101_180630.csv')
  df2 = pd.DataFrame.from_csv(
      'e:/myRepertories/QuantInvest/hs300/'+share2+'_080101_180630.csv')
  if df1.shape[0] > df2.shape[0]:
    compareKLine(df1, df2, share1, share2)
  else:
    compareKLine(df2, df1, share2, share1)

if __name__ == '__main__':
  main()
