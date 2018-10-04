import tushare as ts

def get300shareData(startDate='2008-01-01', endDate='2018-06-30'):
  df = ts.get_hs300s()
  codeList = df.code.tolist()
  for code in codeList:
    df = ts.get_k_data(code, start=startDate, end=endDate)
    df.to_csv('../hs300/' + code + '_' +
              startDate[2:].replace('-','') + '_' +
              endDate[2:].replace('-', '') + '.csv')

def main():
  # get300shareData()

if __name__ == '__main__':
  main()
