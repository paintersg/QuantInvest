from PIL import Image
from PIL import ImageDraw
import matplotlib.pyplot as plt
import random
import numpy as np

# input: 20 days' prices
# input format: np.array
# input detail: [[hp1, lp1, op1, cp1], [hp2, lp2, op2, cp2], ...]
def createOneImage224(prices, index):
  img = Image.new('RGB', (224, 224))
  draw = ImageDraw.Draw(img)

  highestPrice = prices.max()
  lowestPrice = prices.min()

  pricePerPixel = (highestPrice - lowestPrice) / 200

  for dayNum in range(20):
    if prices[dayNum][3] >= prices[dayNum][2]:
      color = (255, 0, 0)
    else:
      color = (0, 255, 0)

    highestPoint = 12 + int((highestPrice - prices[dayNum][0])/pricePerPixel)
    lowestPoint = 12 + int((highestPrice - prices[dayNum][1])/pricePerPixel)
    leftMargin = 12+4+10*dayNum

    draw.rectangle(((leftMargin, highestPoint), (leftMargin, lowestPoint)), color)

    openPoint = 12 + int((highestPrice - prices[dayNum][2])/pricePerPixel)
    closePoint = 12 + int((highestPrice - prices[dayNum][3])/pricePerPixel)
    leftMargin = 12+2+10*dayNum

    draw.rectangle(((leftMargin, openPoint),
                    (leftMargin+4, closePoint)), color)
  
  # img.show()
  # print(img.size)
  img.save('./test_images/'+str(index)+'.jpg')


def createOneImage64(prices, index):
  img = Image.new('RGB', (64, 64))
  draw = ImageDraw.Draw(img)

  highestPrice = prices.max()
  lowestPrice = prices.min()

  pricePerPixel = (highestPrice - lowestPrice) / 50

  for dayNum in range(10):
    if prices[dayNum][3] >= prices[dayNum][2]:
      color = (255, 0, 0)
    else:
      color = (0, 255, 0)

    highestPoint = 7 + int((highestPrice - prices[dayNum][0])/pricePerPixel)
    lowestPoint = 7 + int((highestPrice - prices[dayNum][1])/pricePerPixel)
    leftMargin = 7+2+5*dayNum

    draw.rectangle(((leftMargin, highestPoint),
                    (leftMargin+0, lowestPoint)), color)

    openPoint = 7 + int((highestPrice - prices[dayNum][2])/pricePerPixel)
    closePoint = 7 + int((highestPrice - prices[dayNum][3])/pricePerPixel)
    leftMargin = 7+1+5*dayNum

    draw.rectangle(((leftMargin, openPoint),
                    (leftMargin+2, closePoint)), color)

  # img.show()
  # print(img.size)
  img.save('./train_images/'+str(index)+'.jpg')
  
def randomPrice():
  prices = []
  for i in range(20):
    price = []
    price.append(random.randint(61, 100))
    price.append(random.randint(0, 30))
    price.append(random.randint(31, 60))
    price.append(random.randint(31, 60))
    prices.append(price)
  
  prices = np.array(prices)
  return prices

def main():
  for i in range(6000):
    prices = randomPrice()
    createOneImage64(prices, i)

if __name__ == '__main__':
  main()
