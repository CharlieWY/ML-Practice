import logRegres
from numpy import *


dataArr, labelMat = logRegres.loadDataSet()


weights = logRegres.gradAscent(dataArr, labelMat)
logRegres.plotBestFit(weights.getA())  # getA()将一个numpy矩阵转化为numpy数组，与mat方法功能相反
print(weights)


weights=logRegres.stocGradAscent0(array(dataArr),labelMat)
logRegres.plotBestFit(weights)
print(weights)


weights=logRegres.stocGradAscent1(array(dataArr),labelMat)
logRegres.plotBestFit(weights)
print(weights)