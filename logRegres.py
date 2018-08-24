from numpy import *


def loadDataSet():
    dataMat = []
    labelMat = []
    fr = open('testSet.txt')  # 定义对象fr指向该文件
    for line in fr.readlines():  # 按行读取并操作文件
        lineArr = line.strip().split()  # split参数为空,默认以空字符（空格等）为分隔符进行分割
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        # 插入由θ0，θ1，θ2组成的三元组，因为θ0*x0是回归函数中的常数项，所以θ0=1.0
        labelMat.append(int(lineArr[2]))  # 向labelMat存入标签
    return dataMat, labelMat


def sigmoid(inX):
    return 1.0/(1+exp(-inX))
    # 注意：因为dataMatrix 与weights均为numpy矩阵，相乘也是numpy矩阵，而math.exp()函数只处理python标准数值，所以应用numpy的exp方法


def gradAscent(dataMatIn, classLabels):
    dataMatrix = mat(dataMatIn)  # 将数据转换为numpy矩阵
    labelMat = mat(classLabels).transpose()  # 同上，并转置
    m, n = shape(dataMatrix)  # 获取数据矩阵的形状
    alpha = 0.001  # 定义步长
    maxCycles = 500  # 限定最大迭代次数
    weights = ones((n, 1))  # 初始化所有参数为1,类型为n*1的numpy矩阵
    for k in range(maxCycles):
        h = sigmoid(dataMatrix*weights)
        error = (labelMat-h)  # 计算偏差值
        weights = weights+alpha*dataMatrix.transpose()*error  # 此处经过了一个简单的数学推导，由梯度上升的方程推导得来
    return weights


def plotBestFit(weights):
    import matplotlib.pyplot as plt  # 导入绘图所需的包
    dataMat, labelMat = loadDataSet()
    dataArr = array(dataMat)
    """因为列表的索引只能是整数或者是切片，不能是元组，
    而numpy数组可以可以传入一个以逗号隔开的索引列表来获取单个元素，所以要如此处理
    否则下一个for循环中的一些语句会报错：list indices must be integers or slices, not tuple"""
    n = shape(dataArr)[0]  # 获取矩阵的行数（即数据的组数）
    xcord1 = []
    xcord2 = []
    ycord1 = []
    ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i, 1])  # 存储“1”类的点的两个特征于xcord1和ycord2中，下同
            ycord1.append(dataArr[i, 2])
        else:
            xcord2.append(dataArr[i, 1])
            ycord2.append(dataArr[i, 2])
    fig = plt.figure()  # 创建一个窗口
    ax = fig.add_subplot(111)  # 将画布分割为1行1列，图像绘制在第一块，ax代表这块绘图的区域
    # 绘制散点图(scatter),点的形状设置为square
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')  # 点的形状为默认的圆
    x = arange(-3.0, 3.0, 0.1)
    y = (-weights[0]-weights[1]*x)/weights[2]
    """0是两个分类（类别1和类别0）的分界处），所以我们设定0=w0*x0+
    w1*x1+w2*x2（x0=1）,求解出x2和x1（x和y）的关系式"""
    ax.plot(x, y)  # 绘制出分界线
    plt.xlabel('X1')  # 设置坐标轴的名称
    plt.ylabel('X2')
    plt.show()


def stocGradAscent0(dataMatrix, classLabels):
    m, n = shape(dataMatrix)
    alpha = 0.01
    weights = ones(n)
    for i in range(m):
        h = sigmoid(sum(dataMatrix[i]*weights))
        error = classLabels[i]-h
        weights = weights+alpha*error*dataMatrix[i]
    return weights


def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    m, n = shape(dataMatrix)
    weights = ones(n)
    for j in range(numIter):
        dataIndex = list(range(m))
        """使用list方法(迭代器)将range返回的可迭代对象变为列表（否则书中代码无法在python3下运行）,
		python3中range(3)并不会返回列表[0,1,2]，而是range(0,3)"""
        for i in range(m):
            alpha = 4/(1.0+j+i)+0.01
            randIndex = int(random.uniform(0, len(dataIndex)))  #在所给范围中随机返回一个浮点数
            h = sigmoid(sum(dataMatrix[randIndex]*weights))
            error = classLabels[randIndex]-h
            weights = weights+alpha*error*dataMatrix[randIndex]
            del(dataIndex[randIndex])  #从下标集中删去已使用过的样本的下标
    return weights
