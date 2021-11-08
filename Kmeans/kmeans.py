import numpy as np
import random as rd
from numpy.core.arrayprint import DatetimeFormat

def LoadData(filename):
    dataMat = []
    fo = open(filename)
    for line in fo.readlines():
        line = line.strip().split(',')
        dataMat.append([float(x) for x in line])
    return np.mat(dataMat)

def DrawData(dataMat): # 二维点的绘制
    import matplotlib.pyplot as plt
    vecx = np.array(dataMat[...,0].T)
    vecy = np.array(dataMat[...,1].T)
    plt.scatter(vecx, vecy)
    plt.show()


def GetDis(vecA, vecB):
    return np.sqrt(np.sum(np.power(vecA - vecB, 2)))

# 随机k个质点
def RandCent(dataMat, k):
    n = dataMat.shape[1]
    centriods = np.mat(np.zeros((k, n))) # k 个 n 维向量表示 k 个质点的坐标, 写成矩阵
    for j in range(n):
        minJ = np.min(dataMat[...,j])
        maxJ = np.max(dataMat[...,j])
        rangeJ = maxJ - minJ
        centriods[...,j] = minJ + rangeJ * np.random.rand(k, 1)
    return centriods

def Kmeans(dataMat, k):
    m = dataMat.shape[0]
    belong = np.mat(np.zeros((m, 2))) # belong[i] = [id, dist] 点i属于的质点id，到质点id的距离dist
    centriods = RandCent(dataMat, k)
    Changed = True
    while Changed:
        Changed = False
        for i in range(m): # 更新每个点的最近质点
            minDist = np.inf
            minIndex = -1
            for j in range(k):
                distJI = GetDis(centriods[j,...], dataMat[i,...])
                if distJI < minDist:
                    minDist = distJI
                    minIndex = j
            if belong[i, 0] != minIndex: 
                Changed = True
                belong[i] = minIndex, minDist
        #print(centriods)
        for cent in range(k): # 对每个质点管辖的点求平均值作为新质点坐标
            points = []
            for i in range(m):
                if belong[i, 0] == cent:
                    vec = []
                    for j in range(dataMat.shape[1]):
                        vec.append(dataMat[i,j])
                    points.append(vec)
            centriods[cent,...] = np.mean(points, axis=0)
    return centriods, belong

def RandColor():
    import random
    colorArr = ['1','2','3','4','5','6','7','8','9','A','B','C','D','E','F']
    color = ""
    for i in range(6):
        color += colorArr[random.randint(0,14)]
    return "#"+color

def DrawResult(dataMat, centrios, belong): # 二维聚类的散点图绘制
    import matplotlib.pyplot as plt
    colorType = []
    for i in range(centrios.shape[0]):
        colorType.append(RandColor())
    colors = []
    for i in range(dataMat.shape[0]):
        colors.append(colorType[int(belong[i, 0])])
    x = []
    y = []
    for i in range(dataMat.shape[0]):
        x.append(dataMat[i, 0])
        y.append(dataMat[i, 1])
    x = np.array(x)
    y = np.array(y)
    plt.scatter(x, y, c=colors, s=15)
    plt.scatter(np.array(centrios[...,0].T), np.array(centrios[...,1].T), c='', s=40)
    plt.show()

dataMat = LoadData('Kmeans/k_means_data.csv')
centriods, belong = Kmeans(dataMat, 3)
DrawResult(dataMat, centriods, belong)