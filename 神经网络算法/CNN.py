from numpy import *
from numpy import dtype
from pandas import *
from numpy import array
from numpy import deprecate_with_doc
from numpy.lib.histograms import _ravel_and_check_weights
import pandas as pd
import sklearn.datasets
import matplotlib.pyplot as plt

def loadData(filename, ans=True):
    df = read_csv(filename)
    data = mat(df)
    n = len(data)
    m = len(data[0].tolist()[0])
    Map = {'male':0.0, 'female':1.0, 'S':0.0, 'Q': 1.0, 'C':2.0}
    for i in range(n):
        data[i,1] = Map[data[i, 1]]
        data[i,6] = Map[data[i, 6]]
    for i in range(n):
        for j in range(m-1):
            if isinstance(data[i,j],int):
                data[i,j] = float(data[i, j])
    Yt = data[...,-1]
    Yt = array(Yt)
    X = array(data)
    if ans: return X[...,:m-1], Yt
    else: return X

def sigmod(M):
    M = mat(M, dtype='float')
    return 1 / (1 + exp(-M))

def Mul(A, B):
    C = A.copy()
    for i in range(len(A)):
        for j in range(len(A[0])):
            C[i, j] = A[i, j] * B[i, j]
    return C

def der_sig(M):
    return Mul(sigmod(M), (1 - sigmod(M)))

def Loss(Y, Yt):
    M = Mul(Y - Yt, Y - Yt)
    sum = 0
    for i in range(len(M)):
        sum += M[i,0]
    return sum / len(M)

loss_list = []

def CNN_adam(hiddenNums, X, Yt, alpha=0.01, times=2000):
    global loss_list
    n = len(X)
    featureNums = len(X[0])
    W1 = mat(random.rand(featureNums, hiddenNums))
    W2 = mat(random.rand(hiddenNums, 1))
    m1 = 0; m2 = 0; v1 = 0; v2 = 0
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    for t in range(1, times):
        z1 = dot(X, W1)
        H = sigmod(z1)
        z2 = dot(H, W2)
        Y = sigmod(z2)
        loss_list.append(sum(Mul((Y - Yt), Y - Yt)))
        if (t % 100 == 0): print("loss is : ", loss_list[-1])
        if (loss_list[-1] < 130): alpha = 0.001
#       print("Y = ", Y)
#		print("Yt = ", Yt)
		#print("loss is : ", loss_list[-1])
        dW1 = mat(dot(X.T, Mul(dot(Mul(Y - Yt, der_sig(dot(H, W2))), W2.T), der_sig(dot(X, W1)))),dtype='float')
        dW2 = mat(dot(H.T, Mul(Y - Yt, der_sig(dot(H, W2)))),dtype='float')

        gt1 = dW1
        m1 = beta1 * m1 + (1 - beta1) * gt1
        v1 = beta2 * v1 + (1 - beta2) * Mul(gt1, gt1)
        mhat1 = m1 / (1 - beta1 ** t)
        vhat1 = v1 / (1 - beta2 ** t)
        W1 -= alpha * Mul(mhat1, 1 / (sqrt(vhat1) + eps))

        gt2 = dW2
        m2 = beta1 * m2 + (1 - beta1) * gt2
        v2 = beta2 * v2 + (1 - beta2) * Mul(gt2, gt2)
        mhat2 = m2 / (1 - beta1 ** t)
        vhat2 = v2 / (1 - beta2 ** t)
        W2 -= alpha * Mul(mhat2, 1 / (sqrt(vhat2) + eps))
    return W1, W2

def CNN_sdg(hiddenNums, X, Yt, alpha=0.01, times=50000):
    global loss_list
    n = len(X)
    featureNums = len(X[0])
    X = mat(X); Yt = mat(Yt)
    W1 = mat(random.rand(featureNums, hiddenNums))
    W2 = mat(random.rand(hiddenNums, 1))
    for i in range(times):
        j = random.randint(0,n)
        z1 = dot(X[j], W1)
        H = sigmod(z1)
        z2 = dot(H, W2)
        Y = sigmod(z2)
        loss_list.append(sum(Mul((Y - Yt), Y - Yt)))
        if (i % 100 == 0): print("loss is : ", loss_list[-1])
        if loss_list[-1] < 250:
             alpha = 0.001
        dW1 = mat(dot(X[j].T, Mul(dot(Mul(Y - Yt[j], der_sig(z2)), W2.T), der_sig(z1))),dtype='float')
        dW2 = mat(dot(H.T, Mul(Y - Yt[j], der_sig(z2))),dtype='float')

        W1 -= alpha * dW1
        W2 -= alpha * dW2
    return W1, W2

def predict(X, W1, W2):
    z1 = dot(X, W1)
    H = sigmod(z1)
    z2 = dot(H, W2)
    Y = sigmod(z2)
    if Y > 0.5: return 1
    else: return 0

def Predict(filename, W1, W2):
    #begin 892
    df = read_csv(filename + 'test.csv')
    fw = df['PassengerId']
    df = df.drop(['PassengerId'], axis=1)
    df['Age'].fillna(df['Age'].mean(), inplace = True)
    df['Fare'].fillna(df['Fare'].mean(), inplace = True)

    data = mat(df)
    n = len(data)
    m = len(data[0].tolist()[0])
    Map = {'male':0.0, 'female':1.0, 'S':0.0, 'Q': 1.0, 'C':2.0}
    for i in range(n):
        data[i,1] = Map[data[i, 1]]
        data[i,6] = Map[data[i, 6]]
    for i in range(n):
        for j in range(m):
            data[i,j] = float(data[i, j])
    
    Ans = [[fw[i], predict(data[i], W1, W2)] for i in range(n)]
    fw = pd.DataFrame(Ans, columns=['PassengerId', 'Survived'])
    fw.to_csv(filename + 'submission.csv', index=False)
    

random.seed(0)
X, Yt = loadData('神经网络算法/Titanic_new.csv')
W1, W2 = CNN_adam(hiddenNums=10, X=X, Yt=Yt)
Predict('神经网络算法/', W1, W2)
plt.plot(loss_list) # 绘制 loss 曲线变化图
plt.show()