import pandas as pd
from pandas.core.indexes.base import ensure_index
from pandas.core.indexing import is_nested_tuple
import numpy as np

empty = []
n = 0
new_df = df = pd.read_csv('数据处理/Titanic.csv')

# print(df)
# print(df.isnull().sum())

df = df.drop(['Cabin'], axis=1)

# print(df['Embarked'].mode())

df['Embarked'] = df['Embarked'].fillna('S')

L = ['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare'] # 填补的列(用于参考的几个维度)
Colums = []

def Init():
    global new_df, empty, n

    new_df = df.copy() # 复制一份，最后用于修改和写出

    for i in range(len(df.columns)): # 将维度变成列号
        if df.columns[i] in L:
            Colums.append(i)

    df.columns = [i for i in range(len(new_df.columns))]
    index = []
    
    # 找到空的位置
    for i in df.columns:
        pos = df[df[i].isnull()].index.tolist()
        index.append(pos)
    tmp = []
    for i in df.columns:
        for j in index[i]:
            df.iloc[j, i] = -1
            tmp.append(j) # 第j行有缺失值
    empty = list(set(tmp)) # 去重 empty中是含缺失值的行编号
    n = len(df)
    df['id'] = [i for i in range(n)]
    #exit()

def KNN(K):
    def First(x):
        return x[0]
    global new_df, empty, n
    empty.sort()
    for i in empty:
        #print(i, end = ' ')
        dis = []
        # 求第i个数据和其它数据的欧氏距离
        for j in range(n):
            if i == j:
                continue
            dist = count = 0
            for k in Colums:
                if (df.iloc[i,k] == -1 or df.iloc[j,k] == -1):
                    continue
                count += 1
                dist += (df.iloc[i,k] - df.iloc[j,k])**2.0
            if not(count == 0):
                dist *= len(Colums) / count
                dis.append((dist, df['id'][j]))
        dis.sort(key = First)

        # 选取前K个求加权平均
        for j in Colums:
            if df.iloc[i,j] == -1:
                sum = 0
                count = 0
                for k in range(K):
                    if k >= len(dis):
                        break
                    if df.iloc[dis[k][1], j] == -1:
                        continue
                    sum += df.iloc[dis[k][1], j]
                    count += 1
                if not(count == 0):
                    #print(" OK")
                    new_df.iloc[i, j] = int(sum / count)

def Write_New():
    global new_df
    new_df.to_csv('数据处理/Titanic_new.csv', index=False)

Init()
KNN(8)
Write_New()

print(new_df.isnull().sum())