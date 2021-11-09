从Titanic.csv读入

```
new_df = df = pd.read_csv('数据处理/Titanic.csv')
print(df.isnull().sum())
```

得到缺失信息：

```
PassengerId      0
Survived         0
Pclass           0
Name             0
Sex              0
Age            177
SibSp            0
Parch            0
Ticket           0
Fare             0
Cabin          687
Embarked         2
dtype: int64
```

Embarked 采用众数‘S'填充。

Cabin 缺失过多（一共891个样例）无法填充，选择删除该列。

Age采用KNN填充，K取值为8，取最近的8个的Age的平均值代替，选择的维度有：

```
['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
```

处理完后写到Titanic_new.csv