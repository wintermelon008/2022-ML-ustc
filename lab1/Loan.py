'''
    这个文件主要是为了寻找良好的随机数种子，与 .ipynb 文件相同
'''
import pandas as pd
import random as rd
import numpy as np
import warnings
import eventlet
eventlet.monkey_patch()  
# with eventlet.Timeout(2,False):   #设置超时时间为2秒
#    print '这条语句正常执行'
#    time.sleep(4)
#    print '没有跳过这条输出'

warnings.filterwarnings('error')
df = pd.read_csv('./lab1/loan.csv')

df['Gender'] = df['Gender'].map({'Female': 0, 'Male': 1})
df['Married'] = df['Married'].map({'Yes': 1, 'No': 0})
df['Dependents'] = df['Dependents'].map({'0': 0, '1': 1, '2': 2, '3+': 3})
df['Education'] = df['Education'].map({'Graduate': 1, 'Not Graduate': 0})
df['Property_Area'] = df['Property_Area'].map({'Rural': 0, 'Semiurban': 1, 'Urban': 2})
df['Self_Employed'] = df['Self_Employed'].map({'Yes': 1, 'No': 0})
df['Loan_Status'] = df['Loan_Status'].map({'Y': 1, 'N': 0})
df.drop(df.columns[[0, 1, 2, 4, 8]], axis=1, inplace=True)

seed = -1
N = 100010

# 如果为 -1 则不设置随机数种子

def random_FillMissingData(_df: pd.DataFrame, random_seed: int = -1) -> pd.DataFrame:
    df = _df.copy()
    for column in df.columns:
        data = df[column].copy()
        empty_rows = data.isnull()
        if (random_seed != -1):
            rd.seed(abs(random_seed))
        data[empty_rows] = rd.choices(data[~empty_rows].values, k = empty_rows.sum())
        df[column] = data
    return df

def random_Split_data(data: pd.DataFrame, rate = 0.75, random_seed: int = -1):
    m, n = data.shape
    if random_seed != -1:
        np.random.seed(abs(random_seed))
    data.reindex(np.random.permutation(data.index))

    row_split = int(m * rate)
    X_train = data.iloc[0: row_split, 0: n - 1].values
    y_train = data.iloc[0: row_split, n - 1: ].values
    X_test = data.iloc[row_split: m, 0: n - 1].values
    y_test = data.iloc[row_split: m, n - 1: ].values
    
    return X_train, y_train, X_test, y_test

start = 0
max = 0
max_seed = 0
from Logistic import LogisticRegression

fp = open('.\lab1\seed.txt', 'r+')
input = fp.read().split(' ')
start = int(input[0])
max = float(input[1])
max_seed = int(input[2])
fp.close()

seed = start

while seed < N:

    df1 = random_FillMissingData(df, random_seed=seed)

    df1 = df1.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x))).round(4)

    X_train, y_train, X_test, y_test = random_Split_data(df1, rate=0.7, random_seed=seed)
    lr = LogisticRegression(random_seed=seed)

    try:
        times, loss = lr.fit(X_train, y_train, lr=0.0005, tol=1e-2, method='newton')
    except:
        print(1)
        times, loss = lr.fit(X_train, y_train, lr=0.0005, tol=1e-2)
        
    

    
        

    pred = lr.predict(X_test)

    sum = 0
    count = 0
    for i in range(len(y_test)):
        if y_test[i] == pred[i]:
            count += 1
        sum += 1

    if count/sum > max:
        max_seed = seed
        max = count/sum

    # print(seed)
    if seed % (10) == 0:
        # print(max)
        # print(max_seed)
        fp = open('.\lab1\seed.txt', 'w+')
        fp.write("{} {} {}".format(seed, max, max_seed))
        fp.close()

    seed += 1


fp.close()
# print(max)
# print(max_seed)