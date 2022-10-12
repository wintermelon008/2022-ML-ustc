#### 目录结构

```
--Loan.ipynb

--Logistic.py

--ReadMe.md

--Report.pdf
```



**实验中涉及到的外部库：**

**pandas、random、numpy、 matplotlib、 pylab** 



#### Logistic.py

```python
def __init__(self, penalty="l2", gamma=1, random_seed: int = -1)
    penalty: 正则化方式的种类，默认 L2
    gamma: 正则化参数，默认 1
    random_seed: 随机数种子，为 -1 则代表不固定，否则固定随机数种子。默认 -1
```

```python
def fit(self, X: np.ndarray, y: np.ndarray , lr=0.001, tol=1e-7, max_iter=1e7, method='gradient')
	X: 训练集特征矩阵
    y: 训练集分类矩阵
    lr: 学习率
    tol: 迭代阈值
    max_iter: 最大迭代次数
    method: 迭代方法，可选梯度下降 'gradient' 和牛顿迭代 'newton'
```



#### Loan.ipynb

直接顺序运行即可。