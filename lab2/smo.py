
# you can do anything necessary about the model
from operator import mod
from numpy import double
import numpy as np

def generate_data(dim = 10, num = 100, random_seed = -1):
    if random_seed != -1:
        np.random.seed(random_seed)
    x = np.random.normal(0, 10, [num, dim])

    if random_seed != -1:
        np.random.seed(random_seed)
    coef = np.random.uniform(-1, 1, [dim, 1])

    pred = np.dot(x, coef)
    pred_n = (pred - np.mean(pred)) / np.sqrt(np.var(pred))
    label = np.sign(pred_n)

    if random_seed != -1:
        np.random.seed(random_seed)
    mislabel_value = np.random.uniform(0, 1, num)
    mislabel = 0

    for i in range(num):
        if np.abs(pred_n[i]) < 1 and mislabel_value[i] > 0.9 + 0.1 * np.abs(pred_n[i]):
            label[i] *= -1
            mislabel += 1
    return x, label, mislabel/num

X_data, y_data, mislabel = generate_data(dim=5, num=100) 

# split data
def random_Split_data(X: np.ndarray, y:np.ndarray, rate = 0.7, random_seed: int = -1):
    data: np.ndarray = np.hstack((X, y))
    m, n = data.shape
    if random_seed != -1:
        np.random.seed(abs(random_seed))
    np.random.shuffle(data)
    
    row_split = int(m * rate)
    X_train = data[0: row_split, 0: -1]
    y_train = data[0: row_split, -1: ]
    X_test = data[row_split: m, 0: -1]
    y_test = data[row_split: m, -1: ]
    
    return X_train, y_train, X_test, y_test


X_train, y_train, X_test, y_test = random_Split_data(X_data, y_data, rate=0.7)



class SMO:
    def __init__(self, X, y, C, toler, maxIter):  # samples labels constance tolerate maxIteration
        self.X = X
        self.y = y
        self.C = C
        self.toler = toler
        self.maxIter = maxIter
        self.N = self.X.shape[0]
        self.b = 0
        self.a = np.zeros(self.N)
        self.K = np.zeros((self.N, self.N))
        self.o = 0.5
        for i in range(self.N):
            for j in range(self.N):
                t = 0
                for k in range(self.X.shape[1]):
                    t += (self.X[i][k] - self.X[j][k])**2
                self.K[i][j] = np.exp(-t/(2*self.o**2))

    def select_J(self, i):  # random choose j which is not equal to i.
        j = i
        while j == i:
            j = np.random.randint(0, self.N)
        return j

    def fix_alpha(self, a, H, L):
        if a > H:
            a = H
        if a < L:
            a = L
        return a

    def cal_Ei(self, j):
        t = 0
        for i in range(self.N):
            t += self.a[i]*self.y[i]*self.K[i][j]
        return t + self.b - self.y[j]

    def update(self):
        iter = 0
        while iter < self.maxIter:
            alphaPairsChanged = 0
            for i in range(self.N):
                Ei = self.cal_Ei(i)
                if (Ei < -self.toler and self.a[i] < self.C) or (Ei > self.toler and self.a[i] > 0):
                    j = self.select_J(i)  # choose j != i
                    Ej = self.cal_Ei(j)
                    aiold = self.a[i]
                    ajold = self.a[j]
                    if self.y[i] != self.y[j]:
                        L = max(0, self.a[j] - self.a[i])
                        H = min(self.C, self.C + self.a[j] - self.a[i])
                    else:
                        L = max(0, self.a[j] + self.a[i] - self.C)
                        H = min(self.C, self.a[j] + self.a[i])
                    if L == H:
                        print("L==H")
                        continue
                    eta = self.K[i][i] + self.K[j][j] - 2*self.K[i][j]
                    if eta <= 0 :
                        print('eta <= 0')
                        continue
                    ajnew = ajold + self.y[j]*(Ei - Ej)/eta
                    ajnew = self.fix_alpha(ajnew, H, L)
                    if abs(ajnew - ajold) < 0.00001:
                        print("j not moving enough")
                        continue
                    self.a[j] = self.fix_alpha(ajnew, H, L)
                    self.a[i] = aiold + self.y[i]*self.y[j]*(ajold - ajnew)
                    b1 = -Ei - self.y[i]*self.K[i][i]*(self.a[i] - aiold) - self.y[j]*self.K[i][j]*(ajnew - ajold) + self.b
                    b2 = -Ej - self.y[i]*self.K[i][j]*(self.a[i] - aiold) - self.y[j]*self.K[j][j]*(ajnew - ajold) + self.b
                    if 0 < self.a[i] < self.C:
                        self.b = b1
                    elif 0 < self.a[j] < self.C:
                        self.b = b2
                    else:
                        self.b = (b1 + b2)/2
                    alphaPairsChanged += 1
            if alphaPairsChanged == 0:
                iter += 1
            else:
                iter = 0
        return self.b, self.a

    def score(self, X, y):
        K = np.zeros((X.shape[0], self.N))
        for i in range(X.shape[0]):
            for j in range(self.N):
                t = 0
                for k in range(self.X.shape[1]):
                    t += (self.X[i][k] - self.X[j][k]) ** 2
                K[i][j] = np.exp(-t / (2 * self.o ** 2))
        y_pre = []
        for i in range(X.shape[0]):
            t = 0
            for j in range(self.N):
                t += self.a[j]*self.y[j]*K[i][j]
            y_pre.append(t + self.b)
        s = 0
        print(y_pre)
        for i in range(X.shape[0]):
            if y_pre[i] < 0 and y[i] == -1:
                s += 1
            elif y_pre[i] > 0 and y[i] == 1:
                s += 1
        return s/X.shape[0]

m = SMO(X_train, y_train, 1, 0.001, 100)

score = m.score(X_test, y_test)
print(score)
