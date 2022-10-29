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

X_data, y_data, mislabel = generate_data(dim=10, num=500) 

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



def seconde_min(lt):
    d={}         
    for i, v in enumerate(lt):
        d[v]=i   
    lt.sort()    
    y = lt[1]
    return d[y]      

def seconde_max(lt):
    d={}         
    for i, v in enumerate(lt):
        d[v]=i   
    lt.sort()    
    y = lt[-2]
    return d[y]  


# you can do anything necessary about the model
from numpy import double


class SVM2:
    def __init__(self, X:np.ndarray, y:np.ndarray):
        """
        You can add some other parameters, which I think is not necessary
        """
        m, _ = X.shape
        self.X = X
        self.y = y
        self.K = self.X @ self.X.T
        
        self.alpha = np.zeros((m, 1))
        self.b = np.random.uniform(low=0.0, high=1.0, size=1)
        self.err = np.zeros((m, 1))
        self._update_e()

       
        

    def _cut(self, low, high, a2_uncut):
        if a2_uncut > high:
            return high
        elif a2_uncut < low:
            return low
        return a2_uncut


    def _update_alpha(self, a1_index, a2_index, gamma):

        eta = self.K[a1_index][a1_index] + self.K[a2_index][a2_index] - 2 * self.K[a1_index, a2_index]
        alpha1_old = self.alpha[a1_index, :][0]
        alpha2_old = self.alpha[a2_index, :][0]
        y1 = self.y[a1_index, :][0]
        y2 = self.y[a2_index, :][0]
        err1 = self.err[a1_index, :][0]
        err2 = self.err[a2_index, :][0]

        if eta > 0:
            a2_uncut = alpha2_old + y2 * (err1 - err2) / eta
        else:
            # print("Eta <= 0")
            return False
        
        if y1 != y2:
            low = max(0, alpha2_old - alpha1_old)
            high = min(gamma, gamma + alpha2_old - alpha1_old)
        else:
            low = max(0, alpha2_old + alpha1_old - gamma)
            high = min(gamma, alpha2_old + alpha1_old)

        if low > high:
            # print("Low == High")
            return False

        alpha2_new = self._cut(low, high, a2_uncut)
        

        if (abs(alpha2_old - alpha2_new) < 1e-3):
            # print("Alpha2 moves too slow")
            return False

        alpha1_new = alpha1_old + y1 * y2 * (alpha2_old - alpha2_new)
        self._update_b(a1_index, a2_index, alpha1_new, alpha2_new, gamma)

        self.alpha[a1_index] = alpha1_new
        self.alpha[a2_index] = alpha2_new

        self._update_e()

        return True


    def _update_b(self, a1_index, a2_index, a1_new, a2_new, gamma):
        alpha_1 = self.alpha[a1_index, :][0] 
        alpha_2 = self.alpha[a2_index, :][0] 
        y1 = self.y[a1_index, :][0]
        y2 = self.y[a2_index, :][0]
        err1 = self.err[a1_index, :][0]
        err2 = self.err[a2_index, :][0]

        b1 = -err1 - y1 * self.K[a1_index][a1_index] * (a1_new - alpha_1) - y2 * self.K[a2_index, a1_index] * (a2_new - alpha_2) + self.b
        if 0 < alpha_1 < gamma:
            self.b = b1
            return
      
        b2 = -err2 - y1 * self.K[a1_index, a2_index] * (a1_new - alpha_1) - y2 * self.K[a2_index][a2_index] * (a2_new - alpha_2) + self.b
        if 0 < alpha_2 < gamma:
            self.b = b2
            return
        
        self.b = 0.5 * (b1 + b2)
        return


    def _update_e(self):
        alpha_y = self.alpha * self.y
        self.err = self.K @ alpha_y + self.b - self.y


    def fit(self, gamma = 1, lr = 0.01, tol=1, max_times = 100, silent = True, epslion = 0):
        """
        Fit the coefficients via your methods
        """
        m, n = self.X.shape

        loss_list = []
        times = 0
        
        while times < max_times:
            alpha_y = self.alpha * self.y
            loss = self.alpha.sum() - 0.5 * (alpha_y.T @ self.X @ self.X.T @ alpha_y)[0][0]

            i_list = []

            for i in range(m):
                alpha_i = self.alpha[i, :][0]
                err_i = self.err[i, :][0]
                if (0 < alpha_i < gamma and abs(err_i - 1) > epslion):
                    val = (abs(err_i) - epslion) * 10
                    i_list.append((val, i))
                elif alpha_i == 0 and err_i < 1 - epslion:
                    val = - err_i + 1 - epslion
                    i_list.append((val, i))
                elif alpha_i >= gamma and err_i > 1 + epslion:
                    val = err_i - 1 - epslion
                    i_list.append((val, i))


            if times > 2 and len(i_list) == 0:
                loss_list.append(loss)
                break

            i_list.sort(key = takefirst)

            while len(i_list) > 0:
                _, a1 = i_list.pop()
                e1 = self.err[a1, :][0]

                err_dict = []
                err_list = []

                for i in self.err.tolist():
                    err_list.append(i[0])
                # print(err_list)
                for index, value in enumerate(err_list):
                    err_dict.append((value, index))
                err_dict.sort(key=takefirst)

                k = 0
                if e1 > 0:
                    while k < m:
                        a2 = err_dict[k][1]
                        if a1 != a2 and self._update_alpha(a1, a2, gamma):
                            break
                        k += 1
                else:
                    while k < m:
                        a2 = err_dict[-1 - k][1]
                        if a1 != a2 and self._update_alpha(a1, a2, gamma):
                            break
                        k += 1

                if k == m:
                    continue  # change i
                else:
                    break
            

            if times >= 2 and abs(loss_list[-1] - loss) < tol:
                loss_list.append(loss)
                break

            loss_list.append(loss)
            times += 1

        if times == max_times:
            times -= 1
        return loss_list, times


    def predict(self, X:np.ndarray):
        """
        Use the trained model to generate prediction probabilities on a new
        collection of data points.
        """

        m, n = X.shape

        ans = []
        for i in range(m):
            alpha_y = self.alpha * self.y
            temp = (alpha_y.T @ self.X @ X[i, :].T)[0] + self.b
            if temp > 0:
                ans.append(1)
            else:
                ans.append(-1)
        
        return np.array(ans).reshape(-1, 1)

def takefirst(elm):
    return elm[0]
        

def takefirst(elm):
    return elm[0]

model2 = SVM2(X_train, y_train)
loss, times = model2.fit(gamma = 0.1, tol = 1e-7, max_times=5000, epslion=0.1)
# print(model2.alpha)
# print(loss)
# print(model2.b)
print(times)
pre = model2.predict(X_test)
# print(pre)

def model_cmp(y_pre:np.ndarray, y_test:np.ndarray):
    # y should be in shape m x 1
    corr = 0
    sum = 0
    m, n = y_pre.shape
    
    for i in range(m):
        if y_pre[i] == y_test[i]:
            corr += 1
        sum += 1
    return corr/sum

print(model_cmp(pre, y_test))