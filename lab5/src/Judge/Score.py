# 评价函数
import numpy as np





def Labelize(pre: np.ndarray, max = 3.5, min = -0.5) -> list:
    '''
        标签化函数：将连续结果离散化
        ---------------
        Input:
            pre: np.ndarray
                原始的预测结果
                
        OutPut:
            labelized_pre: list
                标签化后的预测结果	
    '''
    if isinstance(pre, list):
        return pre
    
    m, _ = pre.shape
    result = []
    gap = (max - min) / 4
    
    for i in range(m):
        if pre[i][0] >= max - gap:
            result.append(3)
        elif pre[i][0] >= max - 2 * gap:
            result.append(2)
        elif pre[i][0] >= min + gap:
            result.append(1)
        else:
            result.append(0)
    
    return result


def Err_label(y_pre: (list | np.ndarray), y_real: np.ndarray, max = 3.5, min = -0.5):
    '''
        分类错误率计算函数
        ---------------
        Input:
            y_pre: np.ndarray or list
                预测结果（连续）or 预测结果（离散）
                
            y_real: np.ndarray
                真实结果（离散）
                
        OutPut:
            score: float
                分类错误率	
    '''
    
    if isinstance(y_pre, np.ndarray):
        y_pre = Labelize(y_pre, max, min)
    
    if isinstance(y_real, list):
        y_real = np.array(y_real).reshape(-1, 1)
        
    m, _ = y_real.shape
    score = 0
    
    for i in range(m):
        if y_pre[i] != y_real[i]:
            score += 1
            
    return score * 1.0 / m


def Accuracy(y_pre: (list | np.ndarray), y_real: np.ndarray, max = 3.5, min = -0.5):
    '''
        分类精度计算函数
        ---------------
        Input:
            y_pre: np.ndarray or list
                预测结果（连续）or 预测结果（离散）
                
            y_real: np.ndarray
                真实结果
                
        OutPut:
            score: float
                分类精度	
    '''
    return 1 - Err_label(y_pre, y_real, max, min)