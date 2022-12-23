# 评价函数
import numpy as np

def Err_label(y_pre: np.ndarray, y_real: np.ndarray):
    '''
        分类错误率计算函数
        ---------------
        Input:
            y_pre: np.ndarray
                预测结果
                
            y_real: np.ndarray
                真实结果
                
        OutPut:
            score: float
                分类错误率	
    '''
    if isinstance(y_pre, list):
        y_pre = np.array(y_pre).reshape(-1, 1)
        
    m, _ = y_pre.shape
    score = 0
    
    for i in range(m):
        if y_pre[i] != y_real[i]:
            score += 1
            
    return score * 1.0 / m


def Accuracy(y_pre: np.ndarray, y_real: np.ndarray):
    '''
        分类精度计算函数
        ---------------
        Input:
            y_pre: np.ndarray
                预测结果
                
            y_real: np.ndarray
                真实结果
                
        OutPut:
            score: float
                分类精度	
    '''
    return 1 - Err_label(y_pre, y_real)