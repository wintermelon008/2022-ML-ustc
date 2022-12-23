# 数据预处理的工具

import pandas as pd
import numpy as np
import random as rd


def random_FillMissingData(df: pd.DataFrame, random_seed: int = -1, if_debug = False) -> pd.DataFrame:
    '''
        该函数负责将传入的数据集进行缺失项的随机填充。
        
        ---------------
        Input:
            df: pd.DataFrame
                原始的有空缺数据集
            
            random_seed: int
                随机数种子。范围 0~2^20。若为 -1 则代表不指定
                
        OutPut:
            df: pd.DataFrame
                进行随机填充后的数据集

    '''
    
    _df = df.copy()
    for column in _df.columns:
        data = _df[column].copy()
        empty_rows = data.isnull()
        if (random_seed != -1):
            rd.seed(abs(random_seed))
        data[empty_rows] = rd.choices(data[~empty_rows].values, k = empty_rows.sum())
        _df[column] = data
        
    if if_debug == True:
        _df.to_csv("./debug/random_fill_output.csv", index=False) 
        
    return _df


def Normalization(df: pd.DataFrame, if_debug = False) -> pd.DataFrame:
    '''
        该函数负责将传入的数据集进行最值归一化。
        
        ---------------
        Input:
            df: pd.DataFrame
                原始的无空缺数据集
                
        OutPut:
            df: pd.DataFrame
                进行归一化后的数据集

    '''
    
    _df = df.copy()
    _df = _df.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x))).round(8)
    
    
    if if_debug == True:
        _df.to_csv("./debug/normalization.csv", index=False) 
    
    return _df


def random_Split_data(data: pd.DataFrame, rate = 0.75, random_seed: int = -1, if_debug = False):
    '''
        该函数负责将传入的数据集按比例进行随机拆分。
        
        ---------------
        Input:
            df: pd.DataFrame
                原始数据集
                
            rate: float
                划分比例。为 训练集/总体 的值。默认 0.75
            
            random_seed: int
                随机数种子。范围 0~2^20。若为 -1 则代表不指定
                
        OutPut:
            (X_train, y_train, X_test, y_test): tuple[ndarray, ndarray, ndarray, ndarray]
                训练集与测试集

    '''
    

    m, n = data.shape
    if random_seed != -1:
        np.random.seed(abs(random_seed))
    data.reindex(np.random.permutation(data.index))

    row_split = int(m * rate)
    X_train = data.iloc[0: row_split, 0: n - 1].values
    y_train = data.iloc[0: row_split, n - 1: ].values
    X_test = data.iloc[row_split: m, 0: n - 1].values
    y_test = data.iloc[row_split: m, n - 1: ].values
    
    if if_debug == True:
        pass    #TODO
    
    return X_train, y_train, X_test, y_test



