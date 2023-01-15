# 数据预处理的工具

import pandas as pd
import numpy as np
import random as rd
import seaborn as sns


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
    _df_ft = _df.iloc[:, :-1]
    _df_lb = _df.iloc[:, -1:]
    
    # print(_df_ft.shape)
    
    _df_ft = _df_ft.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x))).round(8)
    _df = pd.concat([_df_ft, _df_lb], axis=1)
    
    if if_debug == True:
        _df.to_csv("./debug/normalization.csv", index=False) 
    
    return _df




def Drop_noise(df: pd.DataFrame, if_debug = False):
    '''
        删除超出 0.07345 sigma 范围的数据
    '''
    _df = df.copy()
    df_describe = _df.describe()
    
    for column in df.columns:
        if column == 'label':
            break
        mean = df_describe.loc['mean',column]
        std = df_describe.loc['std',column]
        minvalue = mean - 0.07345*std   
        maxvalue = mean + 0.07345*std
        _df = _df[_df[column] >= minvalue]
        _df = _df[_df[column] <= maxvalue]
        
        
    if if_debug == True:
        _df.to_csv("./debug/drop_noise.csv", index=False) 
        
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


def random_Split_Data_Label(data: pd.DataFrame, labels = [], rate = 0.75, random_seed: int = -1, if_debug = False):
    if labels == []:
        labels = [0, 1, 2, 3]
        
    _, n = data.shape
        
    X_train = np.empty(shape=[0, n-1])
    y_train = np.empty(shape=[0, 1])
    X_test = np.empty(shape=[0, n-1])
    y_test = np.empty(shape=[0, 1])
    
    for label in labels:
        sub_data = data.query('label==' + str(label))
        sub_X_train, sub_y_train, sub_X_test, sub_y_test = random_Split_data(sub_data, rate, random_seed, if_debug)
        X_train = np.vstack((X_train, sub_X_train))
        y_train = np.vstack((y_train, sub_y_train))
        X_test = np.vstack((X_test, sub_X_test))
        y_test = np.vstack((y_test, sub_y_test))
        
    train = np.hstack((X_train, y_train))
    test = np.hstack((X_test, y_test))
   
    np.random.shuffle(train)
    np.random.shuffle(test)
    
    X_train = train[:, :-1]
    y_train = train[:, -1:]
    X_test = test[:, :-1]
    y_test = test[:, -1:]
    return X_train, y_train, X_test, y_test
        
        
        

def Find_useless_feature(data: pd.DataFrame):
    """
        从数据集中删除贡献不明显的特征
        我们规定，如果某特征在所有类别中分布都类似，则视作无用特征

    Args:
        data (pd.DataFrame): _description_
    """
    
    sub_data_0 = data.query('label==0')
    sub_data_1 = data.query('label==1')
    sub_data_2 = data.query('label==2')
    sub_data_3 = data.query('label==3')
    
    
    for i in range(0, 120):
        fe = "feature_" + str(i)
        fig = sns.distplot(sub_data_0[fe])
        fig_save = fig.get_figure()
        name = "./feature_fig/" + fe + "_l0.png"
        fig_save.savefig(name, dpi=300)
        fig_save.clear()
    
    for i in range(0, 120):
        fe = "feature_" + str(i)
        fig = sns.distplot(sub_data_1[fe])
        fig_save = fig.get_figure()
        name = "./feature_fig/" + fe + "_l1.png"
        fig_save.savefig(name, dpi=300)
        fig_save.clear()
    
    for i in range(0, 120):
        fe = "feature_" + str(i)
        fig = sns.distplot(sub_data_2[fe])
        fig_save = fig.get_figure()
        name = "./feature_fig/" + fe + "_l2.png"
        fig_save.savefig(name, dpi=300)
        fig_save.clear()
        
    for i in range(0, 120):
        fe = "feature_" + str(i)
        fig = sns.distplot(sub_data_3[fe])
        fig_save = fig.get_figure()
        name = "./feature_fig/" + fe + "_l3.png"
        fig_save.savefig(name, dpi=300)
        fig_save.clear()


    
    
    
def Delete_feature(data: pd.DataFrame):
    '''
        删除符合特定条件的特征
    '''
    
    drop_f = [
        'feature_2', 'feature_12',
        'feature_15', 'feature_16',
        'feature_20', 'feature_31',
        'feature_32', 'feature_55',
        'feature_75', 'feature_76',
        'feature_77', 'feature_85',
        'feature_87', 'feature_88',
        'feature_102', 'feature_113'
    ]
    
    leave_f = [
        'feature_9', 'feature_35',
        'feature_43', 'feature_48',
        'feature_67', 'feature_92',
        'feature_98', 'feature_99',
        'feature_100', 'feature_101',
        'feature_106', 'feature_114',
        'label'
    ]
    
    data_new = data[leave_f]
    
    # data_new = data.drop(drop_f, axis=1)
    return data_new
    
    
def Find_drop_feature(X_old: np.ndarray, X_new: np.ndarray):
    _, n_old = X_old.shape
    _, n_new = X_new.shape
    drop = []

    j = 0
    for i in range(0, n_old):
        if X_old[0, i] != X_new[0, j]:
            drop.append(i)
        else:
            j += 1
    return drop
            
