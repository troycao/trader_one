# coding:utf-8
"""
将为nan的列替换为平均指
"""
import numpy as np

def fill_ndarray(ndarray):
    for i in range(ndarray.shape[1]):
        temp_column = ndarray[:, i]
        nan_count = np.count_nonzero(ndarray[temp_column != temp_column])
        if nan_count>0:
            temp_not_nan_col = temp_column[temp_column == temp_column] #取出当前不为nan列的值
            temp_column[np.isnan(temp_column)] = temp_not_nan_col.mean()
    return ndarray
