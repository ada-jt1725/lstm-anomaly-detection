#!/usr/bin/env python
# coding: utf-8
import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import copy
import config
import model
from sklearn.preprocessing import MinMaxScaler
import tqdm
import faulthandler
# 在import之后直接添加以下启用代码即可
# faulthandler.enable()
# 后边正常写你的代码


# 创建表头
def make_signal_list():
    signal_list = list()
    for i in range(-config.slide_range, 0):
        signal_list.append('signal' + str(i))
    return signal_list

if __name__ == "__main__":
    # 标准化
    # dataset = pd.read_csv(config.base_path)
    dataset = pd.read_excel(config.base_path_2)
    columns_to_drop = ['时间异常检测', 'label复查', '监控点名称', '类型名称','时间段', '查', 'Unnamed: 10']
    # 使用 drop 方法删除列
    dataset = dataset.drop(columns=columns_to_drop)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = MinMaxScaler(feature_range=(-1, 1))

    for col in config.signal_columns:
        all_data = dataset[col]
        all_data = scaler.fit_transform(all_data.values.reshape(-1, 1))
        dataset[col] = all_data
        # 平移
        for i in range(-config.slide_range, 0):
            dataset[col + str(i)] = dataset[col].shift(i)

    end = len(dataset)
    anomaly_list=[]
    # 提取异常点
    for i in range(end):
        if dataset['label'][i] > 0:
            anomaly_list.append(dataset.loc[i])
    data1 = pd.DataFrame(data=anomaly_list)
    # PATH为导出文件的路径和文件名
    data1.to_csv(config.anomaly_dataset_path, index=False)

    anomaly = 0
    # 对于标记为异常的点，及异常点上下平移范围的点，都剔除掉
    for i in range(end):
        if dataset['label'][i] > 0:
            anomaly += 1
            item_start = max(i - config.slide_range, 0)
            item_end = min(end, i + config.slide_range)
            for col in config.signal_columns:
                for j in range(item_start, item_end):
                    dataset[col+'-1'][j] = np.nan


    print('anomaly:', anomaly)
    dataset = dataset.dropna() # 删除包含缺失值的行
    dataset = dataset.reset_index(drop=True)

    train_df, test_df = train_test_split(
        dataset,
        test_size=0.05
    )

    print("data saving to file...")
    data2 = pd.DataFrame(data=train_df)
    # PATH为导出文件的路径和文件名
    data2.to_csv(config.after_train_dataset_path, index=False)

    data3 = pd.DataFrame(data=test_df)
    # PATH为导出文件的路径和文件名
    data3.to_csv(config.after_test_dataset_path, index=False)
    print("data saved to file!")


