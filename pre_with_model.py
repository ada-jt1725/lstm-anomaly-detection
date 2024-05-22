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

logging.basicConfig(filename='train.log', level=logging.DEBUG)


def make_signal_list():
    signal_list = list()
    for col in config.signal_columns:
        for i in range(-config.slide_range, 0):
            signal_list.append(col + str(i))
    return signal_list

def create_dataset(df):
    array = np.array(df)
    # print(f"array.shape={array.shape}") # (82736, 400)
    # 将数组分割为两个部分s
    array1 = array[:, :config.slide_range]  # 选择所有行和前200列
    array2 = array[:, config.slide_range:]  # 选择所有行和后200列
    # print(f"array1.shape={array1.shape}, array2.shape={array2.shape}") # (82736, 200), (82736, 200)
    # 将两个部分合并为一个新的数组
    array_new = np.stack((array1, array2), axis=-1)
    dataset = [torch.tensor(s).float() for s in array_new]
    n_seq, seq_len, n_features = torch.stack(dataset).shape

    return dataset, seq_len, n_features

class MyDataset(Dataset):
    def __init__(self, path):
        self.signal_df = pd.read_csv(path)
        self.signal_columns = make_signal_list()

    def __len__(self):
        return len(self.signal_df)

    def __getitem__(self, idx):
        row = self.signal_df.loc[idx]
        x = row[self.signal_columns].values.astype(float)
        return x

def predict(model, dataset):
    predictions, losses = [], []
    criterion = nn.L1Loss(reduction='sum').to(config.device)
    with torch.no_grad():
        model = model.eval()
        for seq_true in dataset:
            seq_true = seq_true.to(config.device)
            # print(seq_true.shape)
            seq_pred = model(seq_true)
            loss = criterion(seq_pred, seq_true)
            predictions.append(seq_pred.cpu().numpy().flatten())
            losses.append(loss.item())
    return predictions, losses

if __name__ == "__main__":
    threshold_list=[]
    for i in range(8, 10):
        threshold_list.append(i+0.1)
        for j in range(10):
            threshold_list.append(i+j*0.2)
    test_dataset = MyDataset(path=config.after_test_dataset_path)
    test_dataset, _ = train_test_split(
        test_dataset,
        test_size=0.1
    )
    test_dataset, _, _ = create_dataset(test_dataset)

    anomaly_dataset = MyDataset(path=config.anomaly_dataset_path)
    anomaly_dataset, _ = train_test_split(
        anomaly_dataset,
        test_size=0.1
    )
    anomaly_train_df, _, _ = create_dataset(anomaly_dataset)



    batch_size = config.batch_size

    model = torch.load('model.pth')
    model = model.to(config.device)

    threshold_accuracy = dict(normal=[], anormal=[])
    for i in range(len(threshold_list)):
        THRESHOLD = threshold_list[i]
        print(f"THRESHOLD:{THRESHOLD}")

        predictions, pred_losses = predict(model, test_dataset)
        # print(pred_losses)
        correct = sum(l <= THRESHOLD for l in pred_losses)
        accu_nor = correct/len(test_dataset)
        print(f'Correct normal predictions: {correct}/{len(test_dataset)}, the accuracy:{accu_nor}')
        threshold_accuracy['normal'].append(accu_nor)

        predictions, pred_losses = predict(model, anomaly_train_df)
        correct = sum(l > THRESHOLD for l in pred_losses)
        accu_anormal = correct/len(anomaly_train_df)
        print(f'Correct anomaly predictions: {correct}/{len(anomaly_train_df)}, the accuracy of anomaly:{accu_anormal}')
        threshold_accuracy['anormal'].append(accu_anormal)

    print(threshold_accuracy)
    threshold_accuracy = pd.DataFrame(data=threshold_accuracy)
    # PATH为导出文件的路径和文件名
    threshold_accuracy.to_csv(config.accu_dataset_path, index=False)
    # best threshold = 6.7
    # normal = 0.8225108225108225
    # anormal = 0.6666666666666666

    '''
    best THRESHOLD:6.2
    epoch = 400, slide_range = (-100, 100)
    Correct normal predictions: 4146/4355, the accuracy:0.9520091848450057
    Correct anomaly predictions: 12/12, the accuracy of anomaly:1.0
    '''

    '''
    epoch = 400, slide_range = (-300, 0)
    THRESHOLD:9.5
    Correct normal predictions: 3816/4116, the accuracy:0.9271137026239067
    Correct anomaly predictions: 12/12, the accuracy of anomaly:1.0

    THRESHOLD:11.3
    Correct normal predictions: 4080/4116, the accuracy:0.9912536443148688
    Correct anomaly predictions: 11/12, the accuracy of anomaly:0.9166666666666666
    '''