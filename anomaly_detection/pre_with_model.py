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
import sys
import os
CURRENT_DIR = os.path.split(os.path.abspath(__file__))[0]  # 当前目录
config_path = CURRENT_DIR.rsplit('/', 1)[0]  # 上三级目录
sys.path.append(config_path)
import config
import model
from sklearn.preprocessing import MinMaxScaler
import tqdm
import faulthandler
from dataset import MyDataset
# 在import之后直接添加以下启用代码即可
# faulthandler.enable()
# 后边正常写你的代码

logging.basicConfig(filename='train.log', level=logging.DEBUG)


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
    for i in range(10, 14):
        threshold_list.append(i+0.3)
        # for j in range(10):
        #     threshold_list.append(i+j*0.2)
    test_dataset = MyDataset(path=config.after_test_dataset_path)
    testset, _, _ = test_dataset.create_dataset()

    anomaly_dataset = MyDataset(path=config.anomaly_dataset_path)
    anomaly_set, _, _ = anomaly_dataset.create_dataset()



    batch_size = config.batch_size

    model = torch.load('model.pth',map_location=torch.device(config.device))
    model = model.to(config.device)

    threshold_accuracy = dict(normal=[], anormal=[])
    for i in range(len(threshold_list)):
        THRESHOLD = threshold_list[i]
        print(f"THRESHOLD:{THRESHOLD}")

        predictions, pred_losses = predict(model, testset)
        # print(pred_losses)
        correct = sum(l <= THRESHOLD for l in pred_losses)
        level_3 = 0
        level_1 = 0

        accu_nor = correct/len(test_dataset)
        print(f'Correct normal predictions: {correct}/{len(test_dataset)}, the accuracy:{accu_nor}')
        threshold_accuracy['normal'].append(accu_nor)

        predictions, pred_losses = predict(model, anomaly_set)
        correct = sum(l > THRESHOLD for l in pred_losses)
        for l in pred_losses:
            if l > 58:
                level_3 += 1
            else:
                level_1 += 1
        print(f"found {level_1} level 1 anomaly points, {level_3} level 3 anomaly points")
        accu_anormal = correct/len(anomaly_set)
        print(f'Correct anomaly predictions: {correct}/{len(anomaly_set)}, the accuracy of anomaly:{accu_anormal}')
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