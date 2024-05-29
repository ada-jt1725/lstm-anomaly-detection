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
from tqdm import tqdm
from dataset import MyDataset
import faulthandler
# 在import之后直接添加以下启用代码即可
# faulthandler.enable()
# 后边正常写你的代码

logging.basicConfig(filename='train.log', level=logging.DEBUG)

def train_model(model, train_dataset, val_dataset, n_epochs):
    print("start training")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.L1Loss(reduction='sum').to(config.device)
    history = dict(train=[], val=[])
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = config.best_loss
    for epoch in tqdm(range(1, n_epochs + 1)):
        print("=========epoch=========")
        print(epoch)
        model = model.train()
        train_losses = []
        for idx, seq_true in enumerate(train_loader):
        # for seq_true in train_dataset:
            optimizer.zero_grad()
            seq_true = seq_true.to(config.device)
            # print(seq_true)
            seq_pred = model(seq_true)
            loss = criterion(seq_pred, seq_true)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        val_losses = []
        model = model.eval()
        with torch.no_grad():
            for idx, seq_true in enumerate(val_loader):
            # for seq_true in val_dataset:
                seq_true = seq_true.to(config.device)
                seq_pred = model(seq_true)
                loss = criterion(seq_pred, seq_true)
                print(loss)
                val_losses.append(loss.item())
        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        history['train'].append(train_loss)
        history['val'].append(val_loss)
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model, config.MODEL_PATH)
        print(f'Epoch {epoch}: train loss {train_loss} val loss {val_loss}')
    model.load_state_dict(best_model_wts)
    return model.eval(), history

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
    print('loading data...')
    train_dataset = MyDataset(path=config.after_train_dataset_path)
    train_set, seq_len, n_features = train_dataset.create_dataset(train_dataset)
    val_set, _, _ = train_dataset.create_dataset(train_dataset, "val")
    print('load data complete:')
    print(f"train_dataset_len={len(train_dataset)}, seq_len={seq_len}, n_features={n_features}") # [82736, 200, 2]
    

    batch_size = config.batch_size
    train_loader = DataLoader(train_set, batch_size=batch_size, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, drop_last=True)

    model = model.RecurrentAutoencoder(seq_len, n_features, 128) 

    model = model.to(config.device)
    model, history = train_model(
        model,
        train_loader,
        val_loader,
        n_epochs=config.n_epochs
    )
    print("history=")
    print(history)
    history_data = pd.DataFrame(data=history)
    # PATH为导出文件的路径和文件名
    history_data.to_csv(config.loss_dataset_path, index=False)


