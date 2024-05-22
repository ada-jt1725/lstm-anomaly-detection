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
from tqdm import tqdm
import faulthandler
# 在import之后直接添加以下启用代码即可
# faulthandler.enable()
# 后边正常写你的代码

logging.basicConfig(filename='train.log', level=logging.DEBUG)


def make_signal_list():
    signal_list = list()
    for col in config.signal_columns:
        for i in range(-config.slide_range, 1):
            signal_list.append(col + str(i))
    return signal_list

def create_dataset(df):
    array = np.array(df)
    # print(f"array.shape={array.shape}") # (82736, 400)
    # 将数组分割为两个部分s
    array1 = array[:, :config.slide_range+1]  # 选择所有行和前200列
    array2 = array[:, config.slide_range+1:]  # 选择所有行和后200列
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
    train_df, val_df = train_test_split(
        train_dataset,
        test_size=0.1
    )
    train_dataset, seq_len, n_features = create_dataset(train_df)
    val_dataset, _, _ = create_dataset(val_df)
    print('load data complete:')
    print(f"train_dataset_len={len(train_dataset)}, seq_len={seq_len}, n_features={n_features}") # [82736, 200, 2]
    

    batch_size = config.batch_size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, drop_last=True)

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


