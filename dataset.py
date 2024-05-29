from torch.utils.data import Dataset, DataLoader
import copy
import config
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import numpy as np
import torch
from sklearn.model_selection import train_test_split


class MyDataset(Dataset):
    def __init__(self, path):
        self.signal_df = pd.read_csv(path)
        self.signal_columns = self.make_signal_list()

    def __len__(self):
        return len(self.signal_df)

    def __getitem__(self, idx):
        row = self.signal_df.loc[idx]
        x = row[self.signal_columns].values.astype(float)
        return x
    def make_signal_list(self):
        signal_list = list()
        for col in config.signal_columns:
            for i in range(-config.slide_range, 0):
                signal_list.append(col + str(i))
        return signal_list
    
    def create_dataset(self, mode="train"):
        test_dataset, val_dataset = train_test_split(
        self,
        test_size=0.1,
        shuffle=False
        )
        if mode == "train":
            array = np.array(test_dataset)
        elif mode == "val":
            array = np.array(val_dataset)
        else:
            print("did not specify a valid mode")
        # print(f"array.shape={array.shape}") # (82736, 400)
        # 将数组分割为两个部分s
        array1 = array[:, :config.slide_range]  # 选择所有行和前300列
        array2 = array[:, config.slide_range:]  # 选择所有行和后300列
        # print(f"array1.shape={array1.shape}, array2.shape={array2.shape}") # (82736, 200), (82736, 200)
        # 将两个部分合并为一个新的数组
        array_new = np.stack((array1, array2), axis=-1)
        dataset = [torch.tensor(s).float() for s in array_new]
        n_seq, seq_len, n_features = torch.stack(dataset).shape
        return dataset, seq_len, n_features