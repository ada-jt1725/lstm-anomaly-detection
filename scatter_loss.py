import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
import torch.nn as nn
import config
import  matplotlib.pyplot as plt
import model
from sklearn import svm
from sklearn.metrics import accuracy_score
from pre_with_model import create_dataset, MyDataset, predict
plt.rcParams['font.family'] = 'Heiti TC'
anomaly_dataset = MyDataset(path=config.anomaly_dataset_path_ori)

anomaly_dataset, _ = train_test_split(
    anomaly_dataset,
    test_size=0.03,
    shuffle=False
)
anomaly_train_df, _, _ = create_dataset(anomaly_dataset)

model = torch.load('model.pth',map_location=torch.device('cpu'))
model = model.to(config.device)

_, pred_losses = predict(model, anomaly_train_df)

print(pred_losses)



df = pd.read_csv(config.anomaly_dataset_path_ori)
d = pd.DataFrame()
d["loss"] = pred_losses
d["label"] = df["label"]
plt.figure()

# 获取样本个数
sample_count = len(d['label'])

# 对于label为3的点，使用红色
plt.scatter([i for i in range(sample_count) if d['label'][i] == 3], 
            [d['loss'][i] for i in range(sample_count) if d['label'][i] == 3], 
            color='red', label='3级')
# # 对于label为3的点，使用红色
# plt.scatter([i for i in range(sample_count) if d['label'][i] == 2], 
#             [d['loss'][i] for i in range(sample_count) if d['label'][i] == 2], 
#             color='blue', label='2级')
# 对于label不为3的点，使用蓝色
plt.scatter([i for i in range(sample_count) if d['label'][i] == 1], 
            [d['loss'][i] for i in range(sample_count) if d['label'][i] in (1,2)], 
            color='green', label='1级')
plt.legend()
plt.show()