import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import sys
import os
CURRENT_DIR = os.path.split(os.path.abspath(__file__))[0]  # 当前目录
config_path = CURRENT_DIR.rsplit('/', 1)[0]  # 上三级目录
sys.path.append(config_path)
import config
import  matplotlib.pyplot as plt
from dataset import MyDataset
from anomaly_detection.pre_with_model import predict
import model

plt.rcParams['font.family'] = 'Heiti TC'
anomaly_dataset = MyDataset(path=config.anomaly_dataset_path_ori)
anomaly_train_df, _, _ = anomaly_dataset.create_dataset()

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

# 对于label不为3的点，使用蓝色
plt.scatter([i for i in range(sample_count) if d['label'][i] == 1], 
            [d['loss'][i] for i in range(sample_count) if d['label'][i] in (1,2)], 
            color='green', label='1级')
plt.legend()
plt.show()