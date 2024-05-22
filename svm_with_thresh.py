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
from sklearn import svm
from sklearn.metrics import accuracy_score
from pre_with_model import create_dataset, MyDataset, predict

anomaly_dataset = MyDataset(path=config.anomaly_dataset_path)

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



df = pd.read_csv(config.anomaly_dataset_path)
df_thresh = pd.DataFrame()
df_thresh["loss"] = pred_losses
df_thresh["label"] = df["label"]

# all
train_df, val_df = train_test_split(df_thresh, test_size=0.2)
X_train = train_df.drop('label', axis=1)
y_train = train_df['label']
X_val = val_df.drop('label', axis=1)
y_val = val_df['label']

clf = svm.SVC(kernel='linear', C=1)

clf.fit(X_train, y_train)

y_pred = clf.predict(X_val)
print("val and pred: \n",f"y_val: \n{y_val}, y_pred: {y_pred}, \nX_val: \n{X_val}")
print("accuracy_score:", accuracy_score(y_val, y_pred))


# df["loss"] = [15.626228332519531, 15.640166282653809, 15.329761505126953, 14.269622802734375, 12.31506633758545, 9.544710159301758, 26.291038513183594, 27.958297729492188, 21.876169204711914, 12.700190544128418, 18.96635627746582, 79.52495574951172, 18.689468383789062, 11.418328285217285, 36.061580657958984, 36.55607604980469, 35.567138671875, 23.547456741333008, 21.830217361450195, 33.31680679321289, 35.59656524658203, 31.908409118652344, 32.71609115600586, 22.04199981689453]

