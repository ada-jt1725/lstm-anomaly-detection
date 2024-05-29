import pandas as pd

path = "anomaly_dataset_binary.csv"

df = pd.read_csv(path)
df = df.drop(columns=["流入速率(Mbps)0","流出速率(Mbps)0"])
df.to_csv('anomaly_dataset_binary_new.csv')