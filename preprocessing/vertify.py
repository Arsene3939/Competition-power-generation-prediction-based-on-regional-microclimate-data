import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score

# 1. 讀取數據
df = pd.read_csv("train_split.csv")

df['DateTime'] = pd.to_datetime(df['DateTime'])
df['date'] = df['DateTime'].dt.date
df['time'] = df['DateTime'].dt.time

# 2. 轉換日期欄位為 datetime 格式 (假設日期欄位名為 'date')
df['date'] = pd.to_datetime(df['date'])

# 3. 提取日期列表
unique_dates = df['date'].dt.date.unique()

# 4. 按比例將日期分為訓練集和驗證集
train_dates, val_dates = train_test_split(unique_dates, test_size=0.1, random_state=42)

# 5. 根據日期選擇對應的訓練集和驗證集資料
train_data = df[df['date'].dt.date.isin(train_dates)]
val_data = df[df['date'].dt.date.isin(val_dates)]

train_data = train_data.drop(['date','time'],axis=1)
val_data = val_data.drop(['date','time'],axis=1)

# 6. 檢查資料集大小
print(f"訓練資料大小: {len(train_data)}")
print(f"驗證資料大小: {len(val_data)}")

# 7. 儲存資料集 (可選)
train_data.to_csv("train_data.csv", index=False)
val_data.to_csv("val_data.csv", index=False)

print("訓練資料與驗證資料已儲存為 train_data.csv 和 val_data.csv")