import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import *
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from model import PowerLSTM
from dataset import PowerDataset, collate_fn

# 检查 CUDA 是否可用
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# 读取测试数据
test_df = pd.read_csv('dataset/test.csv')
test_df = process_time_features(test_df)
test_df['Date'] = test_df['DateTime'].dt.date
test_df['Location'] = test_df.apply(get_location, axis=1)

# 定义特征列
feature_columns = ['Temperature(°C)', 'Humidity(%)', 'Sunlight(Lux)', 'Time_sin', 'Time_cos'] + [f'LocationCode_{i}' for i in range(1, 18)]

max_input_len = 150
output_size = 48

# 修改 create_samples 函数，使其返回日期和设备代号
def create_test_samples(df, feature_columns):
    samples = []
    group = df.groupby(['Date', 'Location'])
    for (date, loc), group_df in group:
        # 输入序列
        input_df = group_df[
            (group_df['DateTime'].dt.time >= pd.to_datetime('06:30:00').time()) &
            (group_df['DateTime'].dt.time <= pd.to_datetime('08:59:59').time())
        ]
        if len(input_df) == 0:
            continue  # 跳过没有输入数据的情况

        input_features = input_df[feature_columns].values

        # 填充或截断输入特征
        if len(input_features) > max_input_len:
            input_features = input_features[-max_input_len:]
        else:
            pad_width = max_input_len - len(input_features)
            pad = np.zeros((pad_width, input_features.shape[1]))
            input_features = np.vstack((pad, input_features))

        sample = {
            'input_features': input_features,
            'date': date,
            'device_code': int(loc)
        }

        samples.append(sample)

    return samples

test_samples = create_test_samples(test_df, feature_columns)

# 加载模型
input_size = len(feature_columns)
hidden_size = 128
num_layers = 2
output_size = 48

model = PowerLSTM(input_size, hidden_size, num_layers, output_size)
model.load_state_dict(torch.load('weights.pth', map_location=device))
model.to(device)
model.eval()
print("Model loaded and set to evaluation mode.")

# 进行预测
predictions = []

with torch.no_grad():
    for sample in test_samples:
        input_features = sample['input_features']
        date = sample['date']
        device_code = sample['device_code']

        # 转换为张量并移动到设备
        inputs = torch.tensor(input_features, dtype=torch.float32).unsqueeze(0).to(device)  # (1, seq_len, input_size)
        lengths = torch.tensor([max_input_len], dtype=torch.long).to(device)

        # 模型预测
        outputs = model(inputs, lengths)
        outputs = outputs.cpu().numpy().flatten()  # (48,)

        # 存储预测结果
        sample['predictions'] = outputs
        predictions.append(sample)

print("Predictions completed.")

# 读取 upload.csv 并解析序号
upload_df = pd.read_csv('dataset/upload.csv')

# 解析序号
upload_df_parsed = upload_df['序號'].apply(parse_prediction_code).apply(pd.Series)

# 创建 (Year, Month, Day, DeviceCode) 到预测结果的映射
prediction_dict = {}
for sample in predictions:
    date = sample['date']
    device_code = sample['device_code']
    year = date.year
    month = date.month
    day = date.day
    key = (year, month, day, device_code)
    prediction_dict[key] = sample['predictions']

# 将预测结果填入 upload_df
predicted_values = []
for index, row in upload_df_parsed.iterrows():
    year = row['Year']
    month = row['Month']
    day = row['Day']
    time = row['Time']
    device_code = row['DeviceCode']

    key = (year, month, day, device_code)
    predictions_for_key = prediction_dict.get(key)
    if predictions_for_key is None:
        # 如果没有对应的预测结果，填充 NaN
        predicted_value = np.nan
    else:
        # 计算在预测数组中的索引
        # 时间从 0900 开始，每10分钟一个点
        time_str = f"{time:04d}"
        hour = int(time_str[:2])
        minute = int(time_str[2:])
        index_in_predictions = ((hour - 9) * 60 + minute) // 10
        if 0 <= index_in_predictions < 48:
            predicted_value = predictions_for_key[index_in_predictions]
        else:
            predicted_value = np.nan

    predicted_values.append(predicted_value)

# 创建提交文件
submit_df = upload_df.copy()
submit_df['答案'] = predicted_values

# 将预测值保留两位小数
submit_df['答案'] = submit_df['答案'].round(2)

# 保存为 submit.csv
submit_df.to_csv('submit.csv', index=False)
print("Submission file 'submit.csv' has been created.")
