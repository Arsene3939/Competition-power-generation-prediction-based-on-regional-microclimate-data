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

DEBUG = False

# 检查 CUDA 是否可用
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# 读取数据
train_df = pd.read_csv('dataset/train.csv')
val_df = pd.read_csv('dataset/validation.csv')
train_df = process_time_features(train_df)
val_df = process_time_features(val_df)

# 数据按日期和地点分组
train_df['Date'] = train_df['DateTime'].dt.date
val_df['Date'] = val_df['DateTime'].dt.date
train_df['Location'] = train_df.apply(get_location, axis=1)
val_df['Location'] = val_df.apply(get_location, axis=1)

feature_columns = ['Temperature(°C)', 'Humidity(%)', 'Sunlight(Lux)', 'Time_sin', 'Time_cos'] + [f'LocationCode_{i}' for i in range(1, 18)]
train_samples = create_samples(train_df, feature_columns)
val_samples = create_samples(val_df, feature_columns)

max_input_len = 150  # 对应2.5小时，每分钟一个数据点
train_inputs = [s[0] for s in train_samples]
train_targets = [s[1] for s in train_samples]
train_inputs_padded = pad_sequences(train_inputs, max_input_len)
train_targets = np.array(train_targets)
val_inputs = [s[0] for s in val_samples]
val_targets = [s[1] for s in val_samples]
val_inputs_padded = pad_sequences(val_inputs, max_input_len)
val_targets = np.array(val_targets)

if DEBUG:
    # 打印输入和目标数组的形状
    print(f"Train inputs shape: {train_inputs_padded.shape}")
    print(f"Train targets shape: {train_targets.shape}")

# 创建 DataLoader
train_dataset = PowerDataset(train_inputs_padded, train_targets)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
val_dataset = PowerDataset(val_inputs_padded, val_targets)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

# 创建 LSTM 模型
input_size = len(feature_columns)
hidden_size = 128  # 可调整
num_layers = 2     # 可调整
output_size = 48   # 预测48个值
model = PowerLSTM(input_size, hidden_size, num_layers, output_size)
model.to(device)  # 将模型移动到 GPU
print("LSTM model is created.")

# 创建损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train(model, train_loader):
    model.train()
    total_loss = 0
    for inputs, targets, lengths in train_loader:
        # 将数据移动到 GPU
        inputs = inputs.to(device)
        targets = targets.to(device)
        lengths = lengths.to(device)

        optimizer.zero_grad()
        outputs = model(inputs, lengths)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    return avg_loss

def evaluate(model, val_loader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for inputs, targets, lengths in val_loader:
            # 将数据移动到 GPU
            inputs = inputs.to(device)
            targets = targets.to(device)
            lengths = lengths.to(device)

            outputs = model(inputs, lengths)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
    avg_loss = total_loss / len(val_loader)
    return avg_loss

# Early stop 
best_val_loss = float('inf')
patience = 100
counter = 0

# Train and validation loop
num_epochs = 1000
best_model_path = 'best.pth'
last_model_path = 'last.pth'
for epoch in range(num_epochs):
    train_loss = train(model, train_loader)
    val_loss = evaluate(model, val_loader)
    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')
    
    # 检查是否改进
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), best_model_path)  # 保存最佳模型
        counter = 0  # 重置计数器
    else:
        counter += 1

    # Early stopping 条件
    if counter >= patience:
        print("Early stopping triggered. Stopping training.")
        break

# 保存模型
torch.save(model.state_dict(), 'weights.pth')
print("Model weights have been saved to 'weights.pth'.")
