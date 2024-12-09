import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def process_time_features(df):
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    df['Hour'] = df['DateTime'].dt.hour
    df['Minute'] = df['DateTime'].dt.minute
    # 將小時和分鐘轉換為小時的小數表示
    df['Time'] = df['Hour'] + df['Minute'] / 60.0
    # 週期性時間特徵
    df['Time_sin'] = np.sin(2 * np.pi * df['Time'] / 24)
    df['Time_cos'] = np.cos(2 * np.pi * df['Time'] / 24)
    return df

def get_location(row):
    for i in range(1, 18):
        if row[f'LocationCode_{i}'] == 1:
            return i
    return np.nan

def create_samples(df, feature_columns):
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

        # 输出序列
        output_df = group_df[
            (group_df['DateTime'].dt.time >= pd.to_datetime('09:00:00').time()) &
            (group_df['DateTime'].dt.time <= pd.to_datetime('16:59:59').time())
        ]
        if len(output_df) == 0:
            continue  # 跳过没有输出数据的情况

        # 确保 'Power' 列为数值类型
        output_df.loc[:, 'Power(mW)'] = pd.to_numeric(output_df['Power(mW)'], errors='coerce')

        # 设置索引并确保是 DatetimeIndex
        output_df = output_df.set_index('DateTime')
        output_df.index = pd.to_datetime(output_df.index)

        # 将 date 转换为字符串
        date_str = date.strftime('%Y-%m-%d')

        # 使用 '10min' 作为频率
        time_index = pd.date_range(start=pd.to_datetime(f"{date_str} 09:00:00"), periods=48, freq='10min')

        # 重采样并对齐时间索引，只选择 'Power' 列
        resampled_output = output_df[['Power(mW)']].resample('10min').mean().reindex(time_index)
        resampled_output.index.name = 'DateTime'
        resampled_output = resampled_output.reset_index()

        # 检查是否有缺失值
        if resampled_output['Power(mW)'].isnull().any():
            continue  # 跳过有缺失值的样本

        input_features = input_df[feature_columns].values
        target = resampled_output['Power(mW)'].values
        samples.append((input_features, target))
    return samples

def pad_sequences(sequences, max_len):
    padded_sequences = []
    for seq in sequences:
        if len(seq) < max_len:
            pad_width = max_len - len(seq)
            pad = np.zeros((pad_width, seq.shape[1]))
            seq = np.vstack((pad, seq))
        elif len(seq) > max_len:
            seq = seq[-max_len:]
        padded_sequences.append(seq)
    return np.array(padded_sequences)

def parse_prediction_code(code):
    code = str(code)
    year = code[0:4]
    month = code[4:6]
    day = code[6:8]
    time = code[8:12]
    device_code = code[12:14]
    return {
        'Year': int(year),
        'Month': int(month),
        'Day': int(day),
        'Time': int(time),
        'DeviceCode': int(device_code),
        'Code': code
    }

def visualize_sample(train_samples, sample_idx):
    input_features, target = train_samples[sample_idx]
    
    # 输入特征可视化
    input_sequence_length = input_features.shape[0]
    time_axis = np.arange(-input_sequence_length, 0)  # 从 -input_sequence_length 到 0
    feature_names = ['Pressure', 'Temperature', 'Humidity', 'Sunlight']
    feature_indices = [0, 1, 2, 3]
    
    plt.figure(figsize=(12, 12))
    for idx, feature_idx in enumerate(feature_indices):
        plt.subplot(len(feature_indices)+1, 1, idx + 1)
        plt.plot(time_axis, input_features[:, feature_idx])
        plt.title(f"Sample {sample_idx} - {feature_names[idx]}")
        plt.xlabel('Time (minutes before 9:00 AM)')
        plt.ylabel(feature_names[idx])
    
    # 目标值可视化
    target_time_axis = pd.date_range(start='09:00', periods=48, freq='10T')
    plt.subplot(len(feature_indices)+1, 1, len(feature_indices)+1)
    plt.plot(target_time_axis, target)
    plt.title(f"Sample {sample_idx} - Power Output")
    plt.xlabel('Time')
    plt.ylabel('Power (mW)')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()