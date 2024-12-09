import pandas as pd

data = pd.read_csv('normalize.csv')
# 設定篩選的時間範圍
data['DateTime'] = pd.to_datetime(data['DateTime'])
data['date'] = data['DateTime'].dt.date
data['time'] = data['DateTime'].dt.time

start_time = pd.to_datetime("09:00:00").time()
end_time = pd.to_datetime("17:00:00").time()

# 篩選 9:00 - 17:00 的資料
filtered_data = data[(data['time'] >= start_time) & (data['time'] <= end_time)]

# 找出每個日期和地點是否包含所需的時間段
attr = [f'LocationCode_{i}' for i in range(1,18)]
attr.append('date')
print(attr)
grouped = filtered_data.groupby(attr)

# 設定 train 和 test
train_dates = set(grouped.groups.keys())
train_data = data[data[attr].apply(tuple, axis=1).isin(train_dates)]
test_data = data[~data[attr].apply(tuple, axis=1).isin(train_dates)]

train_data = train_data.drop(['date','time'],axis=1)
test_data = test_data.drop(['date','time'],axis=1)

# 儲存到不同的 CSV
train_data.to_csv('train_split.csv', index=False)
test_data.to_csv('test_split.csv', index=False)

print('done')