import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

train = pd.read_csv('combined.csv')
df = train

# 設定感測器的最大值
sensor_max = 117758.2

# 分離正常數據與飽和值
normal_data = df[df['Sunlight(Lux)'] < sensor_max]  # 光照度小於最大值的資料
saturated_data = df[df['Sunlight(Lux)'] == sensor_max]  # 光照度等於最大值的資料

# 迴歸分析：使用正常數據擬合 "Sunlight(Lux)" 和 "power" 的關係
X_normal = normal_data['Sunlight(Lux)']  # 特徵：光照度
y_normal = normal_data[['Power(mW)']]  # 標籤：發電量

# 使用多項式回歸進行擬合 (選擇多項式的次數根據資料特性調整，這裡假設為2次)
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(y_normal)

model = LinearRegression(positive=True)
model.fit(X_poly, X_normal)

# 修正飽和值：基於飽和值的發電量預測光照度
saturated_power = saturated_data['Power(mW)']
print(saturated_data.shape)
saturated_X_poly = poly.fit_transform(saturated_power.values.reshape(-1, 1))
corrected_illumination = model.predict(saturated_X_poly)
# 更新飽和值
df.loc[df['Sunlight(Lux)'] == sensor_max, 'Sunlight(Lux)'] = corrected_illumination

# 輸出修正後的資料
df.to_csv("normalize.csv", index=False)