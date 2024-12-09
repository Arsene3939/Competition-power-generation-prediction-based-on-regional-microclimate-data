import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

file_path = "./36_TrainingData/"  # CSV 檔案所在的資料夾路徑
file_names = [f"L{i}_Train.csv" for i in range(1, 18)]  # 產生 1.csv 到 17.csv 的檔名

file_path2 = "./36_TrainingData_Additional_V2/"  # CSV 檔案所在的資料夾路徑
file_names_add = ["L2_Train_2.csv",
                  "L4_Train_2.csv",
                  "L7_Train_2.csv",
                  "L8_Train_2.csv",
                  "L9_Train_2.csv",
                  "L10_Train_2.csv",
                  "L12_Train_2.csv"
                  ]


dataframes = []
for file_name in file_names:
    df = pd.read_csv(file_path + file_name)  # 讀取 CSV
    dataframes.append(df)

for file_name in file_names_add:
    df = pd.read_csv(file_path2 + file_name)  # 讀取 CSV
    dataframes.append(df)

combined_df = pd.concat(dataframes, ignore_index=True)
combined_df = pd.get_dummies(data = combined_df, columns = ['LocationCode'], dtype=int)
combined_df = combined_df.drop(columns=['WindSpeed(m/s)'])
# combined_df = combined_df.drop(columns=['Pressure(hpa)'])

combined_df.to_csv("combined.csv", index=False)  # 將合併的檔案存為 combined.csv

print("done")
#combined_df = pd.read_csv('combined.csv')


