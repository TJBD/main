# 从数据库中提取数据
# 1. 从数据库中提取数据
from .mysql_connect import DatabaseConnection
import json
import struct
import pickle
from sklearn.cluster import KMeans
us_features = None
us_phase = None
us_pulse_map = None
us_waveform_map = None

# tables_name = [
#     "us_features_info",
#     "us_phase_sampledata",
#     "us_pulse_map_sampledata",
#     "us_waveform_map_sampledata",
# ]

# with DatabaseConnection() as db_conn:  # 连接数据库
#     if db_conn.is_connected():
#         cursor = db_conn.cursor()
#         for table in tables_name:
#             query = f"SELECT * FROM {table} WHERE file_name = '30_20230512085500000' "
#             cursor.execute(query)
#             if table == "us_features_info":
#                 us_features = cursor.fetchall()
#             elif table == "us_phase_sampledata":
#                 us_phase = cursor.fetchall()
#             elif table == "us_pulse_map_sampledata":
#                 us_pulse_map = cursor.fetchall()
#             elif table == "us_waveform_map_sampledata":
#                 us_waveform_map = cursor.fetchall()
#             else:
#                 print("error")

# # 关闭连接
# db_conn.close()

table_name = "us_features_info"
# 列名
col_names = ["signal_peak", "signal_valid", "frequency1_corr", "frequency2_corr"]
# 提取表中某些列的数据
with DatabaseConnection() as db_conn:  # 连接数据库
    if db_conn.is_connected():
        cursor = db_conn.cursor()
        # 创建 SQL 查询
        query = f"SELECT {', '.join(col_names)} FROM {table_name}"
        # 执行 SQL 查询
        cursor.execute(query)
        # 获取所有结果
        datas = cursor.fetchall()
        # # 将结果保存成.dat文件
        # with open('data0000.dat', 'wb') as f:
        #     pickle.dump(results, f)

# 使用KMeans进行聚类
# kmeans = KMeans(n_clusters=4)
# kmeans.fit(datas)
# labels = kmeans.labels_

# 打印聚类结果
# print(labels)

# 保存datas和labels
# with open('X.dat', 'wb') as f:
#     pickle.dump(datas, f)
# with open('y.dat', 'wb') as f:
#     pickle.dump(labels, f)

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load datas and labels
with open('X.dat', 'rb') as f:
    datas = pickle.load(f)
with open('y.dat', 'rb') as f:
    labels = pickle.load(f)

# 数据标准化处理
scaler = StandardScaler().fit(datas)
datas = scaler.transform(datas)

# 将numpy数组转换为torch张量
# Convert datas and labels to torch tensors
datas = torch.tensor(datas, dtype=torch.float32)
labels = torch.tensor(labels, dtype=torch.long)

# Define the neural network model
class NeuralNetwork(nn.Module):
    def __init__(self, input_size):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 4)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.softmax(self.fc3(x), dim=1)
        return x

# Create an instance of the neural network model
model = NeuralNetwork(datas.shape[1])

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
for epoch in range(100):
    optimizer.zero_grad()
    outputs = model(datas)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

# Save the trained model and scaler
torch.save(model.state_dict(), 'model_weights.pth')
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)


# Use the trained model to predict labels for new datas
predicted_labels = model(datas).argmax(dim=1)
# 最大概率概率值
predicted_prob = model(datas).max(dim=1).values
# 输出预测标签和对应标签的概率值
print(predicted_prob)
print(predicted_labels)

