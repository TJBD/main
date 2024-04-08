from .mysql_connect import DatabaseConnection
import torch
import pickle
from datareceiver.nn import NeuralNetwork
import numpy as np
def pd_detect(filename):
    table_name = "us_features_info"
    # 列名
    col_names = ["signal_peak", "signal_valid", "frequency1_corr", "frequency2_corr"]
    # 提取表中某些列的数据
    with DatabaseConnection() as db_conn:  # 连接数据库
        if db_conn.is_connected():
            cursor = db_conn.cursor()
            # 创建 SQL 查询
            query = f"SELECT {', '.join(col_names)} FROM {table_name} WHERE file_name = %s"
            # 执行 SQL 查询
            cursor.execute(query, (filename,))
            # 获取所有结果
            datas = cursor.fetchall()
            # 将datas转成ndarray类型
            datas = np.array(datas)
            print(datas.shape)

    # Load the trained model
    model = NeuralNetwork(datas.shape[1])  # Replace input_size with the actual size
    model.load_state_dict(torch.load('model_weights.pth'))
    model.eval()  # Set the model to evaluation mode

    # Load the scaler
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)

    # Process new data with the scaler and make predictions
    new_data = datas  # Replace with the actual data
    new_data = scaler.transform(new_data)  # Scale the new data
    new_data = torch.tensor(new_data, dtype=torch.float32)  # Convert to torch tensor

    predicted_labels = model(new_data).argmax(dim=1).item()
    print(predicted_labels)
    predicted_prob = model(new_data).max(dim=1).values.item()
    print(predicted_prob)
    counts = 1
    return counts, predicted_labels, predicted_prob
