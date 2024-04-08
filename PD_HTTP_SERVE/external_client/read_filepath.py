import requests
import argparse
import os
import json
# 服务器的URL
# URL = "http://127.0.0.1:8080/pddetect/receive-data/"

directory_path = "H:\湖高II线高抗A相\data\超声波4\min"
# directory_path = "H:\湖高II线高抗A相\data\超声波2\min"
# List to hold all the absolute paths
file_paths = []

# 遍历文件夹
for root, dirs, files in os.walk(directory_path):
    # 确保目录按日期排序
    dirs.sort()
    # 确保文件按名字排序
    files.sort()
    for file in files:
        if file.endswith('.dat'):
            # 构造完整的文件路径并添加到列表中
            full_path = os.path.join(root, file)
            file_paths.append(full_path)

# 移除列表中可能的重复项
dat_file_paths = list(dict.fromkeys(file_paths))

# 写入排序后的唯一文件路径到文本文件
output_txt_path = 'sorted_paths_AE4.txt'
with open(output_txt_path, 'w') as file:
    for path in dat_file_paths:
        file.write(f"{path}\n")
        # 打印文件路径
        print(path)

# 输出文本文件的路径
print(f"The sorted .dat file paths have been written to {output_txt_path}")