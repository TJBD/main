import requests
import argparse
import os
import json
# 服务器的URL
URL = "http://127.0.0.1:8000/pddetect/receive-data/"
# URL = "http://47.104.102.28:8000/pddetect/receive-data/"

# 打开文件并读取内容
def send_data(file_path):
    filename = os.path.basename(file_path)
    filetime = extract_datetime_from_filename(filename)
    data_info= {
        "device_type": 4,
        "detection_type": "AA",
        "protocol_ver": "1.0",
        "file_name": filename,
        "file_time": filetime,}
    try:
        # 读取二进制数据           
        with open(file_path, "rb") as file:
            binary_data = file.read()

        # 构造请求
        files = {
            'file': (data_info['file_name'], open(file_path, 'rb')),  # 打开文件
        }
        # 将data_info转换为JSON字符串，并作为表单字段发送
        data = {
            'data_info': json.dumps(data_info),
        }
        response = requests.post(URL, files=files, data=data)
        # 发送POST请求到服务器
        # 检查请求是否成功
        if response.status_code == 200:
            # 解析JSON响应内容并打印
            response_json = response.json()
            print(json.dumps(response_json, indent=4))  # 美化打印JSON数据
        else:
            print(f"Request failed with status code: {response.status_code}")
            print(response.text)

    except FileNotFoundError:
        print(f"The file at {file_path} was not found. Please check the path and try again.")

from datetime import datetime

def extract_datetime_from_filename(filename):
    """
    Extracts the datetime object from a given filename.
    Assumes the filename format is 'prefix_YYYYMMDDHHMMSSfff.dat'
    where 'prefix' can be any string and 'fff' are milliseconds.
    Returns the datetime as a string in the format 'YYYY-MM-DD HH:MM:SS.fff'.
    """
    try:
        # Split the filename and remove the file extension
        datetime_part = filename.split('_')[1].split('.')[0]
        
        # Parse the datetime
        # Assuming the format is "YYYYMMDDHHMMSSfff"
        datetime_obj = datetime.strptime(datetime_part, "%Y%m%d%H%M%S%f")
        
        # Format the datetime as a string
        datetime_str = datetime_obj.strftime("%Y-%m-%d %H:%M:%S.%f")
        
        return datetime_str
    except (IndexError, ValueError):
        # Return None if the filename format is incorrect
        return None

def main():
    # Read the paths from the sorted txt file
    sorted_paths_file = 'sorted_paths_AE4.txt'

    # Initialize a list to store the contents of the .dat files
    dat_files_contents = []

    # Read the file paths from the txt file
    with open(sorted_paths_file, 'r') as file:
        paths = file.readlines()
    
    # Remove newline characters and read each file
    for path in paths:
        print("发送数据{}".format(path))
        path = path.strip()  # Remove the newline character at the end of each line
        # Check if the file exists before trying to open it
        if os.path.exists(path):
            send_data(path)
            # Store the content along with the path in a tuple
            dat_files_contents.append((path))
        else:
            print(f"File does not exist: {path}")

main()