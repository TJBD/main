import binascii
import struct
def get_data_storage_method(t):
    # 定义t值与数据存储方式和单位大小的映射
    storage_methods = {
        2: ("uint8", 1),
        3: ("int16", 2),
        4: ("int32", 4),
        5: ("int64", 8),
        6: ("float", 4),
        7: ("doulbe", 8),
    }
    d, k = storage_methods[int.from_bytes(t, "little")]
    return d, k

# 幅值单位编码
def get_data_unit(byte_data):
    # 定义x值与幅值单位类型的映射
    units = {
        "0x01": "dB",
        "0x02": "dBm",
        "0x03": "dBmV", 
        "0x04": "dBuV",
        "0x05": "V",
        "0x06": "mV",
        "0x07": "uV",
        "0x08": "%",
        "0x09": "A",
        "0x0A": "mA",
        "0x0B": "uA",
        "0x0C": "Ω",
        "0x0D": "mΩ",
        "0x0E": "uΩ",
        "0x0F": "m/s²",
        "0x10": "mm",
        "0x11": "℃",
        "0x12": "℉",
        "0x13": "Pa",
        "0x14": "C",
        "0x15": "mC",
        "0x16": "uC",
        "0x17": "nC",
        "0x18": "pC",
        "0xEF": "ns"
    }
    hex_str = binascii.hexlify(byte_data).decode()
    units = units['0x'+hex_str]
    return units

def parse_data(variable_part, k, d):
    parsed_data = []

    if d == "float":
        parsed_data = [
            struct.unpack("<f", variable_part[i : i + k])[0]
            for i in range(0, len(variable_part), k)
        ]
    elif d in ["uint8", "int16", "int32", "int64"]:
        parsed_data = [
            int.from_bytes(variable_part[i : i + k], "little")
            for i in range(0, len(variable_part), k)
        ]
    elif d == "double":
        parsed_data = [
            struct.unpack("<d", variable_part[i : i + k])[0]
            for i in range(0, len(variable_part), k)
        ]
    else:
        raise ValueError("Unknown data type")
    return parsed_data


def create_hf_prpd_sampledata_table(connection, table_name, m):
    columns_sql = ", ".join([f"`{i}` INT" for i in range(1, m + 1)])
    create_table_sql = f"CREATE TABLE IF NOT EXISTS `{table_name}` (id INT AUTO_INCREMENT PRIMARY KEY, {columns_sql});"
    with connection.cursor() as cursor:
        cursor.execute(create_table_sql)
    connection.commit()


def insert_hf_prpd_sampledata_to_db(connection, table_name, columns_sql, parsed_data):
    cursor = connection.cursor()
    insert_query = f"INSERT INTO `{table_name}` ({columns_sql}) VALUES (%s);"
    cursor.executemany(insert_query, [(val,) for val in parsed_data])
    connection.commit()
