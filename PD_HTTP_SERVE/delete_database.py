from datareceiver.mysql_connect import DatabaseConnection

with DatabaseConnection() as db_conn:  # 连接数据库
    if db_conn.is_connected():
        cursor = db_conn.cursor()
        # 删除数据库指定表
        table_names = [
            "data_info",
            "head_file_info",
            "hf_prpd_info",
            "hf_prpd_sampledata",
            "hf_prps_info",
            "hf_prps_sampledata",
            "hf_pulse_waveform_info",
            "hf_pulse_waveform_sampledata",
            "us_features_info",
            "us_phase_info",
            "us_phase_sampledata",
            "us_pulse_map_info",
            "us_pulse_map_sampledata",
            "us_waveform_map_info",
            "us_waveform_map_sampledata",
            "tev_prpd_info",
            "tev_prps_info", 
            "tev_voltage_info"
        ]

        delect_table_sql = "DROP TABLE IF EXISTS {}"
        for table_name in table_names:
            cursor.execute(delect_table_sql.format(table_name))
            print("删除表：{}".format(table_name))