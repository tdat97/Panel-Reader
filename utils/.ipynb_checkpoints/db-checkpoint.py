from utils.logger import logger

import pymysql
import json

DB_INFO_PATH = "./source/db_info.json"

class DBManager():
    def __init__(self):
        info_dic = self.load_info()
        self.connection = pymysql.connect(**info_dic, charset='utf8', autocommit=True, cursorclass=pymysql.cursors.Cursor)
        self.cursor = self.connection.cursor()
        
    def load_info(self):
        with open(DB_INFO_PATH, 'r', encoding='utf-8') as f:
            info_dic = json.load(f)
        return info_dic
    
    def upload_data(self, table_name, **kwargs):
        keys_str = ", ".join(kwargs.keys())
        values = list(kwargs.values())
        func = lambda x:f"'{x}'" if type(x) == str else f"{x}"
        values = list(map(func, values))
        values_str = ", ".join(values)
        
        sql = f"INSERT INTO {table_name}({keys_str}) VALUES ({values_str});"
        logger.debug(sql)
        self.cursor.execute(sql)
    
    def close(self):
        self.connection.close()