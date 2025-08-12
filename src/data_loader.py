import pandas as pd
import os
from src.config import DATA_PATH_RAW, PG_CONFIG
from src.database_io import read_table_from_postgres

def load_raw_csv(path=DATA_PATH_RAW, sep=","):
    """读取本地 CSV（用于初始化数据库）"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"数据文件不存在: {path}")
    df = pd.read_csv(path, sep=sep)
    df.columns = df.columns.str.strip()  # 清除列名首尾空格
    df = df.drop_duplicates()
    df = df.dropna()
    return df

def load_orders_from_db(where=None):
    """从 PostgreSQL 读取订单表"""
    return read_table_from_postgres("orders", where=where, **PG_CONFIG)
