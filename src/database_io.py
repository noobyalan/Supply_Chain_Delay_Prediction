# src/database_io.py
import pandas as pd
from src.utils import get_postgres_engine

def write_csv_to_postgres(csv_path, table_name, user, password, host, port, db, if_exists="replace"):
    """把本地 CSV 写入 PostgreSQL"""
    df = pd.read_csv(csv_path)
    engine = get_postgres_engine(user, password, host, port, db)
    df.to_sql(table_name, engine, if_exists=if_exists, index=False)
    print(f"[INFO] {len(df)} 行数据已写入 {table_name}")

def read_table_from_postgres(table_name, user, password, host, port, db, where=None):
    """从 PostgreSQL 读表，返回 DataFrame"""
    engine = get_postgres_engine(user, password, host, port, db)
    query = f"SELECT * FROM {table_name} " + (f"WHERE {where}" if where else "")
    return pd.read_sql(query, engine)

def exec_sql(query, user, password, host, port, db):
    """执行任意SQL（增删改查都可）"""
    engine = get_postgres_engine(user, password, host, port, db)
    with engine.connect() as conn:
        conn.execute(query)
        conn.commit()