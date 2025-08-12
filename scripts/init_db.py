# scripts/init_db.py
from src.data_loader import load_raw_csv
from src.database_io import write_csv_to_postgres
from src.config import DATA_PATH_RAW, PG_CONFIG

if __name__ == "__main__":
    print("[INFO] 开始从 CSV 导入 PostgreSQL ...")

    # 读取 CSV（可选，只是为了查看）
    df = load_raw_csv()
    print(f"[INFO] 本地 CSV 行数: {len(df)}")

    # 写入 PostgreSQL
    write_csv_to_postgres(
        csv_path=DATA_PATH_RAW,
        table_name="orders",
        **PG_CONFIG
    )

    print("[INFO] CSV 导入 PostgreSQL 完成！")