import pandas as pd
from src.data_loader import load_raw_data

def test_load_raw_data():
    df = load_raw_data()
    # 是否是DataFrame
    assert isinstance(df, pd.DataFrame)
    # 检查关键列是否存在
    required_columns = ["Warehouse_block", "Mode_of_Shipment", "Product_importance", "Reached.on.Time_Y.N"]
    for col in required_columns:
        assert col in df.columns
    # 数据不能为空
    assert len(df) > 0

  # 输入：无（内部读取固定路径）
  # 输出：DataFrame，列名和行数预期不为空
  # 测：类型是不是 DataFrame
  # 测：是否包含必须列（如标签 Reached.on.Time_Y.N）
  # 测：有没有空数据（数量>0）