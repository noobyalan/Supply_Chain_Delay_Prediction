import numpy as np
import pandas as pd
from src.data_loader import load_raw_data
from src.feature_engineering import process_features

def test_process_features():
    df = load_raw_data()
    X, y, encoders = process_features(df)

    # 1. 类型检查
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)

    # 2. 数据行数匹配
    assert len(X) == len(y)

    # 3. 标签应该是二分类
    assert y.nunique() == 2

    # 4. One-Hot 编码列的值应当是 0 或 1
    ohe_columns = [col for col in X.columns if any(base in col for base in encoders["one_hot_cols"])]
    if ohe_columns:  # 有OneHot列时
        assert set(np.unique(X[ohe_columns].values)) <= {0, 1}

    # 5. Ordinal 编码列的值范围正确
    for col in encoders["ordinal_cols"]:
        unique_vals = set(X[col].unique())
        assert unique_vals <= {0, 1, 2}  # low, medium, high

    # 6. 没有丢失OneHot和Ordinal列
    assert X.shape[1] == 16


 # 输入：DataFrame 原始数据
 # 输出：加工后的 X_processed、y、encoders
 # 测：X_processed 是否是 DataFrame，长度与 y 匹配
 # 测：编码后的特征列保持一致且无缺失
 # 测：标签列的取值范围（0/1）