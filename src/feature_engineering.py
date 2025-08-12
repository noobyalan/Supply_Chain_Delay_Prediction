# src/feature_engineering.py
import os
import pandas as pd
import joblib
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

def process_features(df: pd.DataFrame, save_dir=None):
    """
    对供应链数据进行特征处理
    :param df: 原始 DataFrame
    :param save_dir: 若指定目录，则将处理后的数据和编码器保存下来
    :return: X_processed, y, encoders_dict
    """
    target_col = "Reached.on.Time_Y.N"

    # 分离特征和目标
    X = df.drop(columns=["ID", target_col])
    y = df[target_col]

    # 列划分
    one_hot_cols = ["Warehouse_block", "Mode_of_Shipment"]
    ordinal_cols = ["Product_importance"]
    binary_cols = ["Gender"]
    numeric_cols = [col for col in X.columns if col not in one_hot_cols + ordinal_cols + binary_cols]

    # One-Hot 编码
    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    X_ohe = pd.DataFrame(ohe.fit_transform(X[one_hot_cols]), columns=ohe.get_feature_names_out(one_hot_cols))

    # Ordinal 编码
    ord_encoder = OrdinalEncoder(categories=[["low", "medium", "high"]])
    X_ord = pd.DataFrame(ord_encoder.fit_transform(X[ordinal_cols]), columns=ordinal_cols)

    # 二值编码
    X_bin = X[binary_cols].replace({"M": 0, "F": 1})

    # 数值特征
    X_num = X[numeric_cols]

    # 合并所有特征
    X_processed = pd.concat(
        [X_num.reset_index(drop=True),
         X_bin.reset_index(drop=True),
         X_ord.reset_index(drop=True),
         X_ohe.reset_index(drop=True)],
        axis=1
    )

    # 编码器字典
    encoders = {
        "ohe": ohe,
        "ord": ord_encoder,
        "one_hot_cols": one_hot_cols,
        "ordinal_cols": ordinal_cols,
        "binary_cols": binary_cols,
        "numeric_cols": numeric_cols
    }

    # 如果传入了保存目录，将数据和编码器保存下来
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        X_processed.to_csv(os.path.join(save_dir, "X_processed.csv"), index=False)
        y.to_csv(os.path.join(save_dir, "y.csv"), index=False)
        joblib.dump(encoders, os.path.join(save_dir, "encoders.pkl"))
        print(f"[INFO] 处理结果已存储至 {save_dir}")

    return X_processed, y, encoders

if __name__ == "__main__":
    from src.data_loader import load_orders_from_db
    df = load_orders_from_db()  # 从数据库取原始数据
    process_features(df, save_dir="data/processed")