from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.data_loader import load_orders_from_db
from src.utils import tune_hyperparameters
from src.config import PG_CONFIG
import mlflow
import pandas as pd
import os
import subprocess
import socket
from datetime import datetime

def start_mlflow_server():
    """如果 MLflow server 没有运行则自动启动"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        if sock.connect_ex(("127.0.0.1", 5000)) == 0:
            print("[INFO] MLflow server 已在运行，跳过启动")
            return
        sock.close()
    except Exception as e:
        print(f"[WARN] 检查 MLflow server 失败: {e}")

    print("[INFO] 正在启动 MLflow server ...")
    artifact_dir = os.path.abspath("mlruns_server")
    os.makedirs(artifact_dir, exist_ok=True)

    user = PG_CONFIG["user"]
    password = PG_CONFIG["password"]
    host = PG_CONFIG["host"]
    port = PG_CONFIG["port"]
    db = "mlflow_db"
    
    cmd = [
        "mlflow", "server",
        "--backend-store-uri", f"postgresql://{user}:{password}@{host}:{port}/{db}",
        "--default-artifact-root", artifact_dir,
        "--host", "0.0.0.0",
        "--port", "5000"
    ]
    subprocess.Popen(cmd)
    print(f"[INFO] MLflow server 已启动: Backend=postgresql://{user}@{host}/{db}, Artifact Root={artifact_dir}")


def main():
    # 启动 MLflow server
    start_mlflow_server()
    
    # 设置 MLflow Tracking
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("supply_chain_delay_pipeline")

    # === 1. 加载原始数据 ===
    print("[INFO] 从数据库加载原始数据...")
    df = load_orders_from_db()

    target_col = "Reached.on.Time_Y.N"
    y = df[target_col]
    X = df.drop(columns=[target_col, "ID"])  # ID 不作为特征

    # === 2. 定义特征列 ===
    numeric_features = ["Customer_care_calls", "Customer_rating", "Cost_of_the_Product",
                        "Prior_purchases", "Discount_offered", "Weight_in_gms"]
    categorical_features = ["Warehouse_block", "Mode_of_Shipment", "Product_importance", "Gender"]

    # === 3. 构造预处理器 ===
    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
    categorical_transformer = Pipeline(steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )

    # === 4. 构造 Pipeline（预处理+分类器） ===
    clf = XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", clf)
    ])

    # === 5. 切分数据 ===
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # === 6. 调参搜索空间（注意key写法: classifier__参数名） ===
    param_grid = {
        "classifier__n_estimators": [100, 200],
        "classifier__max_depth": [4, 7],
        "classifier__learning_rate": [0.01, 0.14]
    }

    # === 7. 使用 utils 中的 tune_hyperparameters 调参 ===
    best_pipeline, best_params, metrics = tune_hyperparameters(
        model=pipeline,
        param_grid=param_grid,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        scoring='roc_auc',
        cv=3
    )

    # === 8. 保存到 MLflow Model Registry ===
    with mlflow.start_run():
        mlflow.log_params(best_params)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(
            sk_model=best_pipeline,
            artifact_path="model",
            registered_model_name="suppy_chain_predict"
        )

    print("[INFO] 最佳 Pipeline 已保存到 MLflow。请到 UI 手动设置 production alias。")


if __name__ == "__main__":
    main()