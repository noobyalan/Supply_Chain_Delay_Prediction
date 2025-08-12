from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from src.data_loader import load_orders_from_db  # 改成从PG读取原始数据
from src.feature_engineering import process_features
from src.utils import tune_hyperparameters
from src.config import PG_CONFIG
import pandas as pd
import joblib
import os
from datetime import datetime
import json
import mlflow
import subprocess
import socket

PROCESSED_DIR = "data/processed"


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
    db = "mlflow_db"  # MLflow专用数据库

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
    # 启动 MLflow server（如已运行则跳过）
    start_mlflow_server()

    # 告诉 MLflow 用 server 作为 tracking backend
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("supply_chain_delay")

    # 1. 数据加载 + 特征工程
    X_path = os.path.join(PROCESSED_DIR, "X_processed.csv")
    y_path = os.path.join(PROCESSED_DIR, "y.csv")
    encoders_path = os.path.join(PROCESSED_DIR, "encoders.pkl")

    if os.path.exists(X_path) and os.path.exists(y_path) and os.path.exists(encoders_path):
        print(f"[INFO] 检测到已处理数据，直接加载: {PROCESSED_DIR}")
        X = pd.read_csv(X_path)
        y = pd.read_csv(y_path).iloc[:, 0]  # 读取 Series 格式
        encoders = joblib.load(encoders_path)
    else:
        print("[INFO] 未检测到 processed 数据，开始从数据库加载原始数据并处理...")
        df = load_orders_from_db()
        X, y, encoders = process_features(df, save_dir=PROCESSED_DIR)

    # 切分数据
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 2. 参数范围
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [4, 7],
        'learning_rate': [0.01, 0.14]
    }

    # 3. 调参 + 评估
    best_model, best_params, metrics = tune_hyperparameters(
        model=XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'),
        param_grid=param_grid,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        scoring='roc_auc',
        cv=3
    )

    print("\n模型评估结果：")
    for k, v in metrics.items():
        print(f"{k.capitalize():<9}: {v:.4f}")

    # 4. 保存模型 / 编码器 / 指标 （加时间戳版本）
    os.makedirs("models", exist_ok=True)
    os.makedirs("reports", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    joblib.dump(best_model, f"models/delay_predictor_{timestamp}.pkl")
    joblib.dump(encoders, f"models/feature_encoders_{timestamp}.pkl")
    with open(f"reports/metrics_{timestamp}.json", "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"\n[INFO] 模型与编码器已保存，版本: {timestamp}")
    print(f"[INFO] 最佳参数: {best_params}")

    # 5. MLflow 记录到 server
    with mlflow.start_run():
        mlflow.log_params(best_params)
        mlflow.log_metrics(metrics)
        mlflow.xgboost.log_model(best_model, artifact_path="model")

    print("[INFO] 训练结果已记录到 MLflow Server")


if __name__ == "__main__":
    main()