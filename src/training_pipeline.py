# src/training_pipeline.py
from sklearn.model_selection import train_test_split
from src.data_loader import load_orders_from_db
from src.model import create_pipeline
from src.utils import tune_hyperparameters
from src.config import PG_CONFIG
import mlflow
import os, subprocess, socket

def start_mlflow_server():
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        if sock.connect_ex(("127.0.0.1", 5000)) == 0:
            print("[INFO] MLflow server 已在运行，跳过启动")
            return
        sock.close()
    except Exception as e:
        print(f"[WARN] 检查 MLflow server 失败: {e}")

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
    print(f"[INFO] MLflow server 已启动: {db}")

def main():
    start_mlflow_server()
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("supply_chain_delay_pipeline")

    print("[INFO] 加载数据...")
    df = load_orders_from_db()
    target_col = "Reached.on.Time_Y.N"
    y = df[target_col]
    X = df.drop(columns=[target_col, "ID"])

    # 特征列
    numeric_features = ["Customer_care_calls", "Customer_rating", "Cost_of_the_Product",
                        "Prior_purchases", "Discount_offered", "Weight_in_gms"]
    categorical_features = ["Warehouse_block", "Mode_of_Shipment", "Product_importance", "Gender"]

    # 从 model.py 获取 Pipeline（模型+预处理统一在那边定义）
    pipeline = create_pipeline(numeric_features, categorical_features)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    param_grid = {
        "classifier__n_estimators": [100, 200],
        "classifier__max_depth": [4, 7],
        "classifier__learning_rate": [0.01, 0.14]
    }

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

    with mlflow.start_run():
        mlflow.log_params(best_params)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(
            sk_model=best_pipeline,
            artifact_path="model",
            registered_model_name="suppy_chain_predict"
        )

    print("[INFO] Pipeline 已保存到 MLflow，请在 UI 设置 production alias。")

if __name__ == "__main__":
    main()