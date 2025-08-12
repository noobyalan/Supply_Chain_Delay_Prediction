#!/bin/bash

# 切换到项目根目录
cd "$(dirname "$0")/.."

# 从 config.py 读取 PG 配置
USER=$(python -c "from src.config import PG_CONFIG; print(PG_CONFIG['user'])")
PASS=$(python -c "from src.config import PG_CONFIG; print(PG_CONFIG['password'])")
HOST=$(python -c "from src.config import PG_CONFIG; print(PG_CONFIG['host'])")
PORT=$(python -c "from src.config import PG_CONFIG; print(PG_CONFIG['port'])")

# MLflow 专用数据库名
DB="mlflow_db"

# 检查 mlruns 文件夹是否存在
if [ ! -d "mlruns" ]; then
  mkdir mlruns
fi

echo "[INFO] 启动 MLflow Tracking Server..."
echo "[INFO] Backend Store: PostgreSQL ($DB) via $USER@$HOST:$PORT"
echo "[INFO] Artifact Store: $(pwd)/mlruns"
echo "[INFO] UI: http://127.0.0.1:5000"

mlflow server \
  --backend-store-uri postgresql://$USER:$PASS@$HOST:$PORT/$DB \
  --default-artifact-root "$(pwd)/mlruns" \
  --host 0.0.0.0 \
  --port 5000