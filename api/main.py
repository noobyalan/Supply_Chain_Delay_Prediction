from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.requests import Request
from starlette.exceptions import HTTPException as StarletteHTTPException
import mlflow
import pandas as pd
from pydantic import BaseModel

mlflow.set_tracking_uri("http://127.0.0.1:5000")

MODEL_NAME = "supply_chain_predict"
MODEL_ALIAS = "prod"
MODEL_URI = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
model = mlflow.pyfunc.load_model(MODEL_URI)

app = FastAPI(title="Supply Chain Delay Prediction API")

# 自定义根路由
@app.get("/")
def read_root():
    return {
        "message": " Supply Chain Delay Prediction API is running",
        "docs": "/docs 进入 API 文档",
        "health": "/health 检查 API 健康"
    }

# 自定义 404 handler
@app.exception_handler(StarletteHTTPException)
async def custom_404_handler(request: Request, exc: StarletteHTTPException):
    if exc.status_code == 404:
        return JSONResponse(status_code=404, content={
            "error": f"路径 {request.url.path} 不存在，请访问 /docs 查看可用接口"
        })
    return JSONResponse(status_code=exc.status_code, content={"error": str(exc.detail)})

class OrderFeatures(BaseModel):
    ID: int
    Warehouse_block: str
    Mode_of_Shipment: str
    Customer_care_calls: int
    Customer_rating: int
    Cost_of_the_Product: float
    Prior_purchases: int
    Product_importance: str
    Gender: str
    Discount_offered: float
    Weight_in_gms: float

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/predict")
def predict(features: OrderFeatures):
    df = pd.DataFrame([features.dict()]).drop(columns=["ID"])
    pred = model.predict(df)
    return {"prediction": int(pred[0])}