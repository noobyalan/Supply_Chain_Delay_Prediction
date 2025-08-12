from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Supply Chain Delay Predictor API"}