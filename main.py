import yaml
from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
import pandas as pd
import joblib
import time

from serving_pipeline import build_features_fastapi

class MeterData(BaseModel):
    meter_id: str
    timestamp: datetime
    load_kw: Optional[float] = None

class ForecastRequest(BaseModel):
    data: List[MeterData]

model = None
IMPUTE_METHOD = "forward_fill"
TARGET_SHIFT = 1
FEATURE_COLS = ["load_kw_filled", "is_imputed", "lag_1h", "rolling_avg_3h"]

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, IMPUTE_METHOD, TARGET_SHIFT
    
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)
    IMPUTE_METHOD = config["pipeline"]["imputation_method"]
    TARGET_SHIFT = config["pipeline"].get("target_shift_hours", 1)

    # โหลดไฟล์โมเดลเพียวๆ โดยตรง (ข้ามระบบ OS ได้ 100%)
    try:
        model = joblib.load("model.pkl")
        print("📦 โหลดโมเดล model.pkl ขึ้น API สำเร็จ!")
    except Exception as e:
        print(f"❌ โหลดโมเดลไม่สำเร็จ: {e}")

    yield 

app = FastAPI(title="Energy Trading Platform API", lifespan=lifespan)

@app.post("/api/v1/forecast")
def get_forecast(request: ForecastRequest):
    start_time = time.time()
    if not model:
        return {"error": "Model is not loaded."}
        
    df_raw = pd.DataFrame([dict(d) for d in request.data])
    df_ready = build_features_fastapi(df_raw, method=IMPUTE_METHOD)
    X_infer = df_ready[FEATURE_COLS]
    
    pred = model.predict(X_infer)[0]
    
    latest_time = df_ready.iloc[0]["timestamp"]
    target_time = latest_time + pd.Timedelta(hours=TARGET_SHIFT)
    process_time_ms = round((time.time() - start_time) * 1000, 2)
    
    return {
        "status": "success",
        "current_time": latest_time.isoformat(),
        "forecast_target_time": target_time.isoformat(),
        "prediction_kw": round(pred, 2),
        "latency_ms": process_time_ms
    }