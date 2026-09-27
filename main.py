import os
from fastapi import FastAPI
from pydantic import BaseModel, Field
import mlflow.pyfunc
import pandas as pd

# 1. The Key: Authenticate with Databricks
from dotenv import load_dotenv
load_dotenv() # คำสั่งนี้จะไปดึงค่าใน .env มาจำลองเป็น Environment Variables ให้ระบบทันที
#os.environ["DATABRICKS_HOST"] = "https://dbc-2df2dec5-a500.cloud.databricks.com"

# 2. The Map: Tell MLflow to look inside the Unity Catalog, not your C: drive
mlflow.set_registry_uri("databricks-uc")

app = FastAPI(title="AI Energy Auditor API", version="1.0")

# 3. The Shield: Validate incoming data before it hits the engine
class LoadRequest(BaseModel):
    Month: int = Field(..., ge=1, le=12, description="เดือน (1-12)")
    DayOfWeek: int = Field(..., ge=1, le=7, description="วันในสัปดาห์ (1-7)")
    Off_Peak_Avg_Load: float = Field(..., gt=0, description="ค่า Load เฉลี่ยต้องมากกว่า 0")

model_uri = "models:/main.default.energy_load_forecaster/1"
model = mlflow.pyfunc.load_model(model_uri)

@app.post("/predict")
def predict_peak_load(request: LoadRequest): 
    # 1. Package the raw JSON inputs into a single combined array
    features_array = [[request.Month, request.DayOfWeek, request.Off_Peak_Avg_Load]]
    
    # 2. Wrap the array in a Pandas DataFrame using the exact column name MLflow requires
    input_df = pd.DataFrame({
        "features": features_array
    })
    
    # 3. Pass the structured DataFrame to the PySpark engine
    prediction = model.predict(input_df)
    
    # 4. Extract the final value (PySpark PyFunc often returns a DataFrame or a list)
    if isinstance(prediction, pd.DataFrame):
        # If it returns a dataframe with a 'prediction' column
        final_result = round(float(prediction["prediction"].iloc[0]), 2) 
    else:
        # If it returns a standard array/list
        final_result = round(float(prediction[0]), 2)
    
    return {"Predicted_Peak_Load": final_result}