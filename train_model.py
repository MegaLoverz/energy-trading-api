import os
import sys
import yaml
from pyspark.sql import SparkSession
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
import mlflow
import mlflow.sklearn
import joblib

os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

from data_pipeline import clean_and_build_features

# 1. โหลด Config
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

IMPUTE_METHOD = config["pipeline"]["imputation_method"]
TARGET_SHIFT = config["pipeline"].get("target_shift_hours", 1)
DATA_PATH = config["pipeline"]["data_path"]
TEST_START = config["pipeline"]["test_start_date"]
NUM_TREES = config["model"]["num_trees"]
SEED = config["model"]["seed"]

# 2. เครื่องยนต์ Spark (ใช้จัดการ Big Data เท่านั้น)
spark = SparkSession.builder.appName("Energy_DataPrep").config("spark.ui.showConsoleProgress", "false").getOrCreate()

print(f"📥 [Spark] โหลดและทำความสะอาดข้อมูล...")
df_raw = spark.read.csv(DATA_PATH, header=True, inferSchema=True)
df_ready = clean_and_build_features(df_raw, method=IMPUTE_METHOD, target_shift=TARGET_SHIFT)
df_ready = df_ready.dropna(subset=["target_kw"])

train_df_spark = df_ready.filter(df_ready.timestamp < TEST_START)
test_df_spark = df_ready.filter(df_ready.timestamp >= TEST_START)

# 3. จุดเปลี่ยนผ่าน (Pivot): ดึงข้อมูลจาก Spark มาใส่ Pandas
print("🔄 กำลังสลับเครื่องยนต์ไปที่ Scikit-Learn...")
feature_cols = ["load_kw_filled", "is_imputed", "lag_1h", "rolling_avg_3h"]
train_pd = train_df_spark.select(feature_cols + ["target_kw"]).toPandas()
test_pd = test_df_spark.select(feature_cols + ["target_kw"]).toPandas()

X_train, y_train = train_pd[feature_cols], train_pd["target_kw"]
X_test, y_test = test_pd[feature_cols], test_pd["target_kw"]

# 4. เทรนและบันทึกโมเดลฉบับเบาหวิว
mlflow.set_experiment("Energy_Trading_Forecaster")
with mlflow.start_run(run_name="Hybrid_Sklearn_Model"):
    mlflow.log_param("architecture", "Spark_Data_Sklearn_Model")
    mlflow.log_param("num_trees", NUM_TREES)
    
    print(f"⚙️ [Sklearn] เทรนโมเดล API ความเร็วสูง...")
    rf = RandomForestRegressor(n_estimators=NUM_TREES, random_state=SEED)
    rf.fit(X_train, y_train)
    
    preds = rf.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    
    mlflow.log_metric("rmse", rmse)
    print(f"📊 Model RMSE: {rmse:.4f}")
    
    # ⚠️ บันทึกด้วย mlflow.sklearn (ไม่ใช่ mlflow.spark แล้ว)
    mlflow.sklearn.log_model(rf, "model_artifacts")
    #print("✅ บันทึกโมเดล API ลง MLflow สำเร็จ!")
    # ✨ NEW: สกัดโมเดลเพียวๆ เป็นไฟล์ .pkl สำหรับใช้บน API
    joblib.dump(rf, "model.pkl") 
    print("✅ บันทึกโมเดล model.pkl สำเร็จ!")