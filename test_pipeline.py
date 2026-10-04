import os
import sys
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, TimestampType, DoubleType
from datetime import datetime

# --- เพิ่ม 2 บรรทัดนี้เพื่อแก้ปัญหา PySpark บน Windows ---
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

from data_pipeline import clean_and_build_features 

# 1. จำลอง Spark Session (คล้ายๆ การเปิดเครื่องยนต์ PySpark)
spark = SparkSession.builder.appName("EnergyForecasting_Test").getOrCreate()

# 2. สร้าง Mock Data (ตั้งใจให้ตอน 11:00 น. ข้อมูลหายไป / เป็น None)
mock_data = [
    ("meter_001", datetime(2026, 9, 30, 8, 0), 40.0),
    ("meter_001", datetime(2026, 9, 30, 9, 0), 50.0),
    ("meter_001", datetime(2026, 9, 30, 10, 0), 60.0),
    ("meter_001", datetime(2026, 9, 30, 11, 0), None), # 🚨 เน็ตหลุด! Data แหว่ง
    ("meter_001", datetime(2026, 9, 30, 12, 0), 80.0),
    ("meter_001", datetime(2026, 9, 30, 13, 0), 90.0),
    ("meter_001", datetime(2026, 9, 30, 14, 0), 100.0)
]

schema = StructType([
    StructField("meter_id", StringType(), True),
    StructField("timestamp", TimestampType(), True),
    StructField("load_kw", DoubleType(), True)
])

df_raw = spark.createDataFrame(mock_data, schema)
print("--- ข้อมูลดิบ (Raw Data) ---")
df_raw.show()

# 3. โยนเข้าโรงงานแปรรูปข้อมูลของเรา!
df_result = clean_and_build_features(df_raw)

# 4. โชว์ผลลัพธ์เฉพาะคอลัมน์สำคัญเพื่อตรวจสอบความถูกต้อง
print("--- ข้อมูลที่ผ่าน Pipeline แล้ว (พร้อมเข้าโมเดล) ---")
df_result.select(
    "timestamp", 
    "load_kw", 
    "load_kw_filled", 
    "is_imputed", 
    "load_lag_1h", 
    "rolling_avg_3h"
).show()