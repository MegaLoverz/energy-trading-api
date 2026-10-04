from pyspark.sql import Window
import pyspark.sql.functions as F
from pyspark.ml.feature import VectorAssembler

def clean_and_build_features(raw_df, method="forward_fill", target_shift=1):
    
    # 1. อุดรอยรั่ว (เหมือนเดิม)
    if method == "forward_fill":
        window_impute = Window.partitionBy("meter_id").orderBy("timestamp").rowsBetween(Window.unboundedPreceding, 0)
        df_clean = raw_df.withColumn("load_kw_filled", F.last("load_kw", ignorenulls=True).over(window_impute))
    elif method == "zero_fill":
        df_clean = raw_df.withColumn("load_kw_filled", F.coalesce(F.col("load_kw"), F.lit(0.0)))
    else:
        raise ValueError(f"ไม่รองรับวิธีอุดรอยรั่วแบบ: {method}")

    df_clean = df_clean.withColumn("is_imputed", F.when(F.col("load_kw").isNull(), 1).otherwise(0))
    
    # ---------------------------------------------------------
    # ✨ NEW: Target Shifting (ดึงอนาคตมาเป็นคำเฉลย)
    # ---------------------------------------------------------
    window_shift = Window.partitionBy("meter_id").orderBy("timestamp")
    # F.lead คือการดึงค่าแถวถัดไป (อนาคต) ขึ้นมาแปะบรรทัดปัจจุบัน
    df_clean = df_clean.withColumn("target_kw", F.lead("load_kw_filled", target_shift).over(window_shift))

    # 2. สร้าง Features จากอดีต (เหมือนเดิม)
    window_lag = Window.partitionBy("meter_id").orderBy("timestamp")
    df_clean = df_clean.withColumn("lag_1h", F.lag("load_kw_filled", 1).over(window_lag))
    
    window_rolling = Window.partitionBy("meter_id").orderBy("timestamp").rowsBetween(-3, -1)
    df_clean = df_clean.withColumn("rolling_avg_3h", F.avg("load_kw_filled").over(window_rolling))
    
    # 3. จัดกลุ่ม Features เข้า Vector (เหมือนเดิม)
    assembler = VectorAssembler(
        inputCols=["load_kw_filled", "is_imputed", "lag_1h", "rolling_avg_3h"],
        outputCol="features",
        handleInvalid="skip"
    )
    
    df_ready = assembler.transform(df_clean)
    return df_ready