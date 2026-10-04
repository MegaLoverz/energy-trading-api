import pandas as pd

def build_features_fastapi(df_raw: pd.DataFrame, method="forward_fill") -> pd.DataFrame:
    """โรงงานแปรรูปข้อมูลฉบับ Pandas (เน้นความเร็ว)"""
    df = df_raw.sort_values("timestamp").copy()
    
    # 1. อุดรอยรั่ว
    if method == "forward_fill":
        df["load_kw_filled"] = df["load_kw"].ffill()
    elif method == "zero_fill":
        df["load_kw_filled"] = df["load_kw"].fillna(0.0)
        
    df["is_imputed"] = df["load_kw"].isnull().astype(int)
    
    # 2. สร้าง Features (ดึงอดีต)
    df["lag_1h"] = df["load_kw_filled"].shift(1)
    df["rolling_avg_3h"] = df["load_kw_filled"].rolling(window=3, min_periods=1).mean().shift(1)
    
    # ส่งกลับเฉพาะบรรทัดล่าสุดที่ต้องการทำนาย (แถวเดียว)
    latest_row = df.iloc[[-1]].copy()
    latest_row.fillna(0, inplace=True)
    return latest_row