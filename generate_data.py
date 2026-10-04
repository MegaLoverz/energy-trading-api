import pandas as pd
import numpy as np

print("⏳ กำลังจำลองข้อมูล AMR...")
# 1. สร้างเวลา 30 วัน (1-30 ก.ย. 2026) ความถี่ทุก 1 ชั่วโมง
dates = pd.date_range(start="2026-09-01", end="2026-09-30 23:00:00", freq="h")

# 2. จำลองพฤติกรรมใช้ไฟ (Peak กลางวัน, Off-peak กลางคืน) + แอบใส่ Noise แกว่งๆ เล็กน้อย
loads = 80 + 40 * np.sin(dates.hour * (2 * np.pi / 24) - (np.pi / 2)) + np.random.normal(0, 5, len(dates))

df = pd.DataFrame({
    "meter_id": "M-001",
    "timestamp": dates,
    # ⚠️ แก้ไขบรรทัดนี้: ใช้ np.round ของ NumPy
    "load_kw": np.round(loads, 2) 
})

# 3. สุ่มดึงสายแลนออก (สร้าง Null 5% เพื่อเทส Pipeline อุดรอยรั่ว)
np.random.seed(42)
mask = np.random.rand(len(df)) < 0.05
df.loc[mask, "load_kw"] = np.nan

# 4. เซฟเป็นไฟล์ CSV
df.to_csv("amr_data.csv", index=False)
print(f"✅ สร้างไฟล์ amr_data.csv สำเร็จ! จำนวน {len(df)} แถว")