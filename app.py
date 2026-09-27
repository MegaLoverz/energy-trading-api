import streamlit as st
import requests

# ตั้งค่าหน้าเพจ
st.set_page_config(page_title="TOU Simulator", layout="centered")

st.title("⚡ AI Energy Auditor: TOU Simulator")
st.markdown("ระบบจำลองคาดการณ์ Peak Load สำหรับการประเมินค่าไฟ (PEA)")
st.divider()

# 1. สร้าง Form รับค่าจาก User (ตั้งค่า Min/Max เพื่อกันเหนียวซ้อนอีกชั้นที่หน้าเว็บ)
col1, col2 = st.columns(2)
with col1:
    month = st.number_input("Month (1-12)", min_value=1, max_value=12, value=9)
    day_of_week = st.number_input("Day of Week (1-7)", min_value=1, max_value=7, value=1)
with col2:
    off_peak_load = st.number_input("Off-Peak Avg Load (kW)", min_value=0.1, value=5000.0, step=500.0)

st.divider()

# 2. ปุ่มกดประมวลผล
if st.button("Predict Peak Load 🚀", use_container_width=True):
    
    # 3. แพ็คข้อมูลเป็น JSON Payload
    payload = {
        "Month": int(month),
        "DayOfWeek": int(day_of_week),
        "Off_Peak_Avg_Load": float(off_peak_load)
    }
    
    # URL ของ FastAPI ที่รันอยู่
    API_URL = "http://127.0.0.1:8000/predict"
    
    try:
        with st.spinner("กำลังส่งข้อมูลเข้า Databricks Engine..."):
            # 4. ใช้ requests ยิง POST Method ไปที่ API
            response = requests.post(API_URL, json=payload)
            
            # 5. เช็คผลลัพธ์
            if response.status_code == 200:
                result = response.json()
                predicted_load = result["Predicted_Peak_Load"]
                
                st.success("✅ ประมวลผลสำเร็จ!")
                # แสดงผลตัวเลขแบบมีลูกน้ำและทศนิยม 2 ตำแหน่ง
                st.metric(label="Predicted Peak Load (kW)", value=f"{predicted_load:,.2f}")
            else:
                st.error(f"❌ API Error: {response.status_code}")
                st.json(response.json())
                
    except requests.exceptions.ConnectionError:
        st.error("🚨 ไม่สามารถเชื่อมต่อกับ Backend ได้ โปรดตรวจสอบว่า FastAPI รันอยู่หรือไม่")