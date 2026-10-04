FROM python:3.10-slim

WORKDIR /app

# เปลี่ยนมาเรียกใช้ไฟล์ prod
COPY requirements-prod.txt .
RUN pip install --no-cache-dir -r requirements-prod.txt

COPY main.py .
COPY serving_pipeline.py .
COPY config.yaml .
COPY mlruns/ ./mlruns/

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]