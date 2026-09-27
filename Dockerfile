# 1. Pull the exact Python version Databricks used for the model
FROM python:3.12.3-slim

# 2. Install the Java Virtual Machine (JVM) required by PySpark
RUN apt-get update && \
    apt-get install -y default-jre && \
    apt-get clean

# 3. Set the working directory inside the isolated container
WORKDIR /app

# 4. Copy your requirements and main script into the container
COPY requirements.txt .
COPY main.py .

# 5. Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 6. Open the network port for the API
EXPOSE 8000

# 7. Start the Uvicorn server engine (0.0.0.0 allows external connections)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]