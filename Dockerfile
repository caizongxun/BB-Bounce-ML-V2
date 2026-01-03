# 使用 Python 3.10 作為基礎檢詢誅
FROM python:3.10-slim

# 設定工作目錄
WORKDIR /app

# 設定環境變數以防止 Python 缓存上校
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV FLASK_ENV=production

# 安裝系統依賴
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 複制 requirements.txt 並安裝 Python 依賴
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 複制整個應用程式
COPY . .

# 暗示 Flask API 在這個連接埠接聽
EXPOSE 5000

# 啟動知能 Flask 應用程式
CMD ["python", "realtime_service_v3.py"]
