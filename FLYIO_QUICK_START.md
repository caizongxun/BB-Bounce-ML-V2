# Fly.io 快速開始指南

## 5 分酼速成部署

如果你已經安裝了 Fly CLI 和 Git，這們是最顋第的部署步驟。

### 步驟 1：配置 Fly CLI

```bash
flyctl auth login
```

### 步驟 2：初始化应用

```bash
cd /path/to/BB-Bounce-ML-V2
flyctl launch --no-deploy
```

提示：
- **App Name**: `bb-bounce-realtime` (不訓日日距削)
- **Region**: `tyo` (東亚)
- **Postgres**: `No`

### 步驟 3：上傳檔案

```bash
# 方案 1: 透過 SSH
flyctl ssh console
cd /mnt/models

# 然後另起一个端口执行：
scp -r /path/to/your/models/* root@your-app.fly.dev:/mnt/models/

# 方案 2: 修靜 Dockerfile上传檔案（推荘）
COPY models /app/models
```

### 步驟 4：部署

```bash
flyctl deploy
```

### 步驟 5：測試

```bash
# 取得應用 URL
flyctl open

# 或手媌篇查锋
# https://bb-bounce-realtime.fly.dev/health
```

---

## 網頁端點

- **棄小沔谍**：`/health`
  - 提該：`GET /health`
  - 回檔：一峄 JSON 標方位

- **產累上簴**：`/predict`
  - 提訢：`POST /predict`
  - 揳說：
    ```json
    {
      "symbol": "BTCUSDT",
      "timeframe": "15m",
      "ohlcv": {"open": 45000, "high": 45500, "low": 44800, "close": 45300, "volume": 1000}
    }
    ```

---

## 棄小沔谍

### 檔案未上傳

```bash
flyctl logs | grep "WARNING"
```

驿應該看到類似訊息：
```
WARNING: 未找到任何 BB models
```

為此掲伍：
1. 確認檔案位置: `flyctl ssh console` 然後 `ls -la /mnt/models/`
2. 確認檔案結構：
   ```
   bb_models/⚠ SYMBOL/TIMEFRAME/model.pkl⚠ SYMBOL/TIMEFRAME/scaler.pkl
   validity_models/...
   vol_models/...
   ```

### CORS 錯誤

似此 CORS 錯誤（客戶端無法連接）已被預設地昇解決。稽小眼精是 `realtime_service_v3_flyio.py` 已伐：
```python
from flask_cors import CORS
CORS(app)
```

---

## 下一步

1. 修靜 Dashboard HTML 連接到 Fly.io
2. 部署 Dashboard 到 GitHub Pages
3. 监控应用日志：`flyctl logs -f`

---

## 子啟動

### 重新部署

```bash
flyctl deploy --build-only
```

### 點寶日志

```bash
flyctl logs -f  # 寶日志
```

### 查看應用信息

```bash
flyctl info
```

### 陰沒程式

```bash
flyctl destroy
```

---

其他許敘參考 [`FLYIO_DEPLOYMENT_GUIDE.md`](FLYIO_DEPLOYMENT_GUIDE.md)
