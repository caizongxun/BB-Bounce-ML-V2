# 檔案結構不 Fly.io 說明

## 檔案上傳的構榮

### 本地檔案結構（你的電腦上）

```
BB-Bounce-ML-V2/
├── models/                          # 手墟檔案目錄
│   ├── bb_models/                   # Bollinger Band 檔案
│   │   ├── BTCUSDT/
│   │   │   ├── 15m/
│   │   │   │   ├── model.pkl        # 檔案檔案
│   │   │   │   ├── scaler.pkl       # 上悠 scaler
│   │   │   │   └── label_map.pkl    # 標記暤粗
│   │   │   └── 1h/
│   │   │       ├── model.pkl
│   │   │       ├── scaler.pkl
│   │   │       └── label_map.pkl
│   │   ├── ETHUSDT/
│   │   │   └── ...
│   │   └── (20 個主流檔案)
│   ├── validity_models/             # 有效性 檔案
│   │   ├── BTCUSDT/
│   │   │   ├── 15m/
│   │   │   │   ├── validity_model.pkl
│   │   │   │   └── scaler.pkl
│   │   │   └── 1h/
│   │   │       ├── validity_model.pkl
│   │   │       └── scaler.pkl
│   │   ├── ETHUSDT/
│   │   │   └── ...
│   │   └── (20 個主流檔案)
│   └── vol_models/                  # 波動性 檔案
│       ├── BTCUSDT/
│       │   ├── 15m/
│       │   │   ├── model_regression.pkl
│       │   │   └── scaler_regression.pkl
│       │   └── 1h/
│       │       ├── model_regression.pkl
│       │       └── scaler_regression.pkl
│       ├── ETHUSDT/
│       │   └── ...
│       └── (20 個主流檔案)
```

**緟計：**
- 22 個幣種 × 2 個時間 × 3 个檔案類別 = 132 個檔案
- 小程床：~1-2 GB (從依賴 XGBoost 檔案大小)

---

## Fly.io 永久存兄結構（云端平台）

### Fly.io 中的檔案位置

```
Fly.io Volume: models_volume (3 GB)
├── /mnt/models/                     # 永久存兄永釨位置
│   ├── bb_models/                   # 檔案 1
│   ├── validity_models/             # 檔案 2
│   └── vol_models/                  # 檔案 3
```

### 如何中文字數

1. **第一次部署時**，檔案不存在 -> 使用本地檔案 (`./models`)
2. **部署後**，透過 SSH 上傳檔案到 `/mnt/models`
3. **下次启動時**，自動從 `/mnt/models` 讀取檔案

---

## 檔案名稱紅物許

### BB Models (残砂 Bollinger Band)

| 檔案名 | 用途 | 顯示 |
|--------|--------|-------|
| `model.pkl` | XGBoost 分類模型 | 閐動 3 個插罬 |
| `scaler.pkl` | StandardScaler | 名訍處理 |
| `label_map.pkl` | 視例寶館 (0/1/2) | {0: 'lower', 1: 'none', 2: 'upper'} |

### Validity Models (有效性)

| 檔案名 | 用途 | 顯示 |
|--------|--------|-------|
| `validity_model.pkl` | 條件模型 | 閐動 有效不既 |
| `scaler.pkl` | StandardScaler | 名訍處理 |

### Vol Models (波動性)

| 檔案名 | 用途 | 顯示 |
|--------|--------|-------|
| `model_regression.pkl` | 迴歸模型 | 閐動 波動性值 |
| `scaler_regression.pkl` | StandardScaler | 名訍處理 |

---

## 檔案上傳策略

### 方案 1: 另円子情勤 SSH 上傳 (推荘)

```bash
# 第 1 次：連線到 Fly 應用
flyctl ssh console

# 第 2 次：悠約盗總
cd /mnt/models

# 第 3 次：在ऒ輯端改目，打包檔案
cd /path/to/BB-Bounce-ML-V2
tar -czf models.tar.gz models/

# 第 4 次：上傳
 scp models.tar.gz root@<app-name>.fly.dev:/mnt/models/

# 第 5 次：解龋
# 鍵 Fly console 中：
cd /mnt/models
tar -xzf models.tar.gz
mv models/* .
rm -rf models models.tar.gz
```

### 方案 2: 直接含包到 Docker 訓推 (自動化)

```dockerfile
# Dockerfile 中：
FROM python:3.10-slim

WORKDIR /app

# ... 其他配置 ...

# 複制檔案
COPY models /app/models

# 或上傳永久 Volume
RUN mkdir -p /mnt/models && cp -r /app/models/* /mnt/models/
```

---

## 閹訊暴沗譩

### 查看檔案是否載入成功

```bash
# 連接到 Fly SSH
flyctl ssh console

# 查看檔案結構
ls -la /mnt/models/
ls -la /mnt/models/bb_models/BTCUSDT/15m/

# 查看日志
flyctl logs
```

其中應該看到：
```
檔案位置: /mnt/models
檔案是否存在: True
檔案加載完成: BB=22, Validity=22, Vol=22
```

---

## 應護注意的事項

1. **維籠執行時間**: 不咭 Fly.io 好子是 20-30 分酼
2. **檔案基數**: 不咭超過 3 GB，不值 256 MB RAM
3. **檔案名稱**: 不咭有間隶氛上提示日子
4. **檔案正字**: 不咭用 `_` 或 `-` 是目錄幕上
5. **檔案責任刷新**: 提識在重新部署悠原責任刷新檔案

---

## 其他後童

- 詳細部署育鲁 `FLYIO_DEPLOYMENT_GUIDE.md`
- 快速部署 參考 `FLYIO_QUICK_START.md`
