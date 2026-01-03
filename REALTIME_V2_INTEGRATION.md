# REALTIME_DETECTOR_V2 集成指南

本文檔說明如何將 `realtime_detector_v2.py` 與現有的 `realtime_service.py`、`realtime_dashboard_v2.html` 完整整合，形成企業級實時檢測系統。

## 1. 系統架構概覽

- 二層架構:
  - 層級 1: 位置檢測模型 (超快，10–20ms)
  - 層級 2: 有效性驗證模型 (50–100ms)
- 實時通訊: Flask-SocketIO + WebSocket
- 前端: `realtime_dashboard_v2.html` (實時儀表板)
- 後端: `realtime_service.py` + `realtime_detector_v2.py`

## 2. 檔案結構建議

專案根目錄建議如下:

```
BB-Bounce-ML-V2/
  ├── realtime_detector_v2.py
  ├── realtime_service.py
  ├── realtime_dashboard_v2.html
  ├── QUICKSTART_V2.md
  ├── REALTIME_V2_INTEGRATION.md (本文)
  ├── SYSTEM_ARCHITECTURE_V2.md
  ├── IMPLEMENTATION_SUMMARY.md
  ├── INTEGRATION_CHECKLIST.md
  ├── QUICK_REFERENCE.md
  └── models/               # 22 個幣種的模型檔案
```

## 3. 後端整合步驟

### 3.1 在 `realtime_service.py` 載入 V2 檢測器

```python
from realtime_detector_v2 import RealtimeBBDetectorV2

# 在全域初始化
bb_detector_v2 = RealtimeBBDetectorV2(
    symbols=[
        "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
        "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "TONUSDT", "LINKUSDT",
        "OPUSDT", "ARBUSDT", "SUIUSDT", "APEUSDT", "MATICUSDT",
        "LTCUSDT", "TRXUSDT", "FTMUSDT", "INJUSDT", "SEIUSDT",
        "TIAUSDT", "ORDIUSDT",
    ],
    model_dir="models",          # 模型存放目錄
    device="cuda"                # 或 "cpu"
)
```

### 3.2 實現實時掃描 loop

在 `realtime_service.py` 中新增一個背景執行緒，定期從交易所/資料源抓取最新 K 線，丟給 `bb_detector_v2` 分析。

```python
import threading
import time

from flask_socketio import SocketIO

socketio = SocketIO(app, cors_allowed_origins="*")


def realtime_loop():
    """主實時檢測 Loop: 每秒執行一次"""
    while True:
        try:
            # 1. 從資料源取得最新 K 線 (此處僅示意)
            # candles = fetch_latest_candles(symbols=bb_detector_v2.symbols)

            # 2. 執行二層檢測
            # signals = bb_detector_v2.scan(candles)

            # 3. 逐筆透過 WebSocket 推送給前端
            # for sig in signals:
            #     socketio.emit("realtime_signal", sig)

            time.sleep(1.0)
        except Exception as e:
            print(f"[realtime_loop] error: {e}")
            time.sleep(1.0)


# 在程式啟動時啟動背景執行緒
threading.Thread(target=realtime_loop, daemon=True).start()
```

> 註: 真實程式中請替換 `fetch_latest_candles` 與 `bb_detector_v2.scan` 的實作為實際版本。

### 3.3 WebSocket 事件 (前端互動)

```python
@socketio.on("connect")
def handle_connect():
    print("[socket] client connected")


@socketio.on("disconnect")
def handle_disconnect():
    print("[socket] client disconnected")


@socketio.on("force_refresh")
def handle_force_refresh(data):
    """前端點擊『強制刷新所有幣種』時觸發"""
    try:
        # 示意: 強制重新跑一次完整掃描
        # candles = fetch_latest_candles(symbols=bb_detector_v2.symbols, full_window=True)
        # signals = bb_detector_v2.scan(candles, force_full=True)
        # for sig in signals:
        #     socketio.emit("realtime_signal", sig)
        pass
    except Exception as e:
        print(f"[force_refresh] error: {e}")
```

### 3.4 儀表板路由

```python
from flask import render_template, send_from_directory

@app.route("/detector")
def detector_dashboard():
    # 若使用 Flask templates, 請把 html 放在 templates 目錄下
    return render_template("realtime_dashboard_v2.html")


# 若你選擇把檔案放在專案根目錄，也可以這樣:
# @app.route("/realtime_dashboard_v2.html")
# def detector_dashboard_file():
#     return send_from_directory(".", "realtime_dashboard_v2.html")
```

## 4. 前端整合要點

`realtime_dashboard_v2.html` 已經內建以下 WebSocket 事件處理:

- `socket.on("connect")`: 顯示系統健康狀態
- `socket.on("disconnect")`: 顯示連線中斷
- `socket.on("realtime_signal", payload)`: 接收單筆信號並更新 UI
- `socket.emit("force_refresh", {})`: 由按鈕觸發，請在後端實作對應邏輯

### 4.1 後端輸出 `realtime_signal` 的 payload 格式

建議統一使用以下鍵值，以對應前端欄位:

```python
signal = {
    "symbol": "BTCUSDT",          # 幣種
    "timeframe": "15m",           # 時間框架
    "side": "long",               # long / short
    "validity_prob": 0.82,         # 有效性模型輸出 (0-1)
    "bb_position_label": "Upper", # BB 觸及位置
    "rsi": 68.3,                   # 可選
    "adx": 24.1,                   # 可選
    "vol_ratio": 1.45,             # 可選: 當前成交量 vs 平均
    "timestamp": 1735880000000,    # (ms) or ISO string
}

socketio.emit("realtime_signal", signal)
```

前端會幫你:

- 自動計算品質分類: `EXCELLENT / GOOD / MODERATE / WEAK / POOR`
- 顯示 `P(success)`、RSI、Vol、ADX 等
- 更新幣種列表中的最後信號時間 & 狀態

## 5. 性能與資源建議

- 推薦部署:
  - 5 幣 ~ CPU 10–20% · RAM 300–500MB
  - 22 幣 ~ 建議使用獨立服務 (可分片)
- WebSocket 心跳間隔: 20–30 秒
- 如需高頻 (sub-second) 更新，建議控制前端渲染頻率 (節流)

## 6. 驗收與 Debug

- 先依照 `QUICKSTART_V2.md` 完成基本啟動
- 依 `INTEGRATION_CHECKLIST.md` 做逐項驗收:
  - WebSocket 連線正常
  - 實時信號持續更新
  - 儀表板 CPU/RAM 顯示符合預期
  - false positive 約 < 10%

---

詳細架構說明請參考 `SYSTEM_ARCHITECTURE_V2.md`，實現細節與設計 rationale 可查看 `IMPLEMENTATION_SUMMARY.md`。