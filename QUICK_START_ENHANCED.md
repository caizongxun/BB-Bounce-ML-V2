# 快速開始 - 增強版 RealtimeBBDetectorV2

本指南將帮助你快速配置多時間活侨波動預測系統。

## 第一步：訓練 15m Validity 模型

### 方案 A：完整訓練（推茶）

編輯 `train_validity_model.py`：

```python
if __name__ == '__main__':
    trainer = ValidityModelTrainer()
    
    # 訓練 15m 模型 (摊 2-4 小時)
    print("\n=== Training 15m Validity Models ===")
    results_15m = trainer.train_all_symbols('15m')
    
    # 訓練 1h 模型（已有）
    print("\n=== Training 1h Validity Models ===")
    results_1h = trainer.train_all_symbols('1h')
```

執行：

```bash
python train_validity_model.py
```

### 方案 B：最小化配置

如果你已經有 1h 模型，可以先跳過 15m 訓練，直接使用完成的模型。

```bash
# 驗證目前的 1h 模型是否可用
ls models/validity_models/AAVEUSDT/1h/
# 應輸出：
# feature_names.pkl  scaler.pkl  validity_model.pkl
```

## 第二步：訓練 1h Volatility 模型

### 檢查是否存在

```bash
ls models/vol_models/BTCUSDT/1h/

# 應輸出：
# model_regression.pkl  scaler_regression.pkl
```

### 若沒有，訓練 Vol 模型

```bash
python train_vol_model.py
```

## 第三步：検測模型

確認你的模型目錄結構：

```
models/
├── bb_models/
│   └── BTCUSDT/
│       ├── 15m/ (o) 
│       └── 1h/ (o)
├── validity_models/
│   └── BTCUSDT/
│       ├── 15m/ (o) 新訓練
│       └── 1h/ (o)
├── vol_models/
    └── BTCUSDT/
        └── 1h/ (o)
```

## 第四步：使用增強検測器

### 基本使用

```python
from realtime_detector_v2_enhanced import RealtimeBBDetectorV2Enhanced
import json
from datetime import datetime

# 初始化検測器
detector = RealtimeBBDetectorV2Enhanced(
    model_dir="models",
    device="cpu",
    supported_timeframes=["15m", "1h"]
)

# 模擬 K 線數據
candle = {
    "timestamp": int(datetime.now().timestamp() * 1000),
    "open": 140.5,
    "high": 141.2,
    "low": 139.8,
    "close": 140.8,
    "volume": 5200,
    "bb_upper": 142.0,
    "bb_middle": 140.0,
    "bb_lower": 138.0,
    "rsi": 62.5,
    "adx": 22.3,
    "atr": 1.5
}

# 添加 K 線 - 15m 版本
detector.add_candle("AAVEUSDT", candle, timeframe="15m")

# 添加 K 線 - 1h 版本
detector.add_candle("AAVEUSDT", candle, timeframe="1h")

# 掃描 15m 信號
signal_15m = detector.scan("AAVEUSDT", timeframe="15m")
if signal_15m:
    print("15m Signal:")
    print(json.dumps(signal_15m, indent=2))

# 掃描 1h 信號
signal_1h = detector.scan("AAVEUSDT", timeframe="1h")
if signal_1h:
    print("\n1h Signal:")
    print(json.dumps(signal_1h, indent=2))
```

### 輸出示例

```json
{
    "symbol": "AAVEUSDT",
    "timeframe": "15m",
    "side": "long",
    "bb_position_label": "Lower",
    "layer1_class": 2,
    "validity_prob": 0.82,
    "confidence": 0.82,
    "predicted_volatility": 0.0356,
    "predicted_volatility_unit": "daily_pct_change",
    "rsi": 35.2,
    "adx": 24.1,
    "vol_ratio": 1.45,
    "timestamp": 1735880000000
}
```

## 第五步：集成到實時交易系統

### 更新 realtime_service.py

```python
from realtime_detector_v2_enhanced import RealtimeBBDetectorV2Enhanced

class RealtimeService:
    def __init__(self):
        # 使用增強版本
        self.detector = RealtimeBBDetectorV2Enhanced(
            model_dir="models",
            device="cpu",
            supported_timeframes=["15m", "1h"]
        )
    
    def on_candle(self, symbol: str, candle: dict):
        # 判斷是 15m 或 1h K 線
        timeframe = self.get_timeframe(candle["timestamp"])
        
        # 添加 K 線
        self.detector.add_candle(symbol, candle, timeframe=timeframe)
        
        # 掃描信號
        signal = self.detector.scan(symbol, timeframe=timeframe)
        
        if signal:
            # 高信心度的信號
            if signal['confidence'] > 0.75:
                # 獲取波動預測
                volatility = self.detector.predict_volatility(symbol)
                
                # 決定是否進場
                should_trade = self._should_trade(signal, volatility)
                
                if should_trade:
                    self.execute_trade(symbol, signal, volatility)
    
    def _should_trade(self, signal: dict, volatility: float = None) -> bool:
        """
        根據信號和波動似然性决定是否下單
        """
        # 基本條件：信心度 > 75%
        if signal['confidence'] <= 0.75:
            return False
        
        # 波動性條件：如果有波動預測，確保波動不超過 10%
        if volatility is not None:
            daily_vol_pct = volatility * 100
            if daily_vol_pct > 10:
                print(f"Skip: High volatility ({daily_vol_pct:.2f}%)")
                return False
        
        return True
    
    def execute_trade(self, symbol: str, signal: dict, volatility: float = None):
        """
        執行交易
        """
        entry_price = signal['close'] if 'close' in signal else None
        
        # 計算止損
        if volatility is not None:
            stop_loss_pct = volatility * 1.5  # 波動率的 1.5 倍
        else:
            stop_loss_pct = 0.02  # 預設 2%
        
        trade_params = {
            "symbol": symbol,
            "side": signal['side'],  # "long" or "short"
            "entry_price": entry_price,
            "stop_loss_pct": stop_loss_pct,
            "confidence": signal['confidence'],
            "volatility": volatility,
            "signal": signal
        }
        
        print(f"Execute trade: {trade_params}")
        # 客戶帳戶API...
```

## 第六步：測試多時間活侨

### 樣本代碼

```python
from realtime_detector_v2_enhanced import RealtimeBBDetectorV2Enhanced
import time

detector = RealtimeBBDetectorV2Enhanced(model_dir="models")

# 模擬數據流入
candles_15m = [
    {
        "timestamp": 1735880000000 + i*15*60*1000,
        "open": 140 + i*0.1, "high": 141 + i*0.1, 
        "low": 139 + i*0.1, "close": 140.5 + i*0.1,
        "volume": 5000 + i*100,
        "bb_upper": 142 + i*0.1, "bb_middle": 140 + i*0.1, 
        "bb_lower": 138 + i*0.1,
        "rsi": 50 + i*2, "adx": 20 + i, "atr": 1.5
    }
    for i in range(10)
]

# 添加數據
for i, candle in enumerate(candles_15m):
    detector.add_candle("BTCUSDT", candle, timeframe="15m")
    
    # 掃描
    signal_15m = detector.scan("BTCUSDT", timeframe="15m")
    if signal_15m:
        print(f"[{i}] 15m Signal: {signal_15m['side']} @ {signal_15m['confidence']:.2%}")

print("\n15m Scanning complete!")
```

## 常見故障排除

### 問題 1：模型沒有加載

検查：

```bash
ls models/validity_models/AAVEUSDT/15m/

# 应輸出：
# feature_names.pkl  scaler.pkl  validity_model.pkl
```

如果缺少，需要訓練：

```bash
python train_validity_model.py
```

### 問題 2：15m 掃描返回 None

原因：

1. 沒有皳够的 15m 歷史數據（需要最少 5 根）
2. 15m 分類器未正確加載
3. K 線缺少必需字段（rsi, adx, atr, bb_*）

檢查緩衝區：

```python
buffer_size = len(detector.candle_buffer["AAVEUSDT"]["15m"])
print(f"Buffer size: {buffer_size}")

if buffer_size > 0:
    last_candle = list(detector.candle_buffer["AAVEUSDT"]["15m"])[-1]
    print(f"Last candle: {last_candle}")
```

### 問題 3：波動預測返回 None

原因：

1. vol_models 沒有加載
2. 1h 歷史數據不足 (< 20 根)

検查：

```python
# 確認 vol_models 是否載入
if "AAVEUSDT" in detector.vol_models:
    print("Vol model found")
else:
    print("Vol model NOT found")

# 確認 1h 緩衝區
buffer_1h_size = len(detector.candle_buffer["AAVEUSDT"]["1h"])
print(f"1h buffer size: {buffer_1h_size}")
```

## 效能基準

| 指標 | 預預值 | 先決条件 |
|------|--------|--------|
| BB 分類准確度 | > 98% | 模型訓練 |
| 有效性准確率 | 90-94% | 有 validity_model |
| 有效性召回率 | 80-88% | 有 validity_model |
| 波動預測 R² | 0.45-0.65 | 有 vol_model |
| 單個符號掃描速度 | 30-80ms | 不含外部 API 邀遲 |
| 22 個符號全掃 | 0.7-1.8s | 不含外部 API 邀遲 |

## 整整求審清單

- [ ] 訓練 15m validity 模型
- [ ] 驗證 15m validity 模型 准確度 > 90%
- [ ] 驗證 1h vol 模型是否存在
- [ ] 測試一個符號的 15m + 1h + vol 預測
- [ ] 整合到 realtime_service.py
- [ ] 執行室外響造市场測試

## 下一步

如需了解詳詳訹究竞競臥的模型細節，請參考 `TRAINING_GUIDE_15M.md`。
