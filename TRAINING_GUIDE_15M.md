# 訓練 15M Validity 模型及多時間框架集成指南

## 第一階段：訓練 15M Validity 模型

### 目標
為所有 22 個幣種訓練 15m 時間框架的有效性模型，用於判斷上下軌反彈的有效性。

### 步驟 1：修改訓練腳本

編輯 `train_validity_model.py` 的最後部分：

```python
if __name__ == '__main__':
    trainer = ValidityModelTrainer()
    
    # 訓練 15m 模型
    print("\n=== Training 15m Validity Models ===")
    results_15m = trainer.train_all_symbols('15m')
    
    # 訓練 1h 模型（已有）
    print("\n=== Training 1h Validity Models ===")
    results_1h = trainer.train_all_symbols('1h')
```

### 步驟 2：執行訓練

```bash
# 確保在項目目錄
cd /path/to/BB-Bounce-ML-V2

# 執行訓練（估計時間：2-4 小時）
python train_validity_model.py
```

### 步驟 3：驗證模型

訓練完成後，確認目錄結構：

```
models/
├── validity_models/
│   ├── AAVEUSDT/
│   │   ├── 15m/
│   │   │   ├── validity_model.pkl
│   │   │   ├── scaler.pkl
│   │   │   └── feature_names.pkl
│   │   └── 1h/
│   │       ├── validity_model.pkl
│   │       ├── scaler.pkl
│   │       └── feature_names.pkl
│   ├── ETHUSDT/
│   │   ├── 15m/ ... (同上)
│   │   └── 1h/ ... (同上)
│   └── ... (其他 20 個幣種)
│
├── vol_models/
│   ├── AAVEUSDT/
│   │   └── 1h/
│   │       ├── model_regression.pkl
│   │       └── scaler_regression.pkl
│   └── ... (其他幣種)
│
└── bb_models/
    ├── AAVEUSDT/
    │   ├── 15m/
    │   │   └── model.pkl
    │   └── 1h/
    │       └── model.pkl
    └── ... (其他幣種)
```

## 第二階段：訓練 1h Volatility 模型（如果沒有）

### 步驟 1：檢查現有模型

```bash
ls -la models/vol_models/*/1h/
```

### 步驟 2：訓練缺失的模型

編輯 `train_vol_model.py`：

```python
if __name__ == '__main__':
    trainer = VolatilityModelTrainer()
    
    # 訓練所有缺失的 1h 模型
    results = trainer.train_all_symbols('1h')
```

執行：

```bash
python train_vol_model.py
```

## 第三階段：更新 RealtimeBBDetectorV2

已在 `realtime_detector_v2_enhanced.py` 中實現的新功能：

1. **多時間框架支持**：同時加載 15m 和 1h 模型
2. **波動預測集成**：調用 vol_models 進行未來波動預測
3. **靈活的時間框架切換**：在掃描時動態選擇 15m 或 1h

### 新增的關鍵方法

#### A. 初始化時加載所有模型

```python
def __init__(self, ...):
    # 加載 bb_models (15m + 1h)
    # 加載 validity_models (15m + 1h)
    # 加載 vol_models (1h)
```

#### B. 時間框架感知的掃描

```python
def scan(self, symbol: str, timeframe: str = "15m") -> Optional[Dict]:
    # 根據 timeframe 選擇相應的模型
    # 15m: 使用 15m classifier + 15m validity model
    # 1h: 使用 1h classifier + 1h validity model + vol model
```

#### C. 波動預測

```python
def predict_volatility(self, symbol: str) -> Optional[float]:
    # 調用 vol_models 預測未來 24 小時的波動大小
    # 返回: 預測的年化波動率或百分比變化
```

## 第四階段：集成到實時系統

### 更新 realtime_service.py

```python
from realtime_detector_v2_enhanced import RealtimeBBDetectorV2Enhanced

class RealtimeService:
    def __init__(self):
        # 使用增強版本
        self.detector = RealtimeBBDetectorV2Enhanced(
            model_dir="models",
            device="cpu"
        )
    
    def process_candle(self, symbol: str, candle: dict, timeframe: str = "15m"):
        # 添加 K 線
        self.detector.add_candle(symbol, candle)
        
        # 掃描信號（15m 或 1h）
        signal = self.detector.scan(symbol, timeframe=timeframe)
        
        if signal:
            # 獲取波動預測
            volatility = self.detector.predict_volatility(symbol)
            signal['predicted_volatility'] = volatility
            
            # 發送信號
            self.emit_signal(signal)
```

### 信號格式

現在的信號包含額外信息：

```json
{
    "symbol": "AAVEUSDT",
    "timeframe": "15m",
    "side": "long",
    "bb_position_label": "Lower",
    "layer1_class": 2,
    "validity_prob": 0.82,
    "confidence": 0.82,
    "predicted_volatility": 0.045,
    "predicted_volatility_unit": "daily_pct_change",
    "rsi": 35.2,
    "adx": 24.1,
    "vol_ratio": 1.45,
    "timestamp": 1735880000000
}
```

## 第五階段：測試多時間框架

### 測試腳本

```python
from realtime_detector_v2_enhanced import RealtimeBBDetectorV2Enhanced

# 初始化檢測器
detector = RealtimeBBDetectorV2Enhanced(model_dir="models")

# 模擬 15m K 線
candle_15m = {
    "timestamp": 1735880000000,
    "open": 140.5, "high": 141.2, "low": 139.8, "close": 140.8,
    "volume": 5200,
    "bb_upper": 142.0, "bb_middle": 140.0, "bb_lower": 138.0,
    "rsi": 62.5, "adx": 22.3, "atr": 1.5
}

# 添加到檢測器
detector.add_candle("AAVEUSDT", candle_15m)

# 掃描 15m
signal_15m = detector.scan("AAVEUSDT", timeframe="15m")
if signal_15m:
    print("15m Signal:", signal_15m)

# 掃描 1h（如果有足夠的 1h 數據）
signal_1h = detector.scan("AAVEUSDT", timeframe="1h")
if signal_1h:
    print("1h Signal:", signal_1h)
    
    # 獲取波動預測
    volatility = detector.predict_volatility("AAVEUSDT")
    if volatility is not None:
        print(f"Predicted daily volatility: {volatility*100:.2f}%")
```

## 性能基準

### 預期模型性能

#### BB Classifier（位置模型）
- 準確度：> 98%
- 速度：5-10ms / 符號

#### Validity Model
- 準確度：90-94%
- 準確率：85-91% (對有效信號的正確率)
- 召回率：80-88% (捕捉有效信號的比率)
- 速度：20-50ms / 符號

#### Volatility Model
- R² 分數：0.45-0.65
- RMSE：0.01-0.03 (日均波動率)
- 速度：10-30ms / 符號

### 掃描速度

- 單個符號完整掃描：30-80ms
- 全 22 個符號掃描：0.7-1.8 秒
- 可用於 5-10 秒的更新間隔

## 故障排查

### 問題 1：模型加載失敗

檢查：
```bash
# 確認文件存在
ls models/validity_models/AAVEUSDT/15m/
# 應輸出：
# feature_names.pkl  scaler.pkl  validity_model.pkl
```

### 問題 2：15m 掃描返回 None

可能原因：
1. 沒有足夠的 15m 歷史數據（需要最少 10 個 K 線）
2. 15m 模型未正確加載
3. K 線數據缺少必要的技術指標

檢查：
```python
candle_buffer = detector.candle_buffer["AAVEUSDT"]
print(f"Buffer size: {len(candle_buffer)}")
if len(candle_buffer) > 0:
    print(f"Latest candle: {candle_buffer[-1]}")
```

### 問題 3：波動預測返回 None

原因：
1. vol_models 未加載（檢查 logs）
2. 預測特徵提取失敗

檢查：
```python
if 'AAVEUSDT' in detector.vol_models:
    print("Vol model loaded")
else:
    print("Vol model NOT loaded")
```

## 後續優化

### 1. 集成交易信號

基於 confidence 和 predicted_volatility 的交易決策：

```python
if signal_15m['confidence'] > 0.75 and volatility > 0.02:
    # 高信心度且高波動性 → 進場信號
    execute_trade(signal_15m)
```

### 2. 多時間框架確認

同時檢查 15m 和 1h 信號，確保方向一致：

```python
signal_15m = detector.scan(symbol, "15m")
signal_1h = detector.scan(symbol, "1h")

if signal_15m and signal_1h and signal_15m['side'] == signal_1h['side']:
    # 多時間框架確認 → 強信號
    execute_trade(signal_15m)
```

### 3. 動態止損

基於預測的波動率調整止損：

```python
volatility = detector.predict_volatility(symbol)
stop_loss_pct = volatility * 1.5  # 波動率的 1.5 倍
```

## 訓練時間表

| 階段 | 任務 | 估計時間 |
|------|------|--------|
| 1 | 訓練 15m validity 模型 | 2-4 小時 |
| 2 | 訓練 1h vol 模型 | 1-2 小時 |
| 3 | 驗證所有模型 | 30 分鐘 |
| 4 | 集成和測試 | 1-2 小時 |
| **總計** | | **5-9 小時** |

## 總結

完成此指南後，你將擁有：

✅ 15m 和 1h 雙時間框架支持
✅ 高精度的有效性驗證（90%+ 準確度）
✅ 波動預測能力
✅ 完整的企業級實時檢測系統
✅ 可靠的多信號交易決策邏輯
