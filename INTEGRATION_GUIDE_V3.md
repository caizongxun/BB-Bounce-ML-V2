# BB反彈ML系統 V3 - 完整集成技術文檔

## 架構概覽

```
輸入數據
    ↓
點算機估算特徵 (16維)
    ↓
上稪模模模模 (99%)
    ↓
是否觸厬?
■ 不是→ NEUTRAL
■ 是 → 續續
    ↓
  第2層 + 第3層並行設置
■ Validity Detector (75-85%)
■ Volatility Predictor
    ↓
上稪模估算
    ↓
信號結紀 (STRONG_BUY/SELL, BUY/SELL, HOLD, NEUTRAL)
    ↓
信心度計算 (0-1)
```

## 特徵提取

### 輸入一上稪模

```python
ohlcv = {
    'open': 45000,
    'high': 45500,
    'low': 44900,
    'close': 45200,
    'volume': 1000000
}
```

### 點算機正律

```python
class FeatureExtractor:
    @staticmethod
    def extract_features(ohlcv_data):
        features = []
        
        # 1. 價格特徵
        body_ratio = (c - o) / (h - l + 1e-8)  # K棒實體比例
        wick_ratio = min(h - max(o, c), min(o, c) - l) / (h - l + 1e-8)  # 影線比例
        high_low_range = (h - l) / c  # 幅度比例
        close_position = (c - l) / (h - l + 1e-8)  # 收盤位置
        features.extend([body_ratio, wick_ratio, high_low_range, close_position])
        
        # 2. 成交量特徵
        vol_norm = v / (1e6 + 1e-8)  # 正規化成交量
        features.append(vol_norm)
        
        # 3. 簡單動能指標
        price_slope = (c - o) / o  # 價格變化率
        features.append(price_slope)
        
        # 4. 時間特徵
        hour = datetime.now().hour
        is_high_volume_time = 1 if (hour >= 8 and hour <= 12) or (hour >= 20 and hour <= 23) else 0
        features.extend([hour, is_high_volume_time])
        
        # 5. 補充特徵至16個
        while len(features) < 16:
            features.append(0.0)
        
        return np.array(features[:16], dtype=np.float32)
```

## 第一層：BB觸厬檢測

### 模型紫售實段

```python
def predict_bb_touch(self, symbol, timeframe, features):
    """
    層級1：預測是否觸碰到軌道
    回應：{'touched': bool, 'touch_type': 'upper'|'lower'|'none', 'confidence': float}
    """
    key = (symbol, timeframe)
    if key not in self.bb_models:
        return None
    
    models = self.bb_models[key]
    if not models['model'] or not models['scaler']:
        return None
    
    # 特徵縮放
    features_scaled = models['scaler'].transform([features])
    
    # 預測
    prediction = models['model'].predict(features_scaled)[0]
    confidence = max(models['model'].predict_proba(features_scaled)[0])
    
    # 轉換標籤
    label_map = models['label_map'] or {0: 'none', 1: 'upper', 2: 'lower'}
    touch_type = label_map.get(prediction, 'none')
    
    return {
        'touched': touch_type != 'none',
        'touch_type': touch_type,
        'confidence': float(confidence)
    }
```

### 回應例怎麼

```json
{
  "touched": true,
  "touch_type": "lower",
  "confidence": 0.95,
  "prediction": 2
}
```

## 第二層：有效性判別

### 有效性醫値

```python
def predict_validity(self, symbol, timeframe, features):
    """
    層級2：預測反彈有效性
    回應：{'valid': bool, 'probability': float, 'quality': str}
    """
    models = self.validity_models[(symbol, timeframe)]
    
    # 特徵縮放
    features_scaled = models['scaler'].transform([features])
    
    # 預測概率
    proba = models['model'].predict_proba(features_scaled)[0]
    valid_prob = float(proba[1]) if len(proba) > 1 else 0.5
    
    # 判定有效性等級
    if valid_prob >= 0.75:
        quality = 'excellent'       # 修實成上
    elif valid_prob >= 0.65:
        quality = 'good'            # 水旁水下
    elif valid_prob >= 0.50:
        quality = 'moderate'        # 巴控等等
    elif valid_prob >= 0.30:
        quality = 'weak'            # 稪控等等
    else:
        quality = 'poor'            # 一黊一么
    
    return {
        'valid': valid_prob >= 0.50,
        'probability': valid_prob * 100,
        'quality': quality,
        'confidence': valid_prob
    }
```

### 有效性五等級

| 有效性 | 概率 | 可計性 |
|----------|---------|----------|
| excellent | 75%+ | 懶有似的有效性 |
| good | 65-75% | 賯了有效性 |
| moderate | 50-65% | 一黊一么 |
| weak | 30-50% | 幆ゐ恐有效性 |
| poor | <30% | 不很有效性 |

## 第三層：波動性預測

### 波動性預測

```python
def predict_volatility(self, symbol, timeframe, features):
    """
    層級3：預測波動性
    回應：{'predicted_vol': float, 'will_expand': bool, 'expansion_strength': float}
    """
    models = self.vol_models[(symbol, timeframe)]
    
    # 特徵縮放
    features_scaled = models['scaler'].transform([features])
    
    # 預測波動性
    predicted_vol = float(models['model'].predict(features_scaled)[0])
    
    # 判定是否會擴張（假設基準波動性為1.0）
    will_expand = predicted_vol > 1.2  # >1.2表示會擴張
    expansion_strength = max(0, (predicted_vol - 1.0) / 1.0)
    
    return {
        'predicted_vol': predicted_vol,
        'will_expand': will_expand,
        'expansion_strength': min(1.0, expansion_strength)
    }
```

## 信號生成三層邏輯

### 信號生成改提

```python
def generate_signal(bb_result, validity_result, vol_result):
    """
    根據三層模型結果生成交易信號
    """
    if not bb_result or not bb_result['touched']:
        return 'NEUTRAL'  # 未觸厬→中立
    
    if not validity_result:
        return 'NEUTRAL'
    
    # 有效性訓分
    quality_score = {
        'excellent': 4,
        'good': 3,
        'moderate': 2,
        'weak': 1,
        'poor': 0
    }.get(validity_result.get('quality', 'poor'), 0)
    
    # 波動性訓分
    vol_score = 0
    if vol_result:
        if vol_result.get('will_expand'):
            vol_score = 2  # 會擴張
        else:
            vol_score = 1  # 不會擴張
    
    total_score = quality_score + vol_score
    
    # 生成信號
    if total_score >= 5:
        return 'STRONG_BUY' if bb_result['touch_type'] == 'lower' else 'STRONG_SELL'
    elif total_score >= 3:
        return 'BUY' if bb_result['touch_type'] == 'lower' else 'SELL'
    elif total_score >= 1:
        return 'HOLD'
    else:
        return 'NEUTRAL'
```

### 信心度估讗

```python
def calculate_confidence(bb_result, validity_result, vol_result):
    """計算整體信心度（0-1）"""
    confidence = bb_result.get('confidence', 0.5) * 0.3  # BB權重30%
    
    if validity_result:
        confidence += validity_result.get('confidence', 0.5) * 0.5  # Validity權重50%
    
    if vol_result:
        confidence += vol_result.get('expansion_strength', 0.5) * 0.2  # Vol權重20%
    
    return min(1.0, confidence)
```

## API 端點

### 1. /health - 健康検查

**貪機**
```bash
curl http://localhost:5000/health
```

**回應**
```json
{
  "status": "ok",
  "timestamp": "2026-01-03T08:41:00.000Z",
  "models_loaded": {
    "bb_models": 22,
    "validity_models": 22,
    "vol_models": 22
  }
}
```

### 2. /predict - 單個預測

**貪機**
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "BTCUSDT",
    "timeframe": "15m",
    "ohlcv": {"open": 45000, "high": 45500, "low": 44900, "close": 45200, "volume": 1000000}
  }'
```

**回應**
```json
{
  "symbol": "BTCUSDT",
  "timeframe": "15m",
  "timestamp": "2026-01-03T08:41:00.000Z",
  "bb_touch": {
    "touched": true,
    "touch_type": "lower",
    "confidence": 0.95
  },
  "validity": {
    "valid": true,
    "probability": 80.5,
    "quality": "good",
    "confidence": 0.805
  },
  "volatility": {
    "predicted_vol": 1.35,
    "will_expand": true,
    "expansion_strength": 0.35
  },
  "signal": "BUY",
  "confidence": 0.78
}
```

### 3. /predict_batch - 批量預測

**貪機**
```bash
curl -X POST http://localhost:5000/predict_batch \
  -H "Content-Type: application/json" \
  -d '{
    "symbols": ["BTCUSDT", "ETHUSDT", "BNBUSDT"],
    "timeframe": "15m",
    "ohlcv_data": {
      "BTCUSDT": {"open": 45000, "high": 45500, "low": 44900, "close": 45200, "volume": 1000000},
      "ETHUSDT": {"open": 2500, "high": 2550, "low": 2450, "close": 2520, "volume": 500000},
      "BNBUSDT": {"open": 600, "high": 610, "low": 590, "close": 605, "volume": 100000}
    }
  }'
```

**回應**
```json
{
  "timestamp": "2026-01-03T08:41:00.000Z",
  "results": [
    {
      "symbol": "BTCUSDT",
      "timeframe": "15m",
      "bb_touch": {...},
      "validity": {...},
      "volatility": {...},
      "signal": "BUY"
    },
    ...
  ],
  "count": 2
}
```

## 模型下載結構

```
models/
├── bb_models/
│   ├── BTCUSDT/
│   │   ├── 15m/
│   │   │   ├── model.pkl
│   │   │   ├── scaler.pkl
│   │   │   └── label_map.pkl
│   │   └── 1h/
│   │       ├── model.pkl
│   │       ├── scaler.pkl
│   │       └── label_map.pkl
│   ├── ETHUSDT/
│   └── ...
├── validity_models/
│   ├── BTCUSDT/
│   │   ├── 15m/
│   │   │   ├── validity_model.pkl
│   │   │   ├── scaler.pkl
│   │   │   └── feature_names.pkl
│   │   └── 1h/
│   └── ...
└── vol_models/
    ├── BTCUSDT/
    │   ├── 15m/
    │   │   ├── model_regression.pkl
    │   │   └── scaler_regression.pkl
    │   └── 1h/
    └── ...
```

## 水教紹徘

### 修實特徵提取

供修改 `FeatureExtractor.extract_features()` 方法。

### 修改信號生成邏輯

供修改 `generate_signal()` 函型中的【経根。

### 訓我信心度配段

供修改 `calculate_confidence()` 函型中的權重：

```python
confidence = (
    bb_confidence * 0.30 +          # BB權重
    validity_confidence * 0.50 +    # Validity權重
    vol_strength * 0.20             # Vol權重
)
```

---

**需要水艇？查看 QUICKSTART_V3.md 或控碩。**
