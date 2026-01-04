# 快速開始指南

## 3 步完成標籤創建

### 第 1 步：創建初始標籤（2 分鐘）

```bash
python label_v3_clean.py
```

**預期輸出：**
```
成功加載 BTCUSDT_15m: 219643 行數據
檢測到 8,523 個觸碰點
標籤統計：
  下軌有效反彈：2,150
  下軌無效反彈：3,000
  上軌有效反彈：1,850
  無觸碰：212,643
整體標籤準確率：87.3%
```

**檢查清單：**
- [ ] 觸碰點數 > 1,000
- [ ] 有效反彈 > 500
- [ ] 準確率 > 80%

---

### 第 2 步：參數調優（10-30 分鐘）

如果第 1 步的準確率 < 90%，運行參數調試：

```bash
python label_parameter_tuning.py
```

**輸出示例：**
```
[1/100] 測試參數：
  touch_threshold=0.02
  lookahead=3
  min_rebound_pct=0.05
  準確率 = 78.5%

[2/100] 測試參數：
  touch_threshold=0.05
  lookahead=5
  min_rebound_pct=0.1
  準確率 = 94.2%  ← 更好

...

前 3 個最優參數組合：
排名 1：touch_threshold=0.05, lookahead=5, min_rebound_pct=0.1, 準確率=94.2%
排名 2：touch_threshold=0.1, lookahead=5, min_rebound_pct=0.15, 準確率=93.8%
排名 3：touch_threshold=0.05, lookahead=7, min_rebound_pct=0.1, 準確率=93.5%
```

**根據最優參數更新代碼：**

編輯 `label_v3_clean.py` 的 `main()` 函數：

```python
def main():
    creator = BBTouchLabelCreator(
        bb_period=20,
        bb_std=2,
        touch_threshold=0.05,      # ← 使用最優參數
        lookahead=5,
        min_rebound_pct=0.1
    )
    labels = creator.run_full_pipeline('BTCUSDT', '15m')
```

---

### 第 3 步：確認最終結果

再次運行更新後的腳本驗證：

```bash
python label_v3_clean.py
```

**目標：**
```
整體標籤準確率：95%+ (最好 > 99%)
```

---

## 下一步：應用到所有幣種

### 修改 `label_v3_clean.py`

找到 `main()` 函數，修改為：

```python
def main():
    Path('logs').mkdir(exist_ok=True)
    Path('outputs/labels').mkdir(parents=True, exist_ok=True)
    
    # 最優參數（根據調優結果調整）
    optimal_params = {
        'bb_period': 20,
        'bb_std': 2,
        'touch_threshold': 0.05,      # ← 調整這些
        'lookahead': 5,
        'min_rebound_pct': 0.1
    }
    
    # 所有 22 個幣種
    symbols = [
        'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'XRPUSDT', 'ADAUSDT',
        'DOGEUSDT', 'MATICUSDT', 'LTCUSDT', 'AVAXUSDT', 'SOLUSDT',
        'ATOMUSDT', 'ARBUSDT', 'OPUSDT', 'UNIUSDT', 'LINKUSDT',
        'FILUSDT', 'ETCUSDT', 'ALGOUSDT', 'AAVEUSDT', 'NEARUSDT',
        'BCHUSDT', 'DOTUSDT'
    ]
    
    timeframes = ['15m', '1h']
    
    for symbol in symbols:
        for timeframe in timeframes:
            try:
                logger.info(f'\n{"="*60}')
                logger.info(f'處理 {symbol} {timeframe}')
                logger.info(f'{"="*60}')
                
                creator = BBTouchLabelCreator(**optimal_params)
                creator.load_data(symbol, timeframe)
                creator.calculate_bb_bands()
                creator.detect_touches()
                labels, label_details = creator.create_labels()
                creator.analyze_rebound_characteristics(label_details)
                creator.save_labels(symbol, timeframe)
                creator.validate_labels_with_backtest()
                
            except Exception as e:
                logger.error(f'處理 {symbol} {timeframe} 失敗：{e}')
                continue
```

### 運行完整數據集

```bash
python label_v3_clean.py
```

**這會生成：**
```
outputs/labels/
├── BTCUSDT_15m_labels.csv
├── BTCUSDT_1h_labels.csv
├── ETHUSDT_15m_labels.csv
├── ETHUSDT_1h_labels.csv
├── ...（共 44 個文件，22 個幣種 × 2 個時間框架）
```

---

## 驗證標籤質量

### 檢查統計信息

對每個標籤文件，檢查：

```python
import pandas as pd

df = pd.read_csv('outputs/labels/BTCUSDT_15m_labels.csv')

# 標籤分佈
print("標籤分佈：")
print(df['label'].value_counts())

# 準確率（應該 > 95%)
valid = df[df['label'].isin([1, 2])]
print(f"有效反彈數：{len(valid)}")
print(f"無效反彈數：{len(df[df['label'] == 0])}")
```

---

## 遇到問題？

### 問題 1：準確率 < 85%

**解決方案：**
1. 試試增加 `min_rebound_pct` (0.1 → 0.15 → 0.2)
2. 減少 `lookahead` (5 → 3)
3. 減少 `touch_threshold` (0.1 → 0.05)

### 問題 2：觸碰點太少（< 500）

**解決方案：**
1. 增加 `touch_threshold` (0.05 → 0.1)
2. 增加 `lookahead` (5 → 7)
3. 減少 `min_rebound_pct` (0.1 → 0.05)

### 問題 3：數據加載失敗

**解決方案：**
1. 檢查 `data/BTCUSDT_15m.csv` 是否存在
2. 確保 CSV 有 open, high, low, close, volume 列

---

## 文件和日誌位置

```
根目錄/
├── label_v3_clean.py              ← 主程序
├── label_parameter_tuning.py       ← 參數調試
├── LABEL_CREATION_GUIDE.md         ← 詳細文檔
├── QUICK_START.md                  ← 本文件
│
├── logs/
│   └── label_creation_*.log        ← 執行日誌
│
└── outputs/
    ├── labels/
    │   ├── BTCUSDT_15m_labels.csv
    │   ├── BTCUSDT_1h_labels.csv
    │   └── ...
    └── parameter_tuning/
        └── parameter_tuning_results.json
```

---

## 預期時間

| 步驟 | 時間 | 說明 |
|------|------|------|
| 第 1 步（初始標籤） | 2-5 分鐘 | BTCUSDT_15m |
| 第 2 步（參數調優） | 10-30 分鐘 | 取決於參數數量 |
| 第 3 步（確認結果） | 2-5 分鐘 | 使用最優參數 |
| 應用到 22 個幣種 | 5-10 分鐘 | 批量處理 |
| **總耗時** | **20-50 分鐘** | 取決於機器性能 |

---

## 成功指標

✓ 準確率 > 95%
✓ 觸碰點 > 1,000
✓ 有效反彈 > 500
✓ 生成了 44 個標籤文件
✓ 日誌中沒有錯誤

---

## 下一步

標籤完成後，可以開始訓練 ML 模型：

```bash
python train_model_on_labels.py
```

（這會在下一步實現）
