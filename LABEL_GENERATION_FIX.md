# 檔案找不到原因分析與解決方案

## 問題診斷

您的 `check_available_symbols.py` 腳本尋找的位置:
```
outputs/labels/ETHUSDT_15m_profitability_v2.csv
```

但是您刊下載的檔案位置:
```
HuggingFace 資料集:
- datasets--zongowo111--v2-crypto-ohlcv-data\snapshots\cefd0985c92ea40d154beb2e30861bd083cedc55\klines\...

CSV 檔案位置（下載完成）:
- AAVEUSDT_15m.csv (9.07 MB)
- AAVEUSDT_1h.csv (2.30 MB)
- ADAUSDT_15m.csv (13.72 MB)
...
```

**根本原因:** 這些是 OHLCV（原始市場數據）檔案，但您的訓練腳本需要的是已標籤化的 profitability CSV 檔案。

## 解決步驟

### 步驟 1: 找出您的 CSV 檔案位置

```bash
python find_csv_files.py
```

這將掃描您的專案目錄並告訴您:
- 找到多少個 CSV 檔案
- 它們的確切位置
- 建議的基礎目錄

### 步驟 2: 生成標籤

這時執行標籤生成腳本:

```bash
python label_generation_fix.py
```

這會:
1. 找到您的 `*_15m.csv` OHLCV 檔案
2. 計算 Bollinger Bands 指標
3. 識別有效的 BB 反彈點
4. 在 `outputs/labels/` 目錄生成標籤檔案
5. 列出詳細的統計信息

### 步驟 3: 驗證生成

執行検查腳本:

```bash
python check_available_symbols.py
```

現在應該看到類似的結果:

```
✅ BTCUSDT      -   14.1MB - 219643 行 (有效:  31346)
✅ ETHUSDT      -   12.5MB - 195234 行 (有效:  28745)
✅ BNBUSDT      -   11.2MB - 176543 行 (有效:  26500)
... 更多交易對
```

### 步驟 4: 開始訓練

一旦標籤生成完成（1/23 → 23/23），執行訓練:

```bash
# 訓練 BB 位置分類器
python trainbbmodel.py

# 訓練成交量預測模理
python trainvolmodel.py

# 訓練有效性驗證器
python trainvaliditymodel.py
```

## 目錄結構參考

標籤生成應該建立這個結構:

```
your-project/
├── data/
│   ├── BTCUSDT_15m.csv        ← 下載的 OHLCV 檔案
│   ├── ETHUSDT_15m.csv
│   ├── ADAUSDT_15m.csv
│   └── ... (更多交易對)
│
├── outputs/
│   └── labels/
│       ├── BTCUSDT_15m_profitability_v2.csv    ← 生成的標籤
│       ├── ETHUSDT_15m_profitability_v2.csv
│       ├── ADAUSDT_15m_profitability_v2.csv
│       └── ... (更多標籤檔案)
│
├── models/
│   ├── bbmodels/               ← BB 模型
│   ├── validitymodels/         ← 有效性模型
│   └── volmodels/              ← 成交量模型
│
└── label_generation_fix.py     ← 執行這個
```

## 標籤生成說明

### 標籤定義

- **有效反彈 (1)**: 價格觸及 BB 上/下軌後，在規定蠟燭數內有足夠反彈
- **失敗反彈 (0)**: 價格觸及 BB 上/下軌但反彈不足
- **未標籤 (NaN)**: 未達成交条件的蠟燭

### 參數設置

```python
self.min_bounce_pct = 0.5           # 最小反彈百分比 (%)
self.min_candles_recovery = 5       # 最少恢復蠟燭數
self.lookback_bars = 50             # 初始化期間
```

可以在 `label_generation_fix.py` 中調整這些參數。

## 常見問題

### Q: 為什麼只有 1 個檔案有效?
A: 因為標籤檔案還沒生成。執行 `label_generation_fix.py` 先生成標籤。

### Q: CSV 檔案在哪裡?
A: 執行 `find_csv_files.py` 找出確切位置，然後修改 `label_generation_fix.py` 中的 `data_dir` 參數。

### Q: 標籤生成需要多久?
A: 約 2-4 分鐘（23 個交易對）

### Q: 我可以自訂義標籤邏輯吗?
A: 可以，修改 `_generate_trade_labels()` 方法在 `label_generation_fix.py` 中。

## 下一步

標籤生成完成後:

1. **訓練模型**
   ```bash
   python trainbbmodel.py        # 3-5 分鐘
   python trainvolmodel.py       # 2-3 分鐘
   python trainvaliditymodel.py  # 2-3 分鐘
   ```

2. **啟動實時服勑**
   ```bash
   python realtimeservicev3.py
   ```

3. **訪問冶表板**
   - 打開瀏覽器: http://localhost:5000

## 完整執行流程

```bash
# 1. 找到 CSV 檔案位置
python find_csv_files.py

# 2. 生成標籤（這會生成 profitability_v2.csv 檔案）
python label_generation_fix.py

# 3. 驗證標籤
python check_available_symbols.py

# 4. 訓練模型
python trainbbmodel.py
python trainvolmodel.py
python trainvaliditymodel.py

# 5. 啟動服勑
python realtimeservicev3.py

# 6. 在另一個終端測試
curl http://localhost:5000/health
```

## 技術細節

### 標籤生成演算法

1. **識別進入點**
   - 検查價格是否觸及 BB 上軌（高於 BB_Upper * 0.99）
   - 或觸及 BB 下軌（低於 BB_Lower * 1.01）

2. **渫量潜在反彈**
   - 查看接下來 5 根蠟燭的高點和低點
   - 計算從最低點到最高點的百分比變化

3. **決定標籤**
   - 如果反彈 >= 0.5%: 標籤為 1 (有效交易)
   - 如果反彈 < 0.5%: 標籤為 0 (失敗反彈)

### 特徵計算

標籤檔案還包含:
- Bollinger Bands 指標 (SMA, STD, 上軌, 下軌)
- 反彈類型 (lower_bounce, upper_bounce, none)
- 反彈百分比

這些特徵將被用於訓練 ML 模型。

## 獲得幫助

如果遇到問題:

1. 検查 CSV 檔案是否存在
   ```bash
   python find_csv_files.py
   ```

2. 検查錯誤日誌
   ```bash
   # label_generation_fix.py 會輸出詳細日誌
   python label_generation_fix.py 2>&1 | tee label_generation.log
   ```

3. 驗證資料品質
   ```python
   import pandas as pd
   df = pd.read_csv('data/BTCUSDT_15m.csv')
   print(df.head())
   print(df.info())
   print(df.describe())
   ```
