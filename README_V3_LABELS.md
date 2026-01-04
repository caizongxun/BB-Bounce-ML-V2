# BB Bounce ML V3 - 標籤創建系統

這是針對 **BB-Bounce-ML-V2** 的一個重大改進版本，主要改進了標籤創建邏輯。

## 新功能特性

### V3 改進點

1. **新的標籤邏輯**
   - 基於真實 BB 通道觸碰事件
   - 自動檢測有效/無效反彈
   - 支持上軌和下軌交易邏輯

2. **參數調優工具**
   - 自動測試多個參數組合
   - 找到最優的標籤參數
   - 目標準確率 99%

3. **驗證機制**
   - 回測驗證標籤準確率
   - 統計分析反彈特徵
   - 自動生成驗證報告

---

## 快速開始

### 第 1 步：安裝依賴

```bash
pip install -r requirements.txt
```

### 第 2 步：準備數據

確保你的 `data/` 目錄中有 CSV 文件：
```
data/
├── BTCUSDT_15m.csv
├── BTCUSDT_1h.csv
├── ETHUSDT_15m.csv
├── ETHUSDT_1h.csv
└── ... (其他幣種)
```

CSV 格式要求：
```
time,open,high,low,close,volume
```

### 第 3 步：創建初始標籤

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
  ...
整體標籤準確率：87.3%
```

### 第 4 步：參數調優（可選）

如果準確率 < 90%，運行參數調優：

```bash
python label_parameter_tuning.py
```

這會測試 100+ 個參數組合，找出最優方案。

### 第 5 步：查看結果

標籤文件位置：
```
outputs/labels/
├── BTCUSDT_15m_labels.csv
├── BTCUSDT_1h_labels.csv
├── ETHUSDT_15m_labels.csv
├── ETHUSDT_1h_labels.csv
└── ...
```

---

## 文件說明

| 文件 | 說明 |
|------|------|
| `label_v3_clean.py` | 主程序，創建標籤 |
| `label_parameter_tuning.py` | 參數調優工具 |
| `LABEL_CREATION_GUIDE.md` | 詳細文檔 |
| `QUICK_START.md` | 快速開始指南 |
| `.gitignore` | Git 忽略配置 |
| `requirements.txt` | Python 依賴 |

---

## 標籤定義

```
1  = 觸碰下軌 + 有上漲反彈 ✓
0  = 觸碰下軌 + 無上漲反彈 ✗
2  = 觸碰上軌 + 有下跌反彈 ✓
-1 = 無觸碰 (忽略)
```

---

## 核心參數

| 參數 | 默認值 | 範圍 | 說明 |
|------|-------|------|------|
| `touch_threshold` | 0.05 | 0.02-0.2 | K 棒到軌道的最大距離 (%) |
| `lookahead` | 5 | 3-10 | 後續驗證的 K 棒數 |
| `min_rebound_pct` | 0.1 | 0.05-0.2 | 最小反彈幅度 (%) |

---

## 工作流程

```
1. 加載 K 線數據
   ↓
2. 計算 Bollinger Bands
   ↓
3. 檢測觸碰點 (使用 touch_threshold)
   ↓
4. 驗證後續反彈 (使用 lookahead 和 min_rebound_pct)
   ↓
5. 分配標籤 (1, 0, 2, -1)
   ↓
6. 回測驗證準確率
   ↓
7. 保存標籤 CSV
```

---

## 驗證準確率

系統會自動進行回測：

```
下軌反彈做多（應該上漲）：
  信號數：2,150
  盈利信號數：1,998
  勝率：92.8%

上軌反彈做空（應該下跌）：
  信號數：1,850
  盈利信號數：1,753
  勝率：94.8%

整體標籤準確率：93.7%
```

---

## 預期性能

| 階段 | 準確率 |
|------|--------|
| 初始（推薦參數） | 85-90% |
| 調優後 | > 95% |
| 目標 | 99% |

---

## 使用建議

1. **第一次運行**
   - 使用推薦參數運行 `label_v3_clean.py`
   - 檢查日誌中的準確率

2. **如果準確率 < 90%**
   - 運行 `label_parameter_tuning.py` 找最優參數
   - 查看 `outputs/parameter_tuning/parameter_tuning_results.json`
   - 更新 `label_v3_clean.py` 中的參數

3. **批量處理**
   - 修改 `main()` 函數中的 symbols 列表
   - 運行完整的 22 個幣種批量處理

---

## 故障排除

### 問題：標籤準確率很低 (< 70%)

```bash
# 檢查標籤邏輯
python label_v3_clean.py

# 查看日誌
tail logs/label_creation_*.log

# 手動檢查幾個觸碰點的圖表
```

### 問題：觸碰點太少 (< 500)

調整參數：
- 增加 `touch_threshold` (0.05 → 0.1)
- 增加 `lookahead` (5 → 7)
- 減少 `min_rebound_pct` (0.1 → 0.05)

### 問題：執行很慢

```bash
# 使用縮減的數據集測試
# 編輯 label_v3_clean.py，只加載前 100,000 行
```

---

## 下一步

標籤完成後，可以開始訓練 ML 模型：

```bash
python train_model_on_labels.py
```

（這會在下一步實現）

---

## 更多資訊

詳見：
- `LABEL_CREATION_GUIDE.md` - 詳細技術文檔
- `QUICK_START.md` - 快速開始指南

---

## 版本歷史

### V3 (2026-01-04)
- 完全重新設計標籤邏輯
- 基於真實 BB 觸碰事件
- 自動參數調優工具
- 支持準確率驗證

### V2 (之前版本)
- 初始實現
- 22 個幣種支持

---

## 聯繫與支持

如有問題，請查閱：
1. 執行日誌 (`logs/`)
2. 驗證報告 (`outputs/`)
3. 技術文檔 (`.md` 文件)

---

**目標：達到 99% 的標籤準確率，為 ML 模型提供最優質的訓練數據。**
