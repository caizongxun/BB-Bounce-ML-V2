# BB 反彈標籤創建指南

## 核心邏輯

### 第 1 層：識別觸碰點
```
K 棒觸碰下軌或上軌 (有可配置的閾值)
┌─ 下軌觸碰 (low 接近 lower_band)
│   └─ 距離 < threshold% × BB_width
└─ 上軌觸碰 (high 接近 upper_band)
    └─ 距離 < threshold% × BB_width
```

### 第 2 層：判斷後續反彈
```
在觸碰點後檢查後續 N 根 K 棒
┌─ 下軌觸碰 → 檢查是否有上漲
│   └─ 最高價 - 觸碰價 >= min_rebound_pct
└─ 上軌觸碰 → 檢查是否有下跌
    └─ 觸碰價 - 最低價 >= min_rebound_pct
```

### 第 3 層：標籤定義
```
標籤：
1  = 觸碰下軌 + 有上漲反彈 ✓
0  = 觸碰下軌 + 無上漲反彈 ✗
2  = 觸碰上軌 + 有下跌反彈 ✓
-1 = 無觸碰 (忽略)
```

### 第 4 層：驗證準確率
```
在有效反彈點進行回測
┌─ 下軌做多 (label=1)
│   └─ 5根K棒後價格 > 進場價 → 準確
└─ 上軌做空 (label=2)
    └─ 5根K棒後價格 < 進場價 → 準確
```

---

## 快速開始

### 步驟 1：創建基本標籤

```bash
python label_v3_clean.py
```

**輸出示例：**
```
成功加載 BTCUSDT_15m: 219643 行數據
BB 計算完成
檢測到 8,523 個觸碰點

標籤統計：
  下軌有效反彈 (label=1)：2,150
  下軌無效反彈 (label=0)：3,000
  上軌有效反彈 (label=2)：1,850
  無觸碰 (label=-1)：212,643
  總有效反彈：4,000
  總無效反彈：4,523
  有效率：46.9%

有效反彈統計：
  平均幅度：0.2341%
  中位幅度：0.1850%
  
無效反彈統計：
  平均幅度：0.0512%
  中位幅度：0.0245%

整體標籤準確率：87.3%
  目標：99%
```

### 步驟 2：調試參數找到最優組合

```bash
python label_parameter_tuning.py
```

**這會測試多個參數組合：**
- `touch_threshold`: 0.02, 0.05, 0.1, 0.15, 0.2
- `lookahead`: 3, 5, 7, 10
- `min_rebound_pct`: 0.05, 0.1, 0.15, 0.2

**找出準確率最高的組合**

---

## 參數詳解

### `touch_threshold` (觸碰閾值)
```
定義：K 棒到 BB 軌道的最大距離
單位：BB 寬度的百分比
範圍：0.02 ~ 0.2 (2% ~ 20%)

示例：
- touch_threshold = 0.05 (5%)
  └─ 只標記 low 到 lower_band 距離 < 5% BB寬度的 K 棒
  
建議：
- 0.02：非常嚴格，觸碰點少 (~1-2%)
- 0.05：中等嚴格，平衡點 (推薦起點)
- 0.1：寬鬆，觸碰點多 (~5-10%)
```

### `lookahead` (後續驗證週期)
```
定義：觸碰點後檢查多少根 K 棒
單位：K 棒數
範圍：3 ~ 10

示例：
- lookahead = 5
  └─ 在觸碰點後的 5 根 K 棒內檢查反彈

建議：
- 3：短期反彈
- 5：中期反彈 (推薦起點，對應 15m = 75分鐘)
- 7：較長期反彈
- 10：長期反彈
```

### `min_rebound_pct` (最小反彈幅度)
```
定義：觸碰點後的最小反彈百分比
單位：百分比 (%)
範圍：0.05 ~ 0.2

示例：
- min_rebound_pct = 0.1 (0.1%)
  └─ 價格必須反彈至少 0.1% 才算有效

建議：
- 0.05：非常敏感，容易標記為有效
- 0.1：平衡點 (推薦起點)
- 0.15：較嚴格
- 0.2：非常嚴格，只捕捉強反彈
```

---

## 調試策略

### 如果準確率太低 (< 85%)

```
問題：標籤定義過於寬鬆，太多假反彈

解決方案：
1. 增加 min_rebound_pct (從 0.1 → 0.15 或 0.2)
   └─ 只標記明顯的反彈
   
2. 減少 lookahead (從 5 → 3)
   └─ 更短的時間內要出現反彈
   
3. 減少 touch_threshold (從 0.1 → 0.05)
   └─ 只標記真正接近的觸碰
```

### 如果準確率很高但觸碰點太少 (< 500)

```
問題：標籤定義過於嚴格，丟失了很多機會

解決方案：
1. 減少 min_rebound_pct (從 0.15 → 0.1 或 0.05)
   └─ 允許更小的反彈
   
2. 增加 lookahead (從 5 → 7 或 10)
   └─ 給更多時間出現反彈
   
3. 增加 touch_threshold (從 0.05 → 0.1)
   └─ 標記更多接近的觸碰
```

### 目標：平衡準確率和樣本量

```
理想情況：
- 準確率：> 95% (目標 99%)
- 觸碰點：> 1,000
- 有效反彈：> 500

調整流程：
1. 從推薦參數開始
   └─ touch_threshold=0.05, lookahead=5, min_rebound=0.1
   
2. 查看結果
   ├─ 如果準確率 < 90% → 增加 min_rebound 或減少 lookahead
   ├─ 如果準確率 > 98% 但樣本少 → 減少 min_rebound
   └─ 如果樣本少 < 500 → 增加 touch_threshold
   
3. 重新運行 label_v3_clean.py 驗證
```

---

## 輸出文件

### 標籤 CSV 文件
```
outputs/labels/BTCUSDT_15m_labels.csv

列：
- time: K 棒時間
- open, high, low, close, volume: K 線數據
- upper_band, lower_band, mid_band: BB 軌道
- bb_width: BB 寬度
- label: 標籤 (1, 0, 2, -1)
```

### 參數調試結果
```
outputs/parameter_tuning/parameter_tuning_results.json

記錄所有測試的參數和結果，便於分析最優組合
```

---

## 驗證準確率的含義

### 標籤準確率 99% 表示什麼？

```
實驗：在有效反彈點進行交易，檢查後續是否盈利

99% 準確率 = 99 個有效反彈中，99 個在後續 5 根 K 棒內盈利

這說明：
✓ 標籤邏輯完全正確
✓ 反彈的定義與市場現實相符
✓ 用來訓練 ML 模型會非常有效
```

### 如何達到 99%？

```
1. 確保標籤邏輯清晰
   └─ 定義"有效反彈"的方式要符合市場現實
   
2. 參數調優
   └─ 選擇能區分有效/無效反彈的參數
   
3. 檢查數據質量
   └─ K 線數據是否完整和準確？
```

---

## 常見問題

### Q: 為什麼觸碰點這麼多？
A: 檢查 `touch_threshold` 是否過大。試試減少到 0.05 或 0.02。

### Q: 為什麼有效反彈比例很低 (< 30%)?
A: 這是正常的！不是所有觸碰都會反彈。可以試試增加 `lookahead` 給更多時間。

### Q: 標籤準確率只有 70%，怎麼辦？
A: 
1. 檢查回測邏輯是否正確
2. 試試調整 `min_rebound_pct`
3. 查看圖表，手動檢查幾個例子

### Q: 如何確定參數是否最優？
A: 運行 `label_parameter_tuning.py` 會自動測試並找到準確率最高的組合。

---

## 下一步

準確率達到 > 95% 後：

1. **擴展到其他幣種**
   ```bash
   # 修改 label_v3_clean.py 中的 main() 函數
   symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', ...]
   timeframes = ['15m', '1h']
   ```

2. **訓練 ML 模型**
   ```bash
   # 使用標籤訓練預測模型
   python train_v3_model.py
   ```

3. **實盤驗證**
   ```bash
   # 在實時數據上驗證標籤準確率
   python live_validation.py
   ```

---

## 技術細節

### 觸碰檢測算法

```python
# 計算 K 棒到軌道的距離
dist_to_lower = (close - lower_band) / bb_width

# 觸碰判定
if dist_to_lower < touch_threshold:
    mark_as_touch('lower')
```

### 反彈驗證算法

```python
# 下軌反彈
future_prices = prices[i+1:i+lookahead+1]
max_price = future_prices.max()
rebound_amount = max_price - touch_price
rebound_pct = rebound_amount / touch_price * 100

if rebound_pct >= min_rebound_pct:
    label = 1  # 有效反彈
else:
    label = 0  # 無效反彈
```

### 準確率計算

```python
# 回測邏輯
correct_count = 0
for each_valid_rebound:
    entry_price = touch_price
    close_price = prices[touch_idx + lookahead]
    
    if close_price > entry_price:
        correct_count += 1
    
accuracy = correct_count / total_valid_rebounds * 100
```
