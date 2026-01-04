# Data Labeling System - Complete Technical Documentation

## Table of Contents

1. System Overview
2. Installation and Setup
3. Quick Start
4. Complete API Reference
5. Feature Engineering Details
6. Configuration Guide
7. Troubleshooting
8. Advanced Usage

## 1. System Overview

The BB-Bounce-ML-V2 system is a complete pipeline for automated detection and labeling of Bollinger Bands bounce points in cryptocurrency OHLCV data.

### Core Components

1. **Data Acquisition** - Download from HuggingFace datasets
2. **Indicator Calculation** - 20+ technical indicators
3. **Touch Detection** - Identify Bollinger Band touches
4. **Score Calculation** - Multi-level scoring system
5. **Feature Extraction** - 35+ ML features
6. **Training Set Generation** - Standardized output format
7. **Batch Processing** - Multi-symbol pipeline
8. **Data Validation** - Quality assurance checks

### Supported Assets

Symbols: 22 cryptocurrencies
Timeframes: 15-minute, 1-hour
Data points: 4.8+ million

## 2. Installation and Setup

### Environment Requirements

- Python 3.8 or higher
- 4GB RAM minimum
- Internet connection

### Installation Steps

```bash
git clone https://github.com/caizongxun/BB-Bounce-ML-V2.git
cd BB-Bounce-ML-V2
pip install -r requirements.txt
```

### Verify Installation

```python
from data_labeling_implementation import download_ohlcv_data
df = download_ohlcv_data('BTCUSDT', '15m')
print(f"Successfully loaded {len(df)} candles")
```

## 3. Quick Start

### Basic Workflow (3 Steps)

```python
from data_labeling_implementation import *

# Step 1: Download and calculate
df = download_ohlcv_data('BTCUSDT', '15m')
df = calculate_all_indicators(df)

# Step 2: Label bounces
df = label_bounce_signals(df, 'BTCUSDT', direction='long')

# Step 3: View results
print(df[df['is_bounce_touch']][['close', 'bb_lower', 'bvs', 'bounce_success']])
```

### Generate Training Set (4 Steps)

```python
from data_labeling_implementation import *

# Step 1: Configure symbols
symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']
timeframes = ['15m']

# Step 2: Batch process
all_data = process_all_symbols_and_timeframes(symbols, timeframes)

# Step 3: Create training DataFrame
training_df = create_training_dataframe(all_data)

# Step 4: Save
training_df.to_parquet('training_data.parquet')
```

## 4. Complete API Reference

### Function: download_ohlcv_data

```python
download_ohlcv_data(symbol, timeframe)
```

Downloads OHLCV data from HuggingFace.

Args:
  - symbol: str, one of 22 supported symbols
  - timeframe: str, '15m' or '1h'

Returns:
  - DataFrame with columns: open, high, low, close, volume
  - Index: datetime timestamps

Example:
```python
df = download_ohlcv_data('BTCUSDT', '15m')
print(f"Downloaded {len(df)} candles from {df.index[0]} to {df.index[-1]}")
```

### Function: calculate_all_indicators

```python
calculate_all_indicators(df)
```

Calculates all technical indicators.

Adds columns:
  - bb_upper, bb_middle, bb_lower, bb_width, bb_pct
  - rsi
  - atr
  - adx, di_plus, di_minus
  - volume_ma, volume_ratio

Example:
```python
df = calculate_all_indicators(df)
print(f"Indicators calculated. New columns: {df.columns.tolist()}")
```

### Function: label_bounce_signals

```python
label_bounce_signals(df, symbol, direction='long')
```

Labels all bounce points and calculates scores.

Args:
  - df: DataFrame with indicators
  - symbol: str, symbol name for reference
  - direction: str, 'long' or 'short'

Adds columns:
  - is_bounce_touch: bool, True if touch detected
  - bounce_success: int, 1 if target reached
  - sss, pei, mess, mrms, bvs: float, scores 0-1
  - target_price: float, target level
  - bars_to_target: int, bars to reach target

Example:
```python
df = label_bounce_signals(df, 'BTCUSDT', direction='long')
print(f"Found {df['is_bounce_touch'].sum()} bounce points")
print(f"Success rate: {df[df['is_bounce_touch']]['bounce_success'].mean():.1%}")
```

### Function: extract_features_at_touch

```python
extract_features_at_touch(df, idx, direction='long')
```

Extracts 35+ ML features at a specific point.

Args:
  - df: DataFrame with indicators
  - idx: int, row index
  - direction: str, 'long' or 'short'

Returns:
  - dict with 35+ feature names and values (0-1 range)

Example:
```python
idx = 100
features = extract_features_at_touch(df, idx, 'long')
print(f"BVS: {features['BVS']:.3f}")
print(f"PEI: {features['PEI']:.3f}")
```

### Function: create_training_dataframe

```python
create_training_dataframe(all_labeled_data)
```

Converts labeled data to standard training format.

Args:
  - all_labeled_data: list of DataFrames

Returns:
  - DataFrame with 35+ feature columns and bounce_success target

Example:
```python
training_df = create_training_dataframe([df1, df2, df3])
print(f"Training set: {len(training_df)} samples, {len(training_df.columns)} features")
```

### Function: validate_labeled_data

```python
validate_labeled_data(df)
```

Performs comprehensive quality checks.

Checks:
  - Shape and size
  - Missing values
  - Value ranges (0-1)
  - Statistical properties
  - Class balance

Example:
```python
validate_labeled_data(training_df)
```

### Function: process_all_symbols_and_timeframes

```python
process_all_symbols_and_timeframes(symbols, timeframes)
```

Complete pipeline for multiple symbols.

Args:
  - symbols: list of str
  - timeframes: list of str

Returns:
  - list of labeled DataFrames

Example:
```python
data = process_all_symbols_and_timeframes(
    ['BTCUSDT', 'ETHUSDT'],
    ['15m', '1h']
)
print(f"Processed {len(data)} datasets")
```

## 5. Feature Engineering Details

### Level 1: Environment Features (6)

- MESS: Market Environment Suitability Score (0-1)
- ADX: Average Directional Index (0-1, normalized)
- DI_Plus: Positive Directional Indicator (0-1)
- DI_Minus: Negative Directional Indicator (0-1)
- BB_Width: Bollinger Band width ratio (0-1)
- Price_Position: Price position within bands (0-1)

### Level 2: Support Features (5)

- SSS: Support Strength Score (0-1)
- Support_Tests: Number of support tests (0-1)
- Distance_to_Support: Distance from current to support (0-1)
- Previous_Bounce: Height of previous bounce (0-1)
- Support_Quality: Quality rating of support (0-1)

### Level 3: Exhaustion Signals (8)

- PEI: Price Exhaustion Index (0-1)
- Shadow_Ratio: Lower/upper shadow ratio (0-1)
- Volume_Ratio: Volume spike ratio (0-1)
- RSI_14: RSI value normalized (0-1)
- RSI_Divergence: RSI divergence indicator (0-1)
- Vol_Spike: Volume spike intensity (0-1)
- Body_Ratio: Candle body to range ratio (0-1)
- Shape_Factor: Overall shape quality (0-1)

### Level 4: Momentum (7)

- MRMS: Mean Reversion Momentum Score (0-1)
- RSI_Level: RSI extreme level (0-1)
- Momentum_Change: Rate of momentum change (0-1)
- MOM_9: 9-period momentum (0-1)
- MOM_20: 20-period momentum (0-1)
- Acceleration: Momentum acceleration (0-1)
- Deceleration: Momentum deceleration (0-1)

### Level 5: K-Line Pattern (6)

- Color_Sequence: Color sequence pattern (0-1)
- Body_to_Range: Body to total range (0-1)
- Engulfing: Engulfing pattern score (0-1)
- Inside_Bar: Inside bar pattern (0-1)
- Upper_Shadow: Upper shadow percentage (0-1)
- Lower_Shadow: Lower shadow percentage (0-1)

### Level 6: Composite Scores (3)

- BVS: Bollinger Bands Bounce Validity Score (0-1)
- SBVS: Short-side BVS (opposite for shorts)
- Agreement: Consensus of multiple indicators (0-1)

## 6. Configuration Guide

### Bollinger Bands

```python
BB_PERIOD = 20  # number of periods
BB_STD = 2      # standard deviation multiplier
```

### Touch Detection

```python
TOUCH_THRESHOLD = 1.001  # allow 0.1% deviation
MIN_BARS_TO_CONFIRM = 5  # minimum bars to confirm
TARGET_RATIO = 1.5       # target is 1.5x band width
```

### Scoring Weights

```python
MESS_WEIGHT = 0.25
SSS_WEIGHT = 0.25
PEI_WEIGHT = 0.25
MRMS_WEIGHT = 0.25
```

Modify weights in code:

```python
# Edit in data_labeling_implementation.py
MESS_WEIGHT = 0.30  # increase environment factor
SSS_WEIGHT = 0.25
PEI_WEIGHT = 0.20   # decrease exhaustion
MRMS_WEIGHT = 0.25
```

## 7. Troubleshooting

### No Data Downloaded

Check:
- Internet connection
- Symbol spelling (must be exact)
- Data availability on HuggingFace

### Too Many NaN Values

Cause: Insufficient data for indicator calculation

Solution:
```python
# Use longer data
df = download_ohlcv_data('BTCUSDT', '15m')
if len(df) < 200:
    print("Not enough data for 20-period indicators")
```

### No Bounce Points Found

Cause: Price didn't touch bands in dataset

Try:
- Different symbols
- Longer time range
- Adjust TOUCH_THRESHOLD

### Low Success Rate

Cause: Parameters need tuning

Investigate:
```python
df = df[df['is_bounce_touch']]
print(df[['bvs', 'pei', 'sss', 'bounce_success']].groupby('bounce_success').mean())
```

## 8. Advanced Usage

### Custom Scoring

```python
def custom_score(df, idx):
    # Your custom logic here
    return score_value

# Override in label_bounce_signals
```

### Real-Time Application

```python
def evaluate_current_candle(ohlcv_dict):
    df = pd.DataFrame([ohlcv_dict])
    df = calculate_all_indicators(df)
    features = extract_features_at_touch(df, 0, 'long')
    return features['BVS']
```

### Model Training

```python
import xgboost as xgb

training_df = pd.read_parquet('training_data.parquet')
X = training_df.drop('bounce_success', axis=1)
y = training_df['bounce_success']

model = xgb.XGBClassifier()
model.fit(X, y)
```

## Conclusion

This system provides production-ready data labeling for Bollinger Bands bounce trading research. All components are modular, customizable, and well-documented.

For support and questions, visit GitHub:
https://github.com/caizongxun/BB-Bounce-ML-V2
