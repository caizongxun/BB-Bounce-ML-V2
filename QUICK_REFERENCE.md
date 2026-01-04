# BB-Bounce-ML-V2 Quick Reference

## Installation (1 minute)

```bash
git clone https://github.com/caizongxun/BB-Bounce-ML-V2.git
cd BB-Bounce-ML-V2
pip install -r requirements.txt
```

## Basic Usage (3 lines)

```python
from data_labeling_implementation import *
df = download_ohlcv_data('BTCUSDT', '15m')
df = calculate_all_indicators(df)
df = label_bounce_signals(df, 'BTCUSDT', 'long')
```

## Generate Training Data (4 lines)

```python
all_data = process_all_symbols_and_timeframes(
    ['BTCUSDT', 'ETHUSDT'], ['15m']
)
training_df = create_training_dataframe(all_data)
training_df.to_parquet('training.parquet')
```

## API Quick Reference

### download_ohlcv_data(symbol, timeframe)
Download OHLCV data from HuggingFace
- Returns: DataFrame with OHLCV columns

### calculate_all_indicators(df)
Add technical indicators to dataframe
- Returns: DataFrame with 20+ new indicator columns

### label_bounce_signals(df, symbol, direction)
Detect and label bounce points
- Args: df, symbol name, 'long' or 'short'
- Adds: is_bounce_touch, bvs, pei, sss, mess, bounce_success

### extract_features_at_touch(df, idx, direction)
Extract 35+ ML features at specific point
- Returns: dict with 35+ features (all values 0-1)

### create_training_dataframe(all_data)
Convert labeled data to training format
- Args: list of labeled DataFrames
- Returns: training DataFrame ready for ML

### validate_labeled_data(df)
Validate data quality
- Prints: summary report

## Configuration

### Bollinger Bands
```python
BB_PERIOD = 20      # periods
BB_STD = 2          # standard deviations
```

### Touch Detection
```python
TOUCH_THRESHOLD = 1.001    # 0.1% tolerance
TARGET_RATIO = 1.5         # 1.5x band width
```

### Scoring Weights
```python
MESS_WEIGHT = 0.25
SSS_WEIGHT = 0.25
PEI_WEIGHT = 0.25
MRMS_WEIGHT = 0.25
```

## Scores Explained

All scores range 0-1:

**BVS** (Bounce Validity)
- 0.85+: Highest probability
- 0.70-0.85: High probability
- 0.55-0.70: Medium probability
- 0.40-0.55: Low probability
- 0-0.40: Very low probability

**PEI** (Price Exhaustion)
- Measures exhaustion at support
- Higher = more exhaustion signal

**SSS** (Support Strength)
- Measures quality of support level
- Higher = stronger support

**MESS** (Market Environment)
- Measures suitability of conditions
- Higher = better conditions

**MRMS** (Momentum)
- Measures momentum quality
- Higher = better momentum

## Supported Symbols (22)

BTCUSDT, ETHUSDT, BNBUSDT, XRPUSDT, ADAUSDT, DOGEUSDT, MATICUSDT, SOLUSDT, AVAXUSDT, FTMUSDT, LINKUSDT, UNIUSDT, LITUSDT, XLMUSDT, DOTUSDT, ATOMUSDT, AXSUSDT, SANDUSDT, MANAUSDT, GRTUSDT, SUSHIUSDT, CRVUSDT

## Common Tasks

### View bounce statistics
```python
df = label_bounce_signals(df, 'BTCUSDT', 'long')
bounces = df[df['is_bounce_touch']]
print(f"Total: {len(bounces)}")
print(f"Success: {bounces['bounce_success'].sum()}")
print(f"Rate: {bounces['bounce_success'].mean():.1%}")
```

### Find high-quality bounces
```python
high_quality = df[(df['bvs'] > 0.70) & (df['is_bounce_touch'])]
print(f"High quality bounces: {len(high_quality)}")
print(f"Success rate: {high_quality['bounce_success'].mean():.1%}")
```

### Extract features for a touch point
```python
idx = 100  # your touch point
features = extract_features_at_touch(df, idx, 'long')
print(f"BVS: {features['BVS']:.3f}")
print(f"PEI: {features['PEI']:.3f}")
```

### Train ML model
```python
import xgboost as xgb
training_df = pd.read_parquet('training.parquet')
X = training_df.drop('bounce_success', axis=1)
y = training_df['bounce_success']
model = xgb.XGBClassifier().fit(X, y)
```

### Batch process multiple symbols
```python
symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']
data = process_all_symbols_and_timeframes(symbols, ['15m'])
print(f"Processed {len(data)} datasets")
```

## Troubleshooting

**No data downloaded**
- Check internet connection
- Verify symbol spelling
- Check data availability

**No bounce points found**
- Try different symbols
- Use longer timeframe data
- Adjust TOUCH_THRESHOLD

**Too many NaN values**
- Ensure data length >= 200
- BB_PERIOD needs 20+ candles

**Low success rate**
- Analyze feature distributions
- Check parameters
- Validate data quality

## Performance

Typical processing times:
- Download 1000 candles: 5s
- Calculate indicators: 0.5s
- Label bounces: 1s
- Extract features: 0.2s per touch

## File Structure

```
BB-Bounce-ML-V2/
  data_labeling_implementation.py  (720 lines, core)
  example_usage.py                 (417 lines, examples)
  requirements.txt                 (dependencies)
  README.md                        (overview)
  DATA_LABELING_README.md          (full docs)
  DEPLOYMENT_GUIDE.md              (deployment)
  QUICK_REFERENCE.md               (this file)
```

## Links

- GitHub: https://github.com/caizongxun/BB-Bounce-ML-V2
- Dataset: https://huggingface.co/datasets/zongowo111/v2-crypto-ohlcv-data
- PyPI: pip install -r requirements.txt

## License

MIT License - See repository for details
