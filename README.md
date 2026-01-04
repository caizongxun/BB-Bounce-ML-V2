# BB-Bounce-ML-V2: Bollinger Bands Bounce Detection ML System

Complete automated system for detecting and labeling Bollinger Bands bounce points in cryptocurrency OHLCV data.

## Features

- Automated bounce point detection for 22 cryptocurrencies
- 8-level scoring system (SSS, PEI, MESS, MRMS, BVS)
- 35+ machine learning features
- Support for multiple timeframes (15m, 1h)
- Production-ready code with comprehensive documentation
- 4000+ training samples for ML model development

## Quick Start

### Installation

```bash
git clone https://github.com/caizongxun/BB-Bounce-ML-V2.git
cd BB-Bounce-ML-V2
pip install -r requirements.txt
```

### Basic Usage

```python
from data_labeling_implementation import *

# Download data
df = download_ohlcv_data('BTCUSDT', '15m')
df = calculate_all_indicators(df)

# Label bounce signals
df = label_bounce_signals(df, 'BTCUSDT', direction='long')

# View results
print(df[df['is_bounce_touch']])
```

## Core System

### BVS Score (Bollinger Bands Bounce Validity Score)

```
BVS = MESS*0.25 + SSS*0.25 + PEI*0.25 + MRMS*0.25

Score Ranges:
85-100: Highest probability bounce
70-85:  High probability bounce
55-70:  Medium probability
40-55:  Low probability
0-40:   Very low probability
```

### Scoring Components

1. **SSS** - Support Strength Score
   - Accuracy (40%)
   - Test Count (30%)
   - Multi-test Bonus (30%)

2. **PEI** - Price Exhaustion Index
   - Shadow Ratio (35%)
   - Volume Spike (30%)
   - RSI Extreme (20%)
   - Pattern Factor (15%)

3. **MESS** - Market Environment Suitability Score
   - ADX Strength (30%)
   - BB Width (25%)
   - Price Position (25%)
   - DI Balance (20%)

4. **MRMS** - Mean Reversion Momentum Score
   - RSI Level
   - Momentum Change
   - Acceleration/Deceleration

## Supported Cryptocurrencies

BTCUSDT, ETHUSDT, BNBUSDT, XRPUSDT, ADAUSDT, DOGEUSDT, MATICUSDT, SOLUSDT, AVAXUSDT, FTMUSDT, LINKUSDT, UNIUSDT, LITUSDT, XLMUSDT, DOTUSDT, ATOMUSDT, AXSUSDT, SANDUSDT, MANAUSDT, GRTUSDT, SUSHIUSDT, CRVUSDT

## Documentation

- [DATA_LABELING_README.md](DATA_LABELING_README.md) - Complete technical documentation
- [example_usage.py](example_usage.py) - 7 practical usage examples

## Technical Indicators

The system uses:
- Bollinger Bands (20-period, 2 standard deviations)
- RSI (14-period)
- ATR (14-period)
- ADX (14-period)
- Volume Moving Average (20-period)

## Training Data

- Total samples: 4000+
- Features: 35+
- Timeframes: 15-minute and 1-hour
- Class balance: ~60% success, 40% failure
- Format: Parquet (Pandas compatible)

## Usage Examples

### Single Symbol Processing

```python
df = download_ohlcv_data('BTCUSDT', '15m')
df = calculate_all_indicators(df)
df = label_bounce_signals(df, 'BTCUSDT', direction='long')
```

### Batch Processing

```python
symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']
timeframes = ['15m', '1h']

all_data = process_all_symbols_and_timeframes(symbols, timeframes)
training_df = create_training_dataframe(all_data)
training_df.to_parquet('training_data.parquet')
```

### Data Validation

```python
validate_labeled_data(training_df)
```

## System Requirements

- Python 3.8+
- 4GB+ RAM
- Internet connection (for HuggingFace downloads)

## Dependencies

See requirements.txt for complete list:
- pandas
- numpy
- huggingface_hub
- scikit-learn
- xgboost
- lightgbm

## Performance

- Download 1000 candles: ~5 seconds
- Calculate indicators: ~0.5 seconds
- Label bounce points: ~1 second
- Extract 35+ features: ~0.2 seconds per touch
- Batch process 22 symbols, 2 timeframes: ~3 minutes

## License

MIT License

## Author

Caizong Xun

## Support

GitHub Issues: https://github.com/caizongxun/BB-Bounce-ML-V2/issues

GitHub Discussions: https://github.com/caizongxun/BB-Bounce-ML-V2/discussions
