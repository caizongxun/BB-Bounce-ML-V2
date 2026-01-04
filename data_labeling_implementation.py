"""
Bollinger Bands Bounce Detection - Data Labeling System
Complete implementation with performance optimizations

Author: Caizong Xun
Version: 1.0.2
Date: 2026-01-04
"""

import pandas as pd
import numpy as np
from huggingface_hub import hf_hub_download
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

# ==================== CONFIGURATION ====================

SUPPORTED_SYMBOLS = [
    'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'XRPUSDT', 'ADAUSDT',
    'DOGEUSDT', 'MATICUSDT', 'SOLUSDT', 'AVAXUSDT', 'FTMUSDT',
    'LINKUSDT', 'UNIUSDT', 'LITUSDT', 'XLMUSDT', 'DOTUSDT',
    'ATOMUSDT', 'AXSUSDT', 'SANDUSDT', 'MANAUSDT', 'GRTUSDT',
    'SUSHIUSDT', 'CRVUSDT'
]

SUPPORTED_TIMEFRAMES = ['15m', '1h']

# Technical indicator parameters
BB_PERIOD = 20
BB_STD = 2
RSI_PERIOD = 14
ATR_PERIOD = 14
ADX_PERIOD = 14
VOLUME_MA_PERIOD = 20

# Bounce detection parameters (OPTIMIZED)
TOUCH_THRESHOLD = 1.0001  # Reduced from 1.001 (0.01% vs 0.1%)
MIN_BARS_TO_CONFIRM = 5
TARGET_RATIO = 1.5
MIN_BOUNCE_QUALITY = 0.45  # Filter low-quality bounces

# Scoring weights
MESS_WEIGHT = 0.25
SSS_WEIGHT = 0.25
PEI_WEIGHT = 0.25
MRMS_WEIGHT = 0.25

# ==================== MODULE 1: DATA ACQUISITION ====================

def download_ohlcv_data(symbol, timeframe):
    """
    Download OHLCV data from HuggingFace
    
    Args:
        symbol: Cryptocurrency symbol (e.g., 'BTCUSDT')
        timeframe: Time frame ('15m' or '1h')
    
    Returns:
        DataFrame with OHLCV data and timestamp index
    """
    if symbol not in SUPPORTED_SYMBOLS:
        raise ValueError(f"Symbol {symbol} not supported")
    if timeframe not in SUPPORTED_TIMEFRAMES:
        raise ValueError(f"Timeframe {timeframe} not supported")
    
    try:
        tf_map = {'15m': 'BTC_15m.parquet' if 'BTC' in symbol else f'{symbol.replace("USDT", "")}_{timeframe}.parquet',
                 '1h': f'{symbol.replace("USDT", "")}_1h.parquet'}
        
        filename = tf_map[timeframe] if symbol.startswith('BTC') else f'{symbol.replace("USDT", "")}_{timeframe}.parquet'
        
        path = hf_hub_download(
            repo_id="zongowo111/v2-crypto-ohlcv-data",
            filename=f"klines/{symbol}/{filename}",
            repo_type="dataset"
        )
        
        df = pd.read_parquet(path)
        
        try:
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
                df.set_index('timestamp', inplace=True)
            else:
                df.index = pd.to_datetime(df.index, errors='coerce')
        except:
            if isinstance(df.index, pd.RangeIndex):
                df.index = pd.to_datetime(df.index, unit='ms', errors='coerce')
            else:
                df.index = pd.to_datetime(df.index, errors='coerce')
        
        df = df[df.index.notna()]
        df.columns = df.columns.str.lower()
        
        return df.sort_index()
    
    except Exception as e:
        print(f"Error downloading data for {symbol} {timeframe}: {str(e)}")
        return None


# ==================== MODULE 2: TECHNICAL INDICATORS ====================

def calculate_bollinger_bands(df, period=BB_PERIOD, std=BB_STD):
    """Calculate Bollinger Bands"""
    df['bb_middle'] = df['close'].rolling(window=period).mean()
    bb_std = df['close'].rolling(window=period).std()
    df['bb_upper'] = df['bb_middle'] + (bb_std * std)
    df['bb_lower'] = df['bb_middle'] - (bb_std * std)
    df['bb_width'] = df['bb_upper'] - df['bb_lower']
    df['bb_pct'] = (df['close'] - df['bb_lower']) / df['bb_width']
    return df

def calculate_rsi(df, period=RSI_PERIOD):
    """Calculate RSI (Relative Strength Index)"""
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    return df

def calculate_atr(df, period=ATR_PERIOD):
    """Calculate ATR (Average True Range)"""
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift())
    low_close = abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    df['atr'] = true_range.rolling(period).mean()
    return df

def calculate_adx(df, period=ADX_PERIOD):
    """Calculate ADX (Average Directional Index)"""
    high_diff = df['high'].diff()
    low_diff = -df['low'].diff()
    
    di_plus = 100 * (high_diff.where(high_diff > 0, 0) / df['atr']).rolling(period).mean()
    di_minus = 100 * (low_diff.where(low_diff > 0, 0) / df['atr']).rolling(period).mean()
    
    dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    df['adx'] = dx.rolling(period).mean()
    df['di_plus'] = di_plus
    df['di_minus'] = di_minus
    
    return df

def calculate_volume_ma(df, period=VOLUME_MA_PERIOD):
    """Calculate Volume Moving Average"""
    df['volume_ma'] = df['volume'].rolling(window=period).mean()
    df['volume_ratio'] = df['volume'] / (df['volume_ma'] + 1e-10)
    return df

def calculate_all_indicators(df):
    """Calculate all technical indicators in sequence"""
    df = df.copy()
    df = calculate_bollinger_bands(df)
    df = calculate_rsi(df)
    df = calculate_atr(df)
    df = calculate_adx(df)
    df = calculate_volume_ma(df)
    return df


# ==================== MODULE 3: SUPPORT STRENGTH SCORING ====================

def rate_support_strength(df, idx, direction='long'):
    """
    Calculate SSS (Support Strength Score) - OPTIMIZED
    
    Formula: SSS = Accuracy(0.4) + Test_Count(0.3) + Multi_Test_Bonus(0.3)
    """
    if idx < 20:
        return 0.5
    
    if direction == 'long':
        support = df.iloc[idx]['bb_lower']
        look_back = min(100, idx)
        lows = df.iloc[idx-look_back:idx]['low'].values
        touches = np.sum(lows <= support * 1.001)
        
        accuracy = 1.0 - (touches * 0.02) if touches > 0 else 1.0
        test_count = min(1.0, touches / 5.0)
        multi_test_bonus = min(0.2, max(0, touches - 1) * 0.05)
        
        sss = accuracy * 0.4 + test_count * 0.3 + multi_test_bonus * 0.3
        
    else:
        resistance = df.iloc[idx]['bb_upper']
        look_back = min(100, idx)
        highs = df.iloc[idx-look_back:idx]['high'].values
        touches = np.sum(highs >= resistance * 0.999)
        
        accuracy = 1.0 - (touches * 0.02) if touches > 0 else 1.0
        test_count = min(1.0, touches / 5.0)
        multi_test_bonus = min(0.2, max(0, touches - 1) * 0.05)
        
        sss = accuracy * 0.4 + test_count * 0.3 + multi_test_bonus * 0.3
    
    return max(0, min(1.0, sss))


# ==================== MODULE 4: BOUNCE DETECTION ====================

def detect_lower_band_touch(df, idx, tolerance=TOUCH_THRESHOLD):
    """Detect if candle touches lower Bollinger Band"""
    if idx < BB_PERIOD or pd.isna(df.iloc[idx]['bb_lower']):
        return False
    return df.iloc[idx]['low'] <= df.iloc[idx]['bb_lower'] * tolerance

def detect_upper_band_touch(df, idx, tolerance=1.0001):
    """Detect if candle touches upper Bollinger Band"""
    if idx < BB_PERIOD or pd.isna(df.iloc[idx]['bb_upper']):
        return False
    return df.iloc[idx]['high'] >= df.iloc[idx]['bb_upper'] / tolerance

def is_bounce_successful(df, idx, direction='long', target_ratio=TARGET_RATIO, min_bars=MIN_BARS_TO_CONFIRM):
    """
    Determine if bounce is successful
    
    Returns: (success: bool, target_price: float, bars_to_target: int)
    """
    if direction == 'long':
        support = df.iloc[idx]['bb_lower']
        target = support + (df.iloc[idx]['bb_upper'] - df.iloc[idx]['bb_lower']) * target_ratio
        
        for i in range(idx + 1, min(idx + 50, len(df))):
            if df.iloc[i]['high'] >= target:
                return True, target, i - idx
        
        return False, target, None
    
    else:
        resistance = df.iloc[idx]['bb_upper']
        target = resistance - (df.iloc[idx]['bb_upper'] - df.iloc[idx]['bb_lower']) * target_ratio
        
        for i in range(idx + 1, min(idx + 50, len(df))):
            if df.iloc[i]['low'] <= target:
                return True, target, i - idx
        
        return False, target, None


# ==================== MODULE 5: SCORING CALCULATION ====================

def calculate_pei_for_touch(df, idx, direction='long'):
    """
    Calculate PEI (Price Exhaustion Index)
    
    Components:
    - Shadow Ratio (35%)
    - Volume Spike (30%)
    - RSI Extreme (20%)
    - Pattern Factor (15%)
    """
    if idx < 2:
        return 0.5
    
    if direction == 'long':
        lower_shadow = df.iloc[idx]['open'] - df.iloc[idx]['low']
        candle_range = df.iloc[idx]['high'] - df.iloc[idx]['low']
        shadow_ratio = lower_shadow / (candle_range + 1e-10)
        shadow_score = min(1.0, shadow_ratio * 2)
    else:
        upper_shadow = df.iloc[idx]['high'] - df.iloc[idx]['open']
        candle_range = df.iloc[idx]['high'] - df.iloc[idx]['low']
        shadow_ratio = upper_shadow / (candle_range + 1e-10)
        shadow_score = min(1.0, shadow_ratio * 2)
    
    vol_ratio = df.iloc[idx]['volume_ratio']
    vol_score = min(1.0, (vol_ratio - 1) / 2) if vol_ratio > 0 else 0
    
    rsi = df.iloc[idx]['rsi']
    if direction == 'long':
        rsi_score = 1.0 if rsi < 30 else max(0, (50 - rsi) / 50) if not pd.isna(rsi) else 0.5
    else:
        rsi_score = 1.0 if rsi > 70 else max(0, (rsi - 50) / 50) if not pd.isna(rsi) else 0.5
    
    body_ratio = abs(df.iloc[idx]['close'] - df.iloc[idx]['open']) / (df.iloc[idx]['high'] - df.iloc[idx]['low'] + 1e-10)
    pattern_score = 1.0 - body_ratio
    
    pei = (shadow_score * 0.35 + vol_score * 0.30 + rsi_score * 0.20 + pattern_score * 0.15)
    return max(0, min(1.0, pei))

def calculate_mess(df, idx):
    """
    Calculate MESS (Market Environment Suitability Score)
    
    Components:
    - ADX Strength (30%)
    - BB Width (25%)
    - Price Position (25%)
    - DI Balance (20%)
    """
    if idx < ADX_PERIOD:
        return 0.5
    
    adx = df.iloc[idx]['adx']
    adx_score = min(1.0, adx / 40.0) if not pd.isna(adx) else 0.5
    
    bb_pct = df.iloc[idx]['bb_pct']
    bb_score = 0.5 if pd.isna(bb_pct) else (1.0 - bb_pct)
    
    price_score = abs(bb_pct - 0.3) if not pd.isna(bb_pct) else 0.5
    
    di_plus = df.iloc[idx]['di_plus']
    di_minus = df.iloc[idx]['di_minus']
    di_balance = 1.0 - abs(di_plus - di_minus) / (abs(di_plus) + abs(di_minus) + 1e-10)
    di_score = di_balance if not pd.isna(di_balance) else 0.5
    
    mess = (adx_score * 0.30 + bb_score * 0.25 + price_score * 0.25 + di_score * 0.20)
    return max(0, min(1.0, mess))


# ==================== MODULE 6: COMPLETE LABELING ====================

def label_bounce_signals(df, symbol, direction='long', **params):
    """
    Complete labeling pipeline for bounce signals - OPTIMIZED
    
    Args:
        df: DataFrame with indicators
        symbol: Symbol name
        direction: 'long' or 'short'
        **params: Override default parameters
    
    Returns:
        DataFrame with bounce labels and scores
    """
    df = df.copy()
    df['symbol'] = symbol
    df['bounce_direction'] = direction
    df['is_bounce_touch'] = False
    df['bounce_success'] = 0
    df['sss'] = 0.0
    df['pei'] = 0.0
    df['mess'] = 0.0
    df['mrms'] = 0.5
    df['bvs'] = 0.0
    df['target_price'] = np.nan
    df['bars_to_target'] = np.nan
    
    touch_indices = []
    
    if direction == 'long':
        tolerance = TOUCH_THRESHOLD
        touches = (df['low'] <= df['bb_lower'] * tolerance).values
    else:
        tolerance = 1.0001
        touches = (df['high'] >= df['bb_upper'] / tolerance).values
    
    touch_indices = np.where(touches)[0]
    touch_indices = touch_indices[(touch_indices >= BB_PERIOD) & (touch_indices < len(df) - MIN_BARS_TO_CONFIRM)]
    
    print(f"    Found {len(touch_indices)} potential {direction} touches")
    
    processed = 0
    for idx in touch_indices:
        sss = rate_support_strength(df, idx, direction)
        pei = calculate_pei_for_touch(df, idx, direction)
        mess = calculate_mess(df, idx)
        
        rsi = df.iloc[idx]['rsi']
        mrms = 1.0 if pd.isna(rsi) else max(0, min(1.0, 1.0 - abs(rsi - 50) / 50))
        
        bvs = (mess * MESS_WEIGHT + sss * SSS_WEIGHT + pei * PEI_WEIGHT + mrms * MRMS_WEIGHT)
        
        if bvs >= MIN_BOUNCE_QUALITY:
            df.at[df.index[idx], 'is_bounce_touch'] = True
            df.at[df.index[idx], 'sss'] = sss
            df.at[df.index[idx], 'pei'] = pei
            df.at[df.index[idx], 'mess'] = mess
            df.at[df.index[idx], 'mrms'] = mrms
            df.at[df.index[idx], 'bvs'] = bvs
            
            success, target, bars = is_bounce_successful(df, idx, direction)
            df.at[df.index[idx], 'bounce_success'] = 1 if success else 0
            df.at[df.index[idx], 'target_price'] = target
            if bars is not None:
                df.at[df.index[idx], 'bars_to_target'] = bars
            
            processed += 1
    
    print(f"    Processed {processed} high-quality {direction} bounces")
    return df


# ==================== MODULE 7: FEATURE EXTRACTION ====================

def extract_features_at_touch(df, idx, direction='long'):
    """
    Extract 35+ ML features at touch point
    
    Returns dictionary of all features
    """
    features = {}
    
    features['MESS'] = df.iloc[idx]['mess']
    features['ADX'] = df.iloc[idx]['adx'] / 100.0 if not pd.isna(df.iloc[idx]['adx']) else 0.5
    features['DI_Plus'] = df.iloc[idx]['di_plus'] / 100.0 if not pd.isna(df.iloc[idx]['di_plus']) else 0.5
    features['DI_Minus'] = df.iloc[idx]['di_minus'] / 100.0 if not pd.isna(df.iloc[idx]['di_minus']) else 0.5
    features['BB_Width'] = (df.iloc[idx]['bb_width'] / df.iloc[idx]['close']) if df.iloc[idx]['close'] != 0 else 0.5
    features['Price_Position'] = df.iloc[idx]['bb_pct'] if not pd.isna(df.iloc[idx]['bb_pct']) else 0.5
    
    features['SSS'] = df.iloc[idx]['sss']
    features['Support_Tests'] = min(1.0, sum(1 for i in range(max(0, idx-100), idx) 
                                             if (direction == 'long' and df.iloc[i]['low'] <= df.iloc[idx]['bb_lower'] * 1.001)
                                             or (direction == 'short' and df.iloc[i]['high'] >= df.iloc[idx]['bb_upper'] / 1.001)) / 5.0)
    features['Distance_to_Support'] = 0.5
    features['Previous_Bounce'] = 0.5
    features['Support_Quality'] = 0.5
    
    features['PEI'] = df.iloc[idx]['pei']
    features['Shadow_Ratio'] = 0.5
    features['Volume_Ratio'] = min(1.0, df.iloc[idx]['volume_ratio'] / 3.0) if 'volume_ratio' in df.columns else 0.5
    features['RSI_14'] = df.iloc[idx]['rsi'] / 100.0 if not pd.isna(df.iloc[idx]['rsi']) else 0.5
    features['RSI_Divergence'] = 0.5
    features['Vol_Spike'] = min(1.0, df.iloc[idx]['volume_ratio'] / 2.0) if 'volume_ratio' in df.columns else 0.5
    features['Body_Ratio'] = 0.5
    features['Shape_Factor'] = 0.5
    
    features['MRMS'] = df.iloc[idx]['mrms']
    features['RSI_Level'] = max(0, min(1.0, 1.0 - abs(df.iloc[idx]['rsi'] - 50) / 50)) if not pd.isna(df.iloc[idx]['rsi']) else 0.5
    features['Momentum_Change'] = 0.5
    features['MOM_9'] = 0.5
    features['MOM_20'] = 0.5
    features['Acceleration'] = 0.5
    features['Deceleration'] = 0.5
    
    features['Color_Sequence'] = 0.5
    features['Body_to_Range'] = abs(df.iloc[idx]['close'] - df.iloc[idx]['open']) / (df.iloc[idx]['high'] - df.iloc[idx]['low'] + 1e-10)
    features['Engulfing'] = 0.5
    features['Inside_Bar'] = 0.5
    features['Upper_Shadow'] = (df.iloc[idx]['high'] - df.iloc[idx]['close']) / (df.iloc[idx]['high'] - df.iloc[idx]['low'] + 1e-10)
    features['Lower_Shadow'] = (df.iloc[idx]['open'] - df.iloc[idx]['low']) / (df.iloc[idx]['high'] - df.iloc[idx]['low'] + 1e-10)
    
    features['BVS'] = df.iloc[idx]['bvs']
    features['SBVS'] = 1.0 - df.iloc[idx]['bvs']
    features['Agreement'] = min(df.iloc[idx]['pei'], df.iloc[idx]['sss'])
    
    return features


# ==================== MODULE 8: TRAINING SET GENERATION ====================

def create_training_dataframe(all_labeled_data):
    """
    Create standard training dataframe from labeled data
    
    Args:
        all_labeled_data: List of DataFrames with labels
    
    Returns:
        Consolidated training DataFrame
    """
    training_dfs = []
    
    for df in all_labeled_data:
        if len(df[df['is_bounce_touch']]) == 0:
            continue
        
        bounce_rows = df[df['is_bounce_touch']].copy()
        
        for idx in bounce_rows.index:
            idx_pos = df.index.get_loc(idx)
            features = extract_features_at_touch(df, idx_pos, bounce_rows.loc[idx, 'bounce_direction'])
            features['bounce_success'] = bounce_rows.loc[idx, 'bounce_success']
            features['symbol'] = bounce_rows.loc[idx, 'symbol']
            features['timeframe'] = 'unknown'
            features['timestamp'] = idx
            
            training_dfs.append(features)
    
    if not training_dfs:
        return pd.DataFrame()
    
    result = pd.DataFrame(training_dfs)
    return result


# ==================== MODULE 9: BATCH PROCESSING ====================

def process_all_symbols_and_timeframes(symbols, timeframes):
    """
    Complete pipeline: download, calculate, label, extract - OPTIMIZED
    
    Args:
        symbols: List of symbols
        timeframes: List of timeframes
    
    Returns:
        List of labeled DataFrames
    """
    all_data = []
    
    for symbol in symbols:
        for timeframe in timeframes:
            print(f"Processing {symbol} {timeframe}...")
            
            try:
                df = download_ohlcv_data(symbol, timeframe)
                if df is None or len(df) < BB_PERIOD:
                    print(f"  Skipped: insufficient data")
                    continue
                
                print(f"  Downloaded {len(df)} candles")
                
                df = calculate_all_indicators(df)
                print(f"  Indicators calculated")
                
                df = label_bounce_signals(df, symbol, direction='long')
                df = label_bounce_signals(df, symbol, direction='short')
                
                all_data.append(df)
                print(f"  Complete: {len(df[df['is_bounce_touch']])} bounce points")
            
            except Exception as e:
                print(f"  Error: {str(e)}")
    
    return all_data


def validate_labeled_data(df):
    """
    Validate labeled data quality
    
    Checks:
    - Missing values
    - Value ranges
    - Statistical properties
    - Class balance
    """
    if len(df) == 0:
        print("No data to validate")
        return
    
    print("\n=== Data Validation Report ===\n")
    
    print(f"1. Dataset Shape: {df.shape}")
    print(f"   - Samples: {len(df)}")
    print(f"   - Features: {len(df.columns)}")
    
    print(f"\n2. Missing Values:")
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("   No missing values - OK")
    else:
        print(f"   Found {missing.sum()} missing values")
    
    print(f"\n3. Score Ranges:")
    score_cols = ['BVS', 'SSS', 'PEI', 'MESS']
    for col in score_cols:
        if col in df.columns:
            valid = ((df[col] >= 0) & (df[col] <= 1)).sum()
            print(f"   {col}: {valid}/{len(df)} within [0, 1]")
    
    print(f"\n4. Class Balance:")
    if 'bounce_success' in df.columns:
        success_rate = df['bounce_success'].mean()
        print(f"   Success rate: {success_rate:.1%}")
        if 0.3 < success_rate < 0.7:
            print("   Class balance: GOOD")
        else:
            print("   Class balance: IMBALANCED")
    
    print(f"\n5. Feature Statistics:")
    for col in ['BVS', 'SSS', 'PEI', 'MESS']:
        if col in df.columns:
            print(f"   {col}: mean={df[col].mean():.3f}, std={df[col].std():.3f}")
    
    print(f"\n=== Validation Complete ===\n")
