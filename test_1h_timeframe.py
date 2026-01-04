"""
BTC 1h Bounce Detection - Higher Timeframe Test
Test whether 1h timeframe provides better accuracy

Author: Caizong Xun
Version: 1.0.0
Date: 2026-01-04
"""

import pandas as pd
import numpy as np
from data_labeling_implementation import (
    download_ohlcv_data,
    calculate_all_indicators,
    label_bounce_signals
)

def test_1h_timeframe():
    """
    Test BTC 1h with detailed analysis
    """
    print("="*70)
    print("BTC 1h Bounce Detection - Testing Higher Timeframe")
    print("="*70)
    
    print("\nDownloading BTC 1h data...")
    df = download_ohlcv_data('BTCUSDT', '1h')
    
    if df is None:
        print("Failed to download data")
        return
    
    print(f"Downloaded {len(df)} candles")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")
    print(f"Approx {len(df)/24:.0f} days of data")
    
    print("\nCalculating indicators...")
    df = calculate_all_indicators(df)
    
    print("\nLabeling bounce signals...")
    df = label_bounce_signals(df, 'BTCUSDT', direction='long')
    df = label_bounce_signals(df, 'BTCUSDT', direction='short')
    
    all_touches = df[df['is_bounce_touch']]
    print(f"Total touches detected: {len(all_touches)}")
    
    if len(all_touches) == 0:
        print("No touches found")
        return
    
    print(f"\n=== BASELINE PERFORMANCE ===")
    print(f"Total successful: {all_touches['bounce_success'].sum()}")
    print(f"Overall success rate: {all_touches['bounce_success'].mean():.1%}")
    
    # Analyze by BVS
    print(f"\n=== BVS SCORE DISTRIBUTION ===")
    print(f"Mean: {all_touches['bvs'].mean():.3f}")
    print(f"Std: {all_touches['bvs'].std():.3f}")
    print(f"Min: {all_touches['bvs'].min():.3f}")
    print(f"Max: {all_touches['bvs'].max():.3f}")
    
    print(f"\nBy BVS threshold:")
    for threshold in np.arange(0.50, 0.85, 0.05):
        subset = all_touches[all_touches['bvs'] >= threshold]
        if len(subset) > 0:
            rate = subset['bounce_success'].mean()
            print(f"  BVS >= {threshold:.2f}: {len(subset)} samples, {rate:.1%} success")
    
    # Analyze RSI
    print(f"\n=== RSI ANALYSIS ===")
    
    long_touches = all_touches[all_touches['bounce_direction'] == 'long']
    short_touches = all_touches[all_touches['bounce_direction'] == 'short']
    
    if len(long_touches) > 0:
        print(f"\nLong bounces (RSI should be low):")
        rsi_corr = long_touches[['rsi', 'bounce_success']].corr().iloc[0,1]
        print(f"  Correlation: {rsi_corr:.3f}")
        print(f"  Success mean RSI: {long_touches[long_touches['bounce_success']==1]['rsi'].mean():.1f}")
        print(f"  Failure mean RSI: {long_touches[long_touches['bounce_success']==0]['rsi'].mean():.1f}")
        
        for threshold in [10, 20, 30, 40, 50]:
            subset = long_touches[long_touches['rsi'] < threshold]
            if len(subset) > 0:
                rate = subset['bounce_success'].mean()
                print(f"    RSI < {threshold}: {len(subset)} samples, {rate:.1%} success")
    
    if len(short_touches) > 0:
        print(f"\nShort bounces (RSI should be high):")
        rsi_corr = short_touches[['rsi', 'bounce_success']].corr().iloc[0,1]
        print(f"  Correlation: {rsi_corr:.3f}")
        print(f"  Success mean RSI: {short_touches[short_touches['bounce_success']==1]['rsi'].mean():.1f}")
        print(f"  Failure mean RSI: {short_touches[short_touches['bounce_success']==0]['rsi'].mean():.1f}")
        
        for threshold in [50, 60, 70, 80, 90]:
            subset = short_touches[short_touches['rsi'] > threshold]
            if len(subset) > 0:
                rate = subset['bounce_success'].mean()
                print(f"    RSI > {threshold}: {len(subset)} samples, {rate:.1%} success")
    
    # Test filter combinations
    print(f"\n=== TESTING FILTER COMBINATIONS ===")
    
    combinations = [
        {
            'name': 'RSI Extreme',
            'filter': lambda x: ((x['bounce_direction'] == 'long') & (x['rsi'] < 30)) | ((x['bounce_direction'] == 'short') & (x['rsi'] > 70))
        },
        {
            'name': 'RSI + Low BB Position',
            'filter': lambda x: ((x['bounce_direction'] == 'long') & (x['rsi'] < 35) & (x['bb_pct'] < 0.3)) | ((x['bounce_direction'] == 'short') & (x['rsi'] > 65) & (x['bb_pct'] > 0.7))
        },
        {
            'name': 'RSI + Volume',
            'filter': lambda x: ((x['bounce_direction'] == 'long') & (x['rsi'] < 35) & (x['volume_ratio'] > 1.3)) | ((x['bounce_direction'] == 'short') & (x['rsi'] > 65) & (x['volume_ratio'] > 1.3))
        },
        {
            'name': 'RSI < 25 / > 75 (Very Extreme)',
            'filter': lambda x: ((x['bounce_direction'] == 'long') & (x['rsi'] < 25)) | ((x['bounce_direction'] == 'short') & (x['rsi'] > 75))
        },
        {
            'name': 'RSI + High BVS',
            'filter': lambda x: (x['bvs'] >= 0.65) & (((x['bounce_direction'] == 'long') & (x['rsi'] < 40)) | ((x['bounce_direction'] == 'short') & (x['rsi'] > 60)))
        },
    ]
    
    print()
    for combo in combinations:
        filtered = all_touches[combo['filter'](all_touches)]
        if len(filtered) > 0:
            rate = filtered['bounce_success'].mean()
            print(f"{combo['name']:<30} {len(filtered):<8} samples  {rate:.1%} success")
        else:
            print(f"{combo['name']:<30} No matching samples")
    
    print(f"\n" + "="*70)
    print("COMPARISON: 15m vs 1h")
    print("="*70)
    print(f"\n1h timeframe baseline: {all_touches['bounce_success'].mean():.1%}")
    print(f"15m timeframe baseline: 25.6%")
    print(f"\nImprovement: {(all_touches['bounce_success'].mean() - 0.256)*100:.1f} percentage points")
    
    if all_touches['bounce_success'].mean() >= 0.35:
        print("\nStatus: 1h shows meaningful improvement over 15m")
        print("Further optimization might reach 50%+ accuracy")
    elif all_touches['bounce_success'].mean() >= 0.28:
        print("\nStatus: 1h similar to 15m - timeframe alone won't solve the problem")
    else:
        print("\nStatus: 1h actually worse than 15m")

if __name__ == '__main__':
    test_1h_timeframe()
