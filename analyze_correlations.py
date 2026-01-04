"""
Correlation Analysis for BTC 15m Bounce Success
Identify which factors actually predict bounce success

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

def comprehensive_correlation_analysis():
    """
    Analyze which factors are correlated with bounce success
    """
    print("="*70)
    print("Comprehensive Correlation Analysis - BTC 15m")
    print("="*70)
    
    print("\nDownloading and processing data...")
    df = download_ohlcv_data('BTCUSDT', '15m')
    df = calculate_all_indicators(df)
    df = label_bounce_signals(df, 'BTCUSDT', direction='long')
    df = label_bounce_signals(df, 'BTCUSDT', direction='short')
    
    all_touches = df[df['is_bounce_touch']].copy()
    print(f"Total bounces analyzed: {len(all_touches)}")
    
    print("\n" + "="*70)
    print("PRIMARY FACTORS ANALYSIS")
    print("="*70)
    
    # Analyze RSI
    print("\n1. RSI (Relative Strength Index) - Most Critical")
    print("-" * 70)
    
    # For long bounces
    long_bounces = all_touches[all_touches['bounce_direction'] == 'long']
    if len(long_bounces) > 0:
        rsi_success = long_bounces[long_bounces['bounce_success'] == 1]['rsi'].dropna()
        rsi_fail = long_bounces[long_bounces['bounce_success'] == 0]['rsi'].dropna()
        
        print(f"\nLong Bounces (RSI should be low for good bounces):")
        print(f"  Successful bounces - RSI mean: {rsi_success.mean():.1f}")
        print(f"  Failed bounces - RSI mean: {rsi_fail.mean():.1f}")
        print(f"  Correlation: {long_bounces[['rsi', 'bounce_success']].corr().iloc[0,1]:.3f}")
        
        print(f"\n  RSI ranges for Long:")
        for threshold in [10, 20, 30, 40, 50]:
            subset = long_bounces[long_bounces['rsi'] < threshold]
            if len(subset) > 0:
                rate = subset['bounce_success'].mean()
                print(f"    RSI < {threshold}: {len(subset)} samples, {rate:.1%} success")
    
    # For short bounces
    short_bounces = all_touches[all_touches['bounce_direction'] == 'short']
    if len(short_bounces) > 0:
        rsi_success = short_bounces[short_bounces['bounce_success'] == 1]['rsi'].dropna()
        rsi_fail = short_bounces[short_bounces['bounce_success'] == 0]['rsi'].dropna()
        
        print(f"\nShort Bounces (RSI should be high for good bounces):")
        print(f"  Successful bounces - RSI mean: {rsi_success.mean():.1f}")
        print(f"  Failed bounces - RSI mean: {rsi_fail.mean():.1f}")
        print(f"  Correlation: {short_bounces[['rsi', 'bounce_success']].corr().iloc[0,1]:.3f}")
        
        print(f"\n  RSI ranges for Short:")
        for threshold in [50, 60, 70, 80, 90]:
            subset = short_bounces[short_bounces['rsi'] > threshold]
            if len(subset) > 0:
                rate = subset['bounce_success'].mean()
                print(f"    RSI > {threshold}: {len(subset)} samples, {rate:.1%} success")
    
    # Analyze Volume
    print("\n\n2. Volume Ratio - Secondary Factor")
    print("-" * 70)
    
    vol_corr = all_touches[['volume_ratio', 'bounce_success']].corr().iloc[0,1]
    print(f"\nVolume Ratio Correlation: {vol_corr:.3f}")
    
    print(f"\nVolume ratio distribution:")
    for threshold in [1.0, 1.5, 2.0, 2.5, 3.0]:
        subset = all_touches[all_touches['volume_ratio'] >= threshold]
        if len(subset) > 0:
            rate = subset['bounce_success'].mean()
            print(f"  Volume >= {threshold}x: {len(subset)} samples, {rate:.1%} success")
    
    # Analyze Price Position
    print("\n\n3. Price Position in BB - Tertiary Factor")
    print("-" * 70)
    
    bb_corr = all_touches[['bb_pct', 'bounce_success']].corr().iloc[0,1]
    print(f"\nBB Position Correlation: {bb_corr:.3f}")
    
    print(f"\nBB position ranges (0=lower, 1=upper):")
    for threshold in [0.1, 0.2, 0.3, 0.4, 0.5]:
        subset = all_touches[all_touches['bb_pct'] <= threshold]
        if len(subset) > 0:
            rate = subset['bounce_success'].mean()
            print(f"  BB_pct <= {threshold}: {len(subset)} samples, {rate:.1%} success")
    
    # Analyze ADX
    print("\n\n4. ADX (Trend Strength) - Additional Factor")
    print("-" * 70)
    
    adx_corr = all_touches[['adx', 'bounce_success']].corr().iloc[0,1]
    print(f"\nADX Correlation: {adx_corr:.3f}")
    
    print(f"\nADX strength ranges:")
    for threshold in [10, 15, 20, 25, 30, 35, 40]:
        subset = all_touches[all_touches['adx'] >= threshold]
        if len(subset) > 0:
            rate = subset['bounce_success'].mean()
            print(f"  ADX >= {threshold}: {len(subset)} samples, {rate:.1%} success")
    
    # Combined factors
    print("\n\n" + "="*70)
    print("COMBINED FACTORS ANALYSIS")
    print("="*70)
    
    print("\nFinding optimal filter combinations...\n")
    
    # Test different combinations
    combinations = [
        {
            'name': 'RSI Extreme Only',
            'long': all_touches[(all_touches['bounce_direction'] == 'long') & (all_touches['rsi'] < 30)],
            'short': all_touches[(all_touches['bounce_direction'] == 'short') & (all_touches['rsi'] > 70)]
        },
        {
            'name': 'RSI + Strong Volume',
            'long': all_touches[(all_touches['bounce_direction'] == 'long') & (all_touches['rsi'] < 30) & (all_touches['volume_ratio'] >= 1.5)],
            'short': all_touches[(all_touches['bounce_direction'] == 'short') & (all_touches['rsi'] > 70) & (all_touches['volume_ratio'] >= 1.5)]
        },
        {
            'name': 'RSI + Low Position in BB',
            'long': all_touches[(all_touches['bounce_direction'] == 'long') & (all_touches['rsi'] < 35) & (all_touches['bb_pct'] <= 0.3)],
            'short': all_touches[(all_touches['bounce_direction'] == 'short') & (all_touches['rsi'] > 65) & (all_touches['bb_pct'] >= 0.7)]
        },
        {
            'name': 'RSI + Strong Trend (ADX)',
            'long': all_touches[(all_touches['bounce_direction'] == 'long') & (all_touches['rsi'] < 30) & (all_touches['adx'] >= 20)],
            'short': all_touches[(all_touches['bounce_direction'] == 'short') & (all_touches['rsi'] > 70) & (all_touches['adx'] >= 20)]
        },
        {
            'name': 'RSI + Volume + ADX (Strict)',
            'long': all_touches[(all_touches['bounce_direction'] == 'long') & (all_touches['rsi'] < 25) & (all_touches['volume_ratio'] >= 2.0) & (all_touches['adx'] >= 20)],
            'short': all_touches[(all_touches['bounce_direction'] == 'short') & (all_touches['rsi'] > 75) & (all_touches['volume_ratio'] >= 2.0) & (all_touches['adx'] >= 20)]
        },
        {
            'name': 'Moderate RSI + Volume (Balanced)',
            'long': all_touches[(all_touches['bounce_direction'] == 'long') & (all_touches['rsi'] < 35) & (all_touches['volume_ratio'] >= 1.3)],
            'short': all_touches[(all_touches['bounce_direction'] == 'short') & (all_touches['rsi'] > 65) & (all_touches['volume_ratio'] >= 1.3)]
        },
    ]
    
    results = []
    for combo in combinations:
        long_data = combo['long']
        short_data = combo['short']
        
        combined = pd.concat([long_data, short_data])
        
        if len(combined) > 0:
            success_rate = combined['bounce_success'].mean()
            samples = len(combined)
            results.append({
                'Filter': combo['name'],
                'Samples': samples,
                'Success Rate': success_rate,
                'Long Samples': len(long_data),
                'Short Samples': len(short_data)
            })
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('Success Rate', ascending=False)
    
    print(results_df.to_string(index=False))
    
    print("\n" + "="*70)
    print("RECOMMENDATION")
    print("="*70)
    
    best = results_df.iloc[0]
    print(f"\nBest Filter: {best['Filter']}")
    print(f"Success Rate: {best['Success Rate']:.1%}")
    print(f"Total Samples: {best['Samples']}")
    
    if best['Success Rate'] >= 0.80:
        print("\nStatus: TARGET ACHIEVED (80%+ accuracy)")
    else:
        print(f"\nStatus: Below target. Current best is {best['Success Rate']:.1%}")
        print("\nThe data suggests that Bollinger Band bounce detection at 15m timeframe")
        print("has limited inherent predictability. Consider:")
        print("1. Using larger timeframes (1h, 4h)")
        print("2. Adding more technical indicators")
        print("3. Using machine learning with historical patterns")
        print("4. Combining with market microstructure data")

if __name__ == '__main__':
    comprehensive_correlation_analysis()
