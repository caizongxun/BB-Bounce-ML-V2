"""
BTC 15m Bounce Detection Optimization
Optimization script to achieve 80%+ accuracy on BTC 15m timeframe

Author: Caizong Xun
Version: 1.0.0
Date: 2026-01-04
"""

import pandas as pd
import numpy as np
from data_labeling_implementation import (
    download_ohlcv_data,
    calculate_all_indicators,
    label_bounce_signals,
    validate_labeled_data
)

def test_btc_15m():
    """
    Test BTC 15m with detailed analysis
    """
    print("="*70)
    print("BTC 15m Bounce Detection - Accuracy Optimization")
    print("="*70)
    
    print("\nStep 1: Download BTC 15m data...")
    df = download_ohlcv_data('BTCUSDT', '15m')
    
    if df is None:
        print("Failed to download data")
        return
    
    print(f"Downloaded {len(df)} candles")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")
    
    print("\nStep 2: Calculate indicators...")
    df = calculate_all_indicators(df)
    print("Indicators calculated successfully")
    
    print("\nStep 3: Label bounce signals (LONG)...")
    df = label_bounce_signals(df, 'BTCUSDT', direction='long')
    
    long_touches = df[df['is_bounce_touch'] & (df['bounce_direction'] == 'long')]
    print(f"Long touches found: {len(long_touches)}")
    
    if len(long_touches) > 0:
        long_success = long_touches['bounce_success'].sum()
        long_rate = long_touches['bounce_success'].mean()
        print(f"  Successful: {long_success}/{len(long_touches)}")
        print(f"  Success rate: {long_rate:.1%}")
        
        print(f"\n  Long BVS Score Distribution:")
        print(f"    Mean: {long_touches['bvs'].mean():.3f}")
        print(f"    Std: {long_touches['bvs'].std():.3f}")
        print(f"    Min: {long_touches['bvs'].min():.3f}")
        print(f"    Max: {long_touches['bvs'].max():.3f}")
        
        print(f"\n  Long by BVS Range:")
        for threshold in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
            subset = long_touches[long_touches['bvs'] >= threshold]
            if len(subset) > 0:
                rate = subset['bounce_success'].mean()
                print(f"    BVS >= {threshold}: {len(subset)} samples, {rate:.1%} success")
    
    print("\nStep 4: Label bounce signals (SHORT)...")
    df = label_bounce_signals(df, 'BTCUSDT', direction='short')
    
    short_touches = df[df['is_bounce_touch'] & (df['bounce_direction'] == 'short')]
    print(f"Short touches found: {len(short_touches)}")
    
    if len(short_touches) > 0:
        short_success = short_touches['bounce_success'].sum()
        short_rate = short_touches['bounce_success'].mean()
        print(f"  Successful: {short_success}/{len(short_touches)}")
        print(f"  Success rate: {short_rate:.1%}")
        
        print(f"\n  Short BVS Score Distribution:")
        print(f"    Mean: {short_touches['bvs'].mean():.3f}")
        print(f"    Std: {short_touches['bvs'].std():.3f}")
        print(f"    Min: {short_touches['bvs'].min():.3f}")
        print(f"    Max: {short_touches['bvs'].max():.3f}")
        
        print(f"\n  Short by BVS Range:")
        for threshold in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
            subset = short_touches[short_touches['bvs'] >= threshold]
            if len(subset) > 0:
                rate = subset['bounce_success'].mean()
                print(f"    BVS >= {threshold}: {len(subset)} samples, {rate:.1%} success")
    
    print("\nStep 5: Overall Statistics")
    all_touches = df[df['is_bounce_touch']]
    print(f"Total touches: {len(all_touches)}")
    print(f"Total successful: {all_touches['bounce_success'].sum()}")
    print(f"Overall success rate: {all_touches['bounce_success'].mean():.1%}")
    
    print("\nStep 6: Component Analysis")
    print(f"\nPEI (Price Exhaustion Index) Analysis:")
    print(f"  Mean: {all_touches['pei'].mean():.3f}")
    print(f"  Correlation with success: {all_touches[['pei', 'bounce_success']].corr().iloc[0,1]:.3f}")
    
    print(f"\nSSS (Support Strength Score) Analysis:")
    print(f"  Mean: {all_touches['sss'].mean():.3f}")
    print(f"  Correlation with success: {all_touches[['sss', 'bounce_success']].corr().iloc[0,1]:.3f}")
    
    print(f"\nMESS (Market Environment) Analysis:")
    print(f"  Mean: {all_touches['mess'].mean():.3f}")
    print(f"  Correlation with success: {all_touches[['mess', 'bounce_success']].corr().iloc[0,1]:.3f}")
    
    print(f"\nMRMS (Momentum) Analysis:")
    print(f"  Mean: {all_touches['mrms'].mean():.3f}")
    print(f"  Correlation with success: {all_touches[['mrms', 'bounce_success']].corr().iloc[0,1]:.3f}")
    
    print("\nStep 7: High Precision Filtered Results")
    print("\nFiltering for 80%+ accuracy:")
    
    high_quality = all_touches[
        (all_touches['bvs'] >= 0.70) &
        (all_touches['pei'] >= 0.50) &
        (all_touches['sss'] >= 0.50)
    ]
    
    if len(high_quality) > 0:
        rate = high_quality['bounce_success'].mean()
        print(f"  Criteria: BVS>=0.70, PEI>=0.50, SSS>=0.50")
        print(f"  Samples: {len(high_quality)}")
        print(f"  Success rate: {rate:.1%}")
    
    ultra_high = all_touches[all_touches['bvs'] >= 0.75]
    if len(ultra_high) > 0:
        rate = ultra_high['bounce_success'].mean()
        print(f"\n  Criteria: BVS>=0.75")
        print(f"  Samples: {len(ultra_high)}")
        print(f"  Success rate: {rate:.1%}")
    
    extreme = all_touches[all_touches['bvs'] >= 0.80]
    if len(extreme) > 0:
        rate = extreme['bounce_success'].mean()
        print(f"\n  Criteria: BVS>=0.80")
        print(f"  Samples: {len(extreme)}")
        print(f"  Success rate: {rate:.1%}")
    
    print("\nStep 8: Sample High Quality Bounces")
    if len(high_quality) > 0:
        samples = high_quality.nlargest(5, 'bvs')
        for idx, (_, row) in enumerate(samples.iterrows(), 1):
            print(f"\n  Sample {idx}: {row.name}")
            print(f"    BVS: {row['bvs']:.3f} (PEI:{row['pei']:.3f}, SSS:{row['sss']:.3f}, MESS:{row['mess']:.3f})")
            print(f"    Success: {'Yes' if row['bounce_success'] == 1 else 'No'}")
            print(f"    Target: {row['target_price']:.2f}, Bars: {row['bars_to_target']:.0f}")
    
    print("\n" + "="*70)
    print("Analysis Complete")
    print("="*70)
    
    return df

def analyze_optimal_threshold(df):
    """
    Find optimal BVS threshold for 80%+ accuracy
    """
    print("\n" + "="*70)
    print("Finding Optimal BVS Threshold for 80%+ Accuracy")
    print("="*70)
    
    all_touches = df[df['is_bounce_touch']]
    
    print("\nBVS Threshold Analysis:\n")
    print(f"{'Threshold':<12} {'Samples':<12} {'Success %':<12} {'Avg Target':<12}")
    print("-" * 50)
    
    thresholds = np.arange(0.40, 0.95, 0.05)
    best_threshold = 0.50
    best_rate = 0.0
    
    for threshold in thresholds:
        subset = all_touches[all_touches['bvs'] >= threshold]
        if len(subset) == 0:
            continue
        
        success_rate = subset['bounce_success'].mean()
        avg_target = subset['target_price'].mean()
        
        print(f"{threshold:<12.2f} {len(subset):<12} {success_rate:<12.1%} {avg_target:<12.2f}")
        
        if success_rate >= 0.80 and success_rate > best_rate:
            best_rate = success_rate
            best_threshold = threshold
    
    print("\n" + "="*70)
    print(f"Optimal Threshold: BVS >= {best_threshold:.2f}")
    print(f"Expected Accuracy: {best_rate:.1%}")
    print("="*70)
    
    return best_threshold, best_rate

if __name__ == '__main__':
    df = test_btc_15m()
    
    if df is not None:
        threshold, rate = analyze_optimal_threshold(df)
        
        if rate >= 0.80:
            print(f"\n SUCCESS: Achieved {rate:.1%} accuracy with BVS >= {threshold:.2f}")
        else:
            print(f"\n NOTE: Current best accuracy is {rate:.1%}")
            print("   Adjusting parameters may improve results further.")
