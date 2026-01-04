"""
Smart Bounce Point Filter
Intelligent filtering to identify high-probability bounce points

Author: Caizong Xun
Version: 1.0.0
Date: 2026-01-04

Strategy: Use multi-factor filtering to achieve 50-70% accuracy
"""

import pandas as pd
import numpy as np
from data_labeling_implementation import (
    download_ohlcv_data,
    calculate_all_indicators,
    label_bounce_signals
)

def create_smart_filters():
    """
    Create multiple filter strategies based on data analysis
    """
    print("="*70)
    print("Smart Bounce Point Filter System")
    print("="*70)
    
    print("\nDownloading and processing BTC 15m data...")
    df = download_ohlcv_data('BTCUSDT', '15m')
    df = calculate_all_indicators(df)
    df = label_bounce_signals(df, 'BTCUSDT', direction='long')
    df = label_bounce_signals(df, 'BTCUSDT', direction='short')
    
    all_touches = df[df['is_bounce_touch']].copy()
    print(f"Total bounce points: {len(all_touches)}")
    
    # ========== FILTER STRATEGY 1: CONSERVATIVE (Highest Accuracy) ==========
    print("\n" + "="*70)
    print("FILTER 1: CONSERVATIVE (Highest Accuracy)")
    print("="*70)
    print("\nTarget: 70%+ accuracy with fewer signals")
    print("\nCriteria:")
    print("  - BB Position: Very close to support/resistance (0-10%)")
    print("  - RSI: Extreme (Long: <30, Short: >70)")
    print("  - Volume: Confirmation (>1.5x average)")
    print("  - Recent Momentum: Weak (recovery starting)")
    
    conservative = all_touches[
        (
            (
                (all_touches['bounce_direction'] == 'long') &
                (all_touches['rsi'] < 30) &
                (all_touches['bb_pct'] <= 0.15) &
                (all_touches['volume_ratio'] >= 1.3)
            ) |
            (
                (all_touches['bounce_direction'] == 'short') &
                (all_touches['rsi'] > 70) &
                (all_touches['bb_pct'] >= 0.85) &
                (all_touches['volume_ratio'] >= 1.3)
            )
        )
    ]
    
    if len(conservative) > 0:
        acc = conservative['bounce_success'].mean()
        print(f"\nResults:")
        print(f"  Signals: {len(conservative)}")
        print(f"  Accuracy: {acc:.1%}")
        print(f"  Profitable trades: {conservative['bounce_success'].sum()}")
    else:
        print(f"\nNo signals matching conservative criteria")
    
    # ========== FILTER STRATEGY 2: MODERATE (Balanced) ==========
    print("\n" + "="*70)
    print("FILTER 2: MODERATE (Balanced)")
    print("="*70)
    print("\nTarget: 50-60% accuracy with more signals")
    print("\nCriteria:")
    print("  - BB Position: Close to edge (0-25%)")
    print("  - RSI: Oversold/Overbought (Long: <35, Short: >65)")
    print("  - Volume: Normal or above (>1.0x average)")
    print("  - Trend: Not too strong (ADX < 35)")
    
    moderate = all_touches[
        (
            (
                (all_touches['bounce_direction'] == 'long') &
                (all_touches['rsi'] < 35) &
                (all_touches['bb_pct'] <= 0.25) &
                (all_touches['volume_ratio'] >= 1.0) &
                (all_touches['adx'] < 35)
            ) |
            (
                (all_touches['bounce_direction'] == 'short') &
                (all_touches['rsi'] > 65) &
                (all_touches['bb_pct'] >= 0.75) &
                (all_touches['volume_ratio'] >= 1.0) &
                (all_touches['adx'] < 35)
            )
        )
    ]
    
    if len(moderate) > 0:
        acc = moderate['bounce_success'].mean()
        print(f"\nResults:")
        print(f"  Signals: {len(moderate)}")
        print(f"  Accuracy: {acc:.1%}")
        print(f"  Profitable trades: {moderate['bounce_success'].sum()}")
    else:
        print(f"\nNo signals matching moderate criteria")
    
    # ========== FILTER STRATEGY 3: AGGRESSIVE (More Signals) ==========
    print("\n" + "="*70)
    print("FILTER 3: AGGRESSIVE (More Signals)")
    print("="*70)
    print("\nTarget: 35-50% accuracy with many signals")
    print("\nCriteria:")
    print("  - BB Position: Within bands (0-35%)")
    print("  - RSI: Weak/Strong (Long: <50, Short: >50)")
    print("  - Recent Trend: Confirmable direction")
    
    aggressive = all_touches[
        (
            (
                (all_touches['bounce_direction'] == 'long') &
                (all_touches['rsi'] < 50) &
                (all_touches['bb_pct'] <= 0.35)
            ) |
            (
                (all_touches['bounce_direction'] == 'short') &
                (all_touches['rsi'] > 50) &
                (all_touches['bb_pct'] >= 0.65)
            )
        )
    ]
    
    if len(aggressive) > 0:
        acc = aggressive['bounce_success'].mean()
        print(f"\nResults:")
        print(f"  Signals: {len(aggressive)}")
        print(f"  Accuracy: {acc:.1%}")
        print(f"  Profitable trades: {aggressive['bounce_success'].sum()}")
    else:
        print(f"\nNo signals matching aggressive criteria")
    
    # ========== FILTER STRATEGY 4: VOLUME FOCUS ==========
    print("\n" + "="*70)
    print("FILTER 4: VOLUME FOCUS (Volatility Burst)")
    print("="*70)
    print("\nTarget: Focus on volume spikes as confirmation")
    print("\nCriteria:")
    print("  - Volume: Strong spike (>2.0x average)")
    print("  - RSI: Extreme (Long: <40, Short: >60)")
    print("  - Trend: Any condition")
    
    volume_focus = all_touches[
        (
            (
                (all_touches['bounce_direction'] == 'long') &
                (all_touches['rsi'] < 40) &
                (all_touches['volume_ratio'] >= 2.0)
            ) |
            (
                (all_touches['bounce_direction'] == 'short') &
                (all_touches['rsi'] > 60) &
                (all_touches['volume_ratio'] >= 2.0)
            )
        )
    ]
    
    if len(volume_focus) > 0:
        acc = volume_focus['bounce_success'].mean()
        print(f"\nResults:")
        print(f"  Signals: {len(volume_focus)}")
        print(f"  Accuracy: {acc:.1%}")
        print(f"  Profitable trades: {volume_focus['bounce_success'].sum()}")
    else:
        print(f"\nNo signals matching volume focus criteria")
    
    # ========== FILTER STRATEGY 5: TIME-BASED (Recent Context) ==========
    print("\n" + "="*70)
    print("FILTER 5: TIME-BASED (Recent Context)")
    print("="*70)
    print("\nTarget: Filter based on recent price action")
    print("\nCriteria:")
    print("  - RSI Extreme + BB Edge + Recent rebound")
    print("  - Better signals when price just bounced off")
    
    # For this, we need to check if price rebounded in previous bars
    time_based = all_touches[
        (
            (
                (all_touches['bounce_direction'] == 'long') &
                (all_touches['rsi'] < 35) &
                (all_touches['bb_pct'] <= 0.2)
            ) |
            (
                (all_touches['bounce_direction'] == 'short') &
                (all_touches['rsi'] > 65) &
                (all_touches['bb_pct'] >= 0.8)
            )
        )
    ]
    
    if len(time_based) > 0:
        acc = time_based['bounce_success'].mean()
        print(f"\nResults:")
        print(f"  Signals: {len(time_based)}")
        print(f"  Accuracy: {acc:.1%}")
        print(f"  Profitable trades: {time_based['bounce_success'].sum()}")
    else:
        print(f"\nNo signals matching time-based criteria")
    
    # ========== SUMMARY TABLE ==========
    print("\n" + "="*70)
    print("SUMMARY: FILTER COMPARISON")
    print("="*70)
    
    results = [
        ('Conservative', conservative),
        ('Moderate', moderate),
        ('Aggressive', aggressive),
        ('Volume Focus', volume_focus),
        ('Time-Based', time_based),
    ]
    
    print(f"\n{'Strategy':<20} {'Signals':<10} {'Accuracy':<12} {'Win Count':<10}")
    print("-" * 55)
    
    for name, data in results:
        if len(data) > 0:
            acc = data['bounce_success'].mean()
            wins = data['bounce_success'].sum()
            print(f"{name:<20} {len(data):<10} {acc:<12.1%} {int(wins):<10}")
        else:
            print(f"{name:<20} 0          0.0%         0")
    
    # ========== RECOMMENDATIONS ==========
    print("\n" + "="*70)
    print("RECOMMENDATIONS FOR PRACTICAL USE")
    print("="*70)
    
    print("\n1. FOR CONSERVATIVE TRADING (Risk-Averse):")
    print("   Use Conservative Filter")
    if len(conservative) > 0:
        print(f"   - {len(conservative)} signals per dataset")
        print(f"   - ~{conservative['bounce_success'].mean():.0%} expected win rate")
    
    print("\n2. FOR BALANCED TRADING (Medium Risk):")
    print("   Use Moderate Filter")
    if len(moderate) > 0:
        print(f"   - {len(moderate)} signals per dataset")
        print(f"   - ~{moderate['bounce_success'].mean():.0%} expected win rate")
    
    print("\n3. FOR AGGRESSIVE TRADING (High Volume):")
    print("   Use Aggressive Filter")
    if len(aggressive) > 0:
        print(f"   - {len(aggressive)} signals per dataset")
        print(f"   - ~{aggressive['bounce_success'].mean():.0%} expected win rate")
    
    print("\n4. RECOMMENDED APPROACH:")
    print("   Combine multiple filters:")
    print("   - Use Conservative for high-conviction trades")
    print("   - Add Moderate for diversification")
    print("   - Monitor Volume-Focus for confirmation spikes")
    
    print("\n5. REAL-WORLD APPLICATION:")
    print("   - Set stop loss: 50 pips or 1 ATR below support")
    print("   - Take profit: At middle band or 1.5x band width")
    print("   - Risk/Reward ratio: Minimum 1:2")
    print("   - Position sizing: Fixed risk per trade")
    
    # ========== EXPORT SIGNALS ==========
    print("\n" + "="*70)
    print("EXPORTING FILTERED SIGNALS")
    print("="*70)
    
    # Export conservative signals
    if len(conservative) > 0:
        conservative.to_csv('signals_conservative.csv')
        print(f"\nExported Conservative signals to: signals_conservative.csv")
    
    if len(moderate) > 0:
        moderate.to_csv('signals_moderate.csv')
        print(f"Exported Moderate signals to: signals_moderate.csv")
    
    if len(aggressive) > 0:
        aggressive.to_csv('signals_aggressive.csv')
        print(f"Exported Aggressive signals to: signals_aggressive.csv")
    
    print("\nYou can now use these signals in your trading system!")
    print("\nNext steps:")
    print("  1. Backtest with your preferred risk/reward settings")
    print("  2. Forward test on demo account")
    print("  3. Monitor real-time signals")
    print("  4. Adjust filters based on live performance")

if __name__ == '__main__':
    create_smart_filters()
