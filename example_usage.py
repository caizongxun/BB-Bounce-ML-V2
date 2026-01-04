"""
BB-Bounce-ML-V2 Usage Examples
Demonstrates all major functions with 7 practical scenarios
"""

import pandas as pd
import numpy as np
from data_labeling_implementation import (
    download_ohlcv_data,
    calculate_all_indicators,
    label_bounce_signals,
    create_training_dataframe,
    validate_labeled_data,
    process_all_symbols_and_timeframes,
    extract_features_at_touch
)


def example_1_single_symbol():
    """Example 1: Complete workflow for single cryptocurrency"""
    print("\n" + "="*60)
    print("Example 1: Single Symbol Complete Workflow")
    print("="*60)
    
    print("\n1. Download BTCUSDT 15m data...")
    btc_df = download_ohlcv_data('BTCUSDT', '15m')
    
    if btc_df is None:
        print("Download failed")
        return None
    
    print(f"   Downloaded {len(btc_df)} candles")
    print(f"   Time range: {btc_df.index[0]} to {btc_df.index[-1]}")
    
    print("\n2. Calculate all indicators...")
    btc_df = calculate_all_indicators(btc_df)
    print("   Indicators calculated successfully")
    
    print("\n3. Label bounce signals...")
    btc_df = label_bounce_signals(btc_df, 'BTCUSDT', direction='long')
    btc_df = label_bounce_signals(btc_df, 'BTCUSDT', direction='short')
    
    long_touches = len(btc_df[(btc_df['is_bounce_touch']) & (btc_df['bounce_direction'] == 'long')])
    short_touches = len(btc_df[(btc_df['is_bounce_touch']) & (btc_df['bounce_direction'] == 'short')])
    print(f"   Long touches: {long_touches}")
    print(f"   Short touches: {short_touches}")
    
    print("\n4. View sample results...")
    marked_rows = btc_df[btc_df['is_bounce_touch']].head(3)
    for idx, row in marked_rows.iterrows():
        print(f"\n   Time: {idx}")
        print(f"      Direction: {row['bounce_direction']}")
        print(f"      SSS: {row['sss']:.3f}, PEI: {row['pei']:.3f}, MESS: {row['mess']:.3f}")
        print(f"      Success: {row['bounce_success']}, Target: {row['target_price']:.2f}")
    
    return btc_df


def example_2_statistics():
    """Example 2: Analyze bounce point statistics"""
    print("\n" + "="*60)
    print("Example 2: Bounce Point Statistics")
    print("="*60)
    
    btc_df = download_ohlcv_data('BTCUSDT', '15m')
    if btc_df is None:
        return
    
    btc_df = calculate_all_indicators(btc_df)
    btc_df = label_bounce_signals(btc_df, 'BTCUSDT', direction='long')
    
    touches = btc_df[btc_df['is_bounce_touch'] == True]
    
    if len(touches) == 0:
        print("No bounce touches found")
        return
    
    print(f"\n1. Basic Statistics")
    print(f"   Total touches: {len(touches)}")
    print(f"   Successful: {touches['bounce_success'].sum()}")
    print(f"   Failed: {len(touches) - touches['bounce_success'].sum()}")
    print(f"   Success rate: {touches['bounce_success'].mean():.1%}")
    
    print(f"\n2. By BVS Score Range")
    bvs_ranges = [(0, 0.5), (0.5, 0.65), (0.65, 0.80), (0.80, 1.0)]
    for low, high in bvs_ranges:
        range_touches = touches[(touches['bvs'] >= low) & (touches['bvs'] < high)]
        if len(range_touches) > 0:
            success_rate = range_touches['bounce_success'].mean()
            print(f"   BVS {low:.2f}-{high:.2f}: {len(range_touches)} touches, success {success_rate:.1%}")
    
    print(f"\n3. Time to Target")
    success_touches = touches[touches['bounce_success'] == 1]
    if len(success_touches) > 0 and success_touches['bars_to_target'].notna().sum() > 0:
        bars = success_touches['bars_to_target'].dropna()
        print(f"   Average: {bars.mean():.1f} candles")
        print(f"   Median: {bars.median():.1f} candles")
        print(f"   Min: {bars.min():.0f} candles")
        print(f"   Max: {bars.max():.0f} candles")


def example_3_feature_analysis():
    """Example 3: Analyze feature distributions"""
    print("\n" + "="*60)
    print("Example 3: Feature Analysis")
    print("="*60)
    
    symbols = ['BTCUSDT', 'ETHUSDT']
    all_data = process_all_symbols_and_timeframes(symbols, ['15m'])
    
    if not all_data:
        print("No data processed")
        return
    
    training_df = create_training_dataframe(all_data)
    
    if len(training_df) == 0:
        print("No training data generated")
        return
    
    print(f"\n1. Core Feature Statistics")
    core_features = ['BVS', 'PEI', 'MESS', 'SSS']
    for feature in core_features:
        if feature in training_df.columns:
            print(f"\n   {feature}:")
            print(f"      Mean: {training_df[feature].mean():.3f}")
            print(f"      Std: {training_df[feature].std():.3f}")
            print(f"      Min: {training_df[feature].min():.3f}")
            print(f"      Max: {training_df[feature].max():.3f}")
    
    print(f"\n2. Success vs Failure Comparison")
    for feature in core_features:
        if feature in training_df.columns:
            success_mean = training_df[training_df['bounce_success'] == 1][feature].mean()
            fail_mean = training_df[training_df['bounce_success'] == 0][feature].mean()
            print(f"   {feature}: Success {success_mean:.3f} vs Fail {fail_mean:.3f}")


def example_4_batch_processing():
    """Example 4: Batch process multiple symbols"""
    print("\n" + "="*60)
    print("Example 4: Batch Processing Multiple Symbols")
    print("="*60)
    
    symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']
    timeframes = ['15m']
    
    print(f"\nProcessing {len(symbols)} symbols, {len(timeframes)} timeframes...")
    
    all_data = process_all_symbols_and_timeframes(symbols, timeframes)
    
    print(f"\nSuccessfully processed: {len(all_data)} datasets")
    
    training_df = create_training_dataframe(all_data)
    
    print(f"\nTraining Set Statistics:")
    print(f"   Total samples: {len(training_df)}")
    print(f"   Successful bounces: {training_df['bounce_success'].sum()}")
    print(f"   Failed bounces: {len(training_df) - training_df['bounce_success'].sum()}")
    print(f"   Success rate: {training_df['bounce_success'].mean():.1%}")
    print(f"   Features: {len(training_df.columns)}")
    
    training_df.to_parquet('training_data_example.parquet', index=False)
    print(f"\n   Saved to training_data_example.parquet")
    
    return training_df


def example_5_data_validation():
    """Example 5: Validate data quality"""
    print("\n" + "="*60)
    print("Example 5: Data Quality Validation")
    print("="*60)
    
    symbols = ['BTCUSDT']
    all_data = process_all_symbols_and_timeframes(symbols, ['15m'])
    
    if not all_data:
        print("No data to validate")
        return
    
    training_df = create_training_dataframe(all_data)
    
    if len(training_df) == 0:
        print("No training data to validate")
        return
    
    validate_labeled_data(training_df)


def example_6_real_time_feature_extraction():
    """Example 6: Extract features for real-time trading"""
    print("\n" + "="*60)
    print("Example 6: Real-Time Feature Extraction")
    print("="*60)
    
    print("\nDownloading data...")
    df = download_ohlcv_data('BTCUSDT', '15m')
    
    if df is None or len(df) < 50:
        print("Insufficient data")
        return
    
    df = calculate_all_indicators(df)
    
    print("\nSearching for bounce touch point...")
    touch_idx = None
    for i in range(20, len(df) - 10):
        if df.iloc[i]['close'] <= df.iloc[i]['bb_lower'] * 1.001:
            touch_idx = i
            break
    
    if touch_idx is None:
        print("No touch point found")
        return
    
    print(f"\nFound touch at index {touch_idx}")
    print(f"Time: {df.index[touch_idx]}")
    print(f"Price: {df.iloc[touch_idx]['close']:.2f}")
    
    print(f"\nExtracting features...")
    features = extract_features_at_touch(df, touch_idx, direction='long')
    
    print(f"\nKey Features:")
    key_features = ['BVS', 'PEI', 'SSS', 'MESS', 'RSI_14', 'Volume_Ratio']
    for feat in key_features:
        if feat in features:
            print(f"   {feat}: {features[feat]:.3f}")
    
    bvs = features['BVS']
    print(f"\nDecision:")
    if bvs > 0.80:
        print(f"   BVS = {bvs:.3f} - STRONG BUY SIGNAL")
    elif bvs > 0.65:
        print(f"   BVS = {bvs:.3f} - BUY SIGNAL")
    elif bvs > 0.50:
        print(f"   BVS = {bvs:.3f} - WEAK BUY SIGNAL")
    else:
        print(f"   BVS = {bvs:.3f} - AVOID ENTRY")


def example_7_data_export():
    """Example 7: Export data in multiple formats"""
    print("\n" + "="*60)
    print("Example 7: Data Export")
    print("="*60)
    
    symbols = ['BTCUSDT']
    all_data = process_all_symbols_and_timeframes(symbols, ['15m'])
    
    if not all_data:
        print("No data to export")
        return
    
    training_df = create_training_dataframe(all_data)
    
    if len(training_df) == 0:
        print("No training data to export")
        return
    
    print("\nExporting in multiple formats...")
    
    training_df.to_parquet('export_example.parquet', index=False)
    print("   Exported: export_example.parquet (Parquet format)")
    
    training_df.to_csv('export_example.csv', index=False)
    print("   Exported: export_example.csv (CSV format)")
    
    summary = {
        'Total_Samples': len(training_df),
        'Successful_Bounces': training_df['bounce_success'].sum(),
        'Success_Rate': f"{training_df['bounce_success'].mean():.1%}",
        'Average_BVS': f"{training_df['BVS'].mean():.3f}",
        'Average_PEI': f"{training_df['PEI'].mean():.3f}",
    }
    
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv('export_summary.csv', index=False)
    print("   Exported: export_summary.csv (Summary statistics)")


def main():
    """Run all examples"""
    print("\n" + "="*60)
    print("BB-Bounce-ML-V2 Usage Examples")
    print("="*60)
    
    example_1_single_symbol()
    example_2_statistics()
    example_4_batch_processing()
    example_5_data_validation()
    example_6_real_time_feature_extraction()
    example_7_data_export()
    
    print("\n" + "="*60)
    print("All examples completed")
    print("="*60)


if __name__ == '__main__':
    main()
