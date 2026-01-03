#!/usr/bin/env python3
"""
增強検測器測試脚本

此脚本演示如何使用 RealtimeBBDetectorV2Enhanced 進行滥釥。
"""

import sys
from pathlib import Path
from datetime import datetime
import json
from realtime_detector_v2_enhanced import RealtimeBBDetectorV2Enhanced


def test_model_loading():
    """
    測試 1：模型加載
    """
    print("\n" + "="*60)
    print("Test 1: Model Loading")
    print("="*60)
    
    detector = RealtimeBBDetectorV2Enhanced(model_dir="models")
    
    # 確認至少有一個符號可以加載
    loaded_symbols = [
        sym for sym in ["BTCUSDT", "ETHUSDT", "AAVEUSDT"]
        if (sym in detector.classifiers and "15m" in detector.classifiers[sym]) or 
           (sym in detector.validity_models and "15m" in detector.validity_models[sym])
    ]
    
    print(f"\nLoaded symbols: {len(loaded_symbols)}")
    if loaded_symbols:
        print(f"  Examples: {loaded_symbols[:3]}")
    else:
        print("  WARNING: No models loaded! Check models/ directory.")
    
    # 確認波動模型
    vol_symbols = [sym for sym in detector.vol_models.keys()]
    print(f"\nVolatility models: {len(vol_symbols)}")
    if vol_symbols:
        print(f"  Examples: {vol_symbols[:3]}")
    else:
        print("  INFO: No vol models loaded (expected if not trained yet)")
    
    return detector, len(loaded_symbols) > 0


def test_single_symbol(detector, symbol="BTCUSDT"):
    """
    測試 2：對單個符號進行 15m 掃描
    """
    print("\n" + "="*60)
    print(f"Test 2: Single Symbol Scan (15m) - {symbol}")
    print("="*60)
    
    # 模擬 10 根 15m K 線
    now = int(datetime.now().timestamp() * 1000)
    candles = []
    
    for i in range(10):
        candle = {
            "timestamp": now - (9 - i) * 15 * 60 * 1000,
            "open": 42500 + i * 50,
            "high": 42800 + i * 50,
            "low": 42200 + i * 50,
            "close": 42600 + i * 50,
            "volume": 120 + i * 5,
            "bb_upper": 43000 + i * 100,
            "bb_middle": 42500 + i * 50,
            "bb_lower": 42000 + i * 0,
            "rsi": 50 + i * 3,
            "adx": 20 + i * 0.5,
            "atr": 400 + i * 10,
        }
        candles.append(candle)
        detector.add_candle(symbol, candle, timeframe="15m")
    
    print(f"\nAdded {len(candles)} candles to buffer")
    
    # 掃描 15m
    signal = detector.scan(symbol, timeframe="15m")
    
    if signal:
        print(f"\n15m Signal Found:")
        print(json.dumps(signal, indent=2))
    else:
        print("\n15m Signal: None (may need different data or more candles)")
    
    return signal


def test_multi_timeframe(detector, symbol="BTCUSDT"):
    """
    測試 3：多時間活侨 (15m + 1h)
    """
    print("\n" + "="*60)
    print(f"Test 3: Multi-Timeframe Scan - {symbol}")
    print("="*60)
    
    # 模擬 1h 一根 K 線
    now = int(datetime.now().timestamp() * 1000)
    candle_1h = {
        "timestamp": now,
        "open": 42500,
        "high": 43000,
        "low": 42000,
        "close": 42700,
        "volume": 500,
        "bb_upper": 43200,
        "bb_middle": 42500,
        "bb_lower": 41800,
        "rsi": 60,
        "adx": 22,
        "atr": 500,
    }
    
    # 添加到 1h 緩衝區
    detector.add_candle(symbol, candle_1h, timeframe="1h")
    
    # 掃描 1h
    signal_1h = detector.scan(symbol, timeframe="1h")
    
    print(f"\n1h Buffer size: {len(detector.candle_buffer[symbol]['1h'])} candles")
    
    if signal_1h:
        print(f"\n1h Signal Found:")
        print(json.dumps(signal_1h, indent=2))
        
        # 獲取波動預測
        volatility = detector.predict_volatility(symbol)
        if volatility is not None:
            print(f"\nVolatility Prediction: {volatility:.4f} ({volatility*100:.2f}% daily)")
        else:
            print(f"\nVolatility Prediction: None (vol_model not available)")
    else:
        print(f"\n1h Signal: None")
    
    return signal_1h


def test_all_symbols(detector, limit=3):
    """
    測試 4：傳鏟掃描 (限制數量)
    """
    print("\n" + "="*60)
    print(f"Test 4: Batch Scanning (limit: {limit} symbols)")
    print("="*60)
    
    # 模擬數據
    now = int(datetime.now().timestamp() * 1000)
    candle = {
        "timestamp": now,
        "open": 42500,
        "high": 42800,
        "low": 42200,
        "close": 42600,
        "volume": 120,
        "bb_upper": 43000,
        "bb_middle": 42500,
        "bb_lower": 42000,
        "rsi": 55,
        "adx": 22,
        "atr": 400,
    }
    
    # 添加到選定的符號
    symbols_to_test = detector.symbols[:limit]
    for symbol in symbols_to_test:
        detector.add_candle(symbol, candle, timeframe="15m")
    
    print(f"\nScanning {len(symbols_to_test)} symbols...")
    
    signals = []
    for symbol in symbols_to_test:
        signal = detector.scan(symbol, timeframe="15m")
        if signal:
            signals.append(signal)
            print(f"  {symbol}: Signal found (confidence: {signal['confidence']:.2%})")
        else:
            print(f"  {symbol}: No signal")
    
    print(f"\nTotal signals: {len(signals)}/{len(symbols_to_test)}")
    return signals


def test_buffer_sizes(detector):
    """
    測試 5：緩衝區大小
    """
    print("\n" + "="*60)
    print("Test 5: Buffer Sizes")
    print("="*60)
    
    for symbol in detector.symbols[:3]:
        buffer_15m = len(detector.candle_buffer[symbol]["15m"])
        buffer_1h = len(detector.candle_buffer[symbol]["1h"])
        
        print(f"\n{symbol}:")
        print(f"  15m buffer: {buffer_15m} candles")
        print(f"  1h buffer: {buffer_1h} candles")


def main():
    """
    執行所有測試
    """
    print("\n" + "*" * 60)
    print("Enhanced RealtimeBBDetectorV2 Test Suite")
    print("*" * 60)
    
    # Test 1: Model Loading
    detector, has_models = test_model_loading()
    
    if not has_models:
        print("\n" + "!" * 60)
        print("WARNING: No models loaded!")
        print("Please run:")
        print("  python train_bb_model.py  (for bb_models/SYMBOL/15m/model.pkl)")
        print("  python train_validity_model.py  (for validity models)")
        print("  python train_vol_model.py  (for vol models)")
        print("!" * 60)
        return
    
    # Test 2: Single Symbol 15m
    test_single_symbol(detector, symbol="BTCUSDT")
    
    # Test 3: Multi-Timeframe
    test_multi_timeframe(detector, symbol="BTCUSDT")
    
    # Test 4: Batch Scanning
    test_all_symbols(detector, limit=3)
    
    # Test 5: Buffer Sizes
    test_buffer_sizes(detector)
    
    print("\n" + "*" * 60)
    print("All tests completed!")
    print("*" * 60 + "\n")


if __name__ == "__main__":
    main()
