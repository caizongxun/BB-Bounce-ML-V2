#!/usr/bin/env python3
"""
模型加載調試脚本
"""

import os
from pathlib import Path
import joblib

MODELS_DIR = Path('./models')
SYMBOLS = ['BTCUSDT', 'ETHUSDT']
TIMEFRAMES = ['15m', '1h']

print('="'*60)
print('BB反彈 ML 系統 - 模型加載調試')
print('="'*60)

print(f'\n模型目錄: {MODELS_DIR.absolute()}')
print(f'目錄是否存在: {MODELS_DIR.exists()}')

if MODELS_DIR.exists():
    print(f'\n目錄內容:')
    for item in MODELS_DIR.iterdir():
        print(f'  {item.name}/')

print('\n' + '='*60)
print('有效性模型 (有效性模型)')
print('='*60)

for symbol in SYMBOLS:
    for timeframe in TIMEFRAMES:
        model_path = MODELS_DIR / 'validity_models' / symbol / timeframe / 'validity_model.pkl'
        scaler_path = MODELS_DIR / 'validity_models' / symbol / timeframe / 'scaler.pkl'
        
        print(f'\n{symbol} {timeframe}:')
        print(f'  模型路徑: {model_path}')
        print(f'  模型存在: {model_path.exists()}')
        print(f'  Scaler存在: {scaler_path.exists()}')
        
        if model_path.exists() and scaler_path.exists():
            try:
                model = joblib.load(model_path)
                scaler = joblib.load(scaler_path)
                print(f'  ✅ 成功加載')
                print(f'    模型類型: {type(model).__name__}')
                print(f'    模型參數: n_estimators={getattr(model, "n_estimators", "N/A")}')
                print(f'    Scaler類型: {type(scaler).__name__}')
            except Exception as e:
                print(f'  ❌ 加載失敗: {e}')
        else:
            print(f'  ⚠️  模型或scaler不存在')

print('\n' + '='*60)
print('波動性模型')
print('='*60)

for symbol in SYMBOLS:
    for timeframe in TIMEFRAMES:
        model_path = MODELS_DIR / 'vol_models' / symbol / timeframe / 'model_regression.pkl'
        scaler_path = MODELS_DIR / 'vol_models' / symbol / timeframe / 'scaler_regression.pkl'
        
        print(f'\n{symbol} {timeframe}:')
        print(f'  模型路徑: {model_path}')
        print(f'  模型存在: {model_path.exists()}')
        print(f'  Scaler存在: {scaler_path.exists()}')
        
        if model_path.exists() and scaler_path.exists():
            try:
                model = joblib.load(model_path)
                scaler = joblib.load(scaler_path)
                print(f'  ✅ 成功加載')
                print(f'    模型類型: {type(model).__name__}')
                print(f'    Scaler類型: {type(scaler).__name__}')
            except Exception as e:
                print(f'  ❌ 加載失敖: {e}')
        else:
            print(f'  ⚠️  模型或scaler不存在')

print('\n' + '='*60)
print('結論')
print('='*60)
print('\n如果上面有模型显示 \u26a0️ \u4e0d存在，表示：')
print('1. 模型訓練源拋賺 按 train_validity_model.py 訓練')
print('2. 模型文件需要劾佐到 models/ 目錄')
print('\n客戶端感懿: 直接挥 F5 刷新且鐵棄企不明箫企物管')
