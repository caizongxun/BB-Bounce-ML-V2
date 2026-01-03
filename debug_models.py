#!/usr/bin/env python
# 完整的模型調試工具 - 逐步診斷三層模型

import os
import pickle
import joblib
import numpy as np
from pathlib import Path
import json

MODELS_DIR = Path('./models')
SYMBOL = 'ETHUSDT'
TIMEFRAME = '15m'

print("="*70)
print("BB 反彈 ML 模型調試工具")
print("="*70)

# ============================================================
# 1. 檢查目錄結構
# ============================================================
print("\n[步驟 1] 檢查目錄結構")
print("-" * 70)

for model_type in ['bb_models', 'validity_models', 'vol_models']:
    base_path = MODELS_DIR / model_type / SYMBOL / TIMEFRAME
    print(f"\n{model_type} 路徑: {base_path.absolute()}")
    print(f"  存在: {base_path.exists()}")
    
    if base_path.exists():
        files = list(base_path.iterdir())
        print(f"  檔案數量: {len(files)}")
        for f in files:
            size_kb = f.stat().st_size / 1024
            print(f"    - {f.name} ({size_kb:.1f} KB)")
    else:
        print("  [警告] 目錄不存在！")

# ============================================================
# 2. 加載模型
# ============================================================
print("\n" + "="*70)
print("[步驟 2] 加載模型")
print("-" * 70)

models_loaded = {}

# 2.1 加載 BB 模型
print("\n2.1 加載 BB 觸及分類模型")
bb_path = MODELS_DIR / 'bb_models' / SYMBOL / TIMEFRAME
bb_model = None
bb_scaler = None
bb_label_map = None

if bb_path.exists():
    try:
        bb_model_file = bb_path / 'model.pkl'
        bb_scaler_file = bb_path / 'scaler.pkl'
        bb_label_map_file = bb_path / 'label_map.pkl'
        
        if bb_model_file.exists():
            bb_model = joblib.load(bb_model_file)
            print(f"  model.pkl: 成功 ✓")
            print(f"    - 類型: {type(bb_model)}")
            print(f"    - 類別數: {len(bb_model.classes_) if hasattr(bb_model, 'classes_') else 'N/A'}")
            if hasattr(bb_model, 'classes_'):
                print(f"    - 類別列表: {bb_model.classes_}")
        else:
            print(f"  model.pkl: 不存在 ✗")
        
        if bb_scaler_file.exists():
            bb_scaler = joblib.load(bb_scaler_file)
            print(f"  scaler.pkl: 成功 ✓")
            print(f"    - 類型: {type(bb_scaler)}")
            if hasattr(bb_scaler, 'n_features_in_'):
                print(f"    - 輸入特徵數: {bb_scaler.n_features_in_}")
        else:
            print(f"  scaler.pkl: 不存在 ✗")
        
        if bb_label_map_file.exists():
            bb_label_map = joblib.load(bb_label_map_file)
            print(f"  label_map.pkl: 成功 ✓")
            print(f"    - 內容: {bb_label_map}")
            print(f"    - 鍵類型: {type(list(bb_label_map.keys())[0]) if bb_label_map else 'N/A'}")
        else:
            print(f"  label_map.pkl: 不存在 (可選)")
            bb_label_map = {0: 'lower', 1: 'none', 2: 'upper'}
            print(f"    - 使用預設 label_map")
        
        models_loaded['bb'] = True
    except Exception as e:
        print(f"  [錯誤] {e}")
        models_loaded['bb'] = False
else:
    print(f"  [警告] {bb_path} 不存在")
    models_loaded['bb'] = False

# 2.2 加載 Validity 模型
print("\n2.2 加載有效性判別模型")
validity_path = MODELS_DIR / 'validity_models' / SYMBOL / TIMEFRAME
validity_model = None
validity_scaler = None

if validity_path.exists():
    try:
        validity_model_file = validity_path / 'validity_model.pkl'
        validity_scaler_file = validity_path / 'scaler.pkl'
        
        if validity_model_file.exists():
            validity_model = joblib.load(validity_model_file)
            print(f"  validity_model.pkl: 成功 ✓")
            print(f"    - 類型: {type(validity_model)}")
            if hasattr(validity_model, 'classes_'):
                print(f"    - 類別: {validity_model.classes_}")
        else:
            print(f"  validity_model.pkl: 不存在 ✗")
        
        if validity_scaler_file.exists():
            validity_scaler = joblib.load(validity_scaler_file)
            print(f"  scaler.pkl: 成功 ✓")
            print(f"    - 類型: {type(validity_scaler)}")
            if hasattr(validity_scaler, 'n_features_in_'):
                print(f"    - 輸入特徵數: {validity_scaler.n_features_in_}")
        else:
            print(f"  scaler.pkl: 不存在 ✗")
        
        models_loaded['validity'] = True
    except Exception as e:
        print(f"  [錯誤] {e}")
        models_loaded['validity'] = False
else:
    print(f"  [警告] {validity_path} 不存在")
    models_loaded['validity'] = False

# 2.3 加載 Volatility 模型
print("\n2.3 加載波動性預測模型")
vol_path = MODELS_DIR / 'vol_models' / SYMBOL / TIMEFRAME
vol_model = None
vol_scaler = None

if vol_path.exists():
    try:
        vol_model_file = vol_path / 'model_regression.pkl'
        vol_scaler_file = vol_path / 'scaler_regression.pkl'
        
        if vol_model_file.exists():
            vol_model = joblib.load(vol_model_file)
            print(f"  model_regression.pkl: 成功 ✓")
            print(f"    - 類型: {type(vol_model)}")
        else:
            print(f"  model_regression.pkl: 不存在 ✗")
        
        if vol_scaler_file.exists():
            vol_scaler = joblib.load(vol_scaler_file)
            print(f"  scaler_regression.pkl: 成功 ✓")
            print(f"    - 類型: {type(vol_scaler)}")
            if hasattr(vol_scaler, 'n_features_in_'):
                print(f"    - 輸入特徵數: {vol_scaler.n_features_in_}")
        else:
            print(f"  scaler_regression.pkl: 不存在 ✗")
        
        models_loaded['vol'] = True
    except Exception as e:
        print(f"  [錯誤] {e}")
        models_loaded['vol'] = False
else:
    print(f"  [警告] {vol_path} 不存在")
    models_loaded['vol'] = False

# ============================================================
# 3. 測試模型預測
# ============================================================
print("\n" + "="*70)
print("[步驟 3] 測試模型預測")
print("-" * 70)

# 生成虛擬數據
test_features_bb = np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.85, 0.75]], dtype=np.float32)
test_features_validity = np.random.randn(1, 17).astype(np.float32)  # 假設有 17 個特徵
test_features_vol = np.random.randn(1, 15).astype(np.float32)  # 假設有 15 個特徵

print(f"\n3.1 BB 觸及分類預測")
if bb_model and bb_scaler:
    try:
        features_scaled = bb_scaler.transform(test_features_bb)
        prediction = bb_model.predict(features_scaled)[0]
        probabilities = bb_model.predict_proba(features_scaled)[0]
        
        print(f"  預測類別: {prediction}")
        print(f"  類別概率: {probabilities}")
        print(f"  信心度: {np.max(probabilities):.4f}")
        
        # 轉換為標籤
        best_class = np.argmax(probabilities)
        label_map_str = {str(k): str(v) for k, v in bb_label_map.items()}
        touch_type = label_map_str.get(str(best_class), 'unknown')
        touched = (best_class != 1) and (np.max(probabilities) > 0.3)
        
        print(f"  最高概率類別: {best_class}")
        print(f"  對應標籤: {touch_type}")
        print(f"  是否觸及: {touched}")
        print(f"  預測成功 ✓")
    except Exception as e:
        print(f"  [錯誤] {e}")
else:
    print(f"  [跳過] 模型未加載")

print(f"\n3.2 有效性判別預測")
if validity_model and validity_scaler:
    try:
        # 檢查特徵數量
        expected_features = validity_scaler.n_features_in_ if hasattr(validity_scaler, 'n_features_in_') else 17
        if test_features_validity.shape[1] != expected_features:
            test_features_validity = np.random.randn(1, expected_features).astype(np.float32)
        
        features_scaled = validity_scaler.transform(test_features_validity)
        proba = validity_model.predict_proba(features_scaled)[0]
        valid_prob = float(proba[1]) if len(proba) > 1 else 0.5
        
        print(f"  預測概率 (有效): {valid_prob:.4f}")
        print(f"  預測概率 (無效): {proba[0]:.4f}")
        
        if valid_prob >= 0.75:
            quality = 'excellent'
        elif valid_prob >= 0.65:
            quality = 'good'
        elif valid_prob >= 0.50:
            quality = 'moderate'
        else:
            quality = 'poor'
        
        print(f"  品質等級: {quality}")
        print(f"  預測成功 ✓")
    except Exception as e:
        print(f"  [錯誤] {e}")
else:
    print(f"  [跳過] 模型未加載")

print(f"\n3.3 波動性預測")
if vol_model and vol_scaler:
    try:
        # 檢查特徵數量
        expected_features = vol_scaler.n_features_in_ if hasattr(vol_scaler, 'n_features_in_') else 15
        if test_features_vol.shape[1] != expected_features:
            test_features_vol = np.random.randn(1, expected_features).astype(np.float32)
        
        features_scaled = vol_scaler.transform(test_features_vol)
        predicted_vol = float(vol_model.predict(features_scaled)[0])
        
        print(f"  預測波動性: {predicted_vol:.4f}")
        print(f"  會擴張: {predicted_vol > 1.2}")
        print(f"  擴張強度: {max(0, (predicted_vol - 1.0) / 1.0):.4f}")
        print(f"  預測成功 ✓")
    except Exception as e:
        print(f"  [錯誤] {e}")
else:
    print(f"  [跳過] 模型未加載")

# ============================================================
# 4. 總結
# ============================================================
print("\n" + "="*70)
print("[步驟 4] 調試總結")
print("-" * 70)

print(f"\nBB 模型加載: {'✓' if models_loaded.get('bb') else '✗'}")
print(f"Validity 模型加載: {'✓' if models_loaded.get('validity') else '✗'}")
print(f"Vol 模型加載: {'✓' if models_loaded.get('vol') else '✗'}")

print(f"\n[建議]")
if not all(models_loaded.values()):
    print("1. 缺少某些模型，請檢查:")
    print(f"   - {MODELS_DIR.absolute()} 目錄結構")
    print(f"   - 確保在 {SYMBOL}/{TIMEFRAME} 下有所有必要檔案")
    print("2. 執行訓練腳本重新生成模型")
else:
    print("✓ 所有模型已成功加載")
    print("✓ 預測管道正常運作")
    print("\n後續步驟:")
    print("1. 檢查 /predict API 端點")
    print("2. 檢查輸入特徵數量是否正確")
    print("3. 檢查特徵提取邏輯")

print("\n" + "="*70)
