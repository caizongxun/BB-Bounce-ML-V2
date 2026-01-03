"""
BB反彈ML系統 - 實時服務 V3
支持三層模型整合：BB觸及檢測 -> 有效性判別 -> 波動性預測
"""

import os
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime
import logging
import json
from collections import deque
from threading import Thread
import time

# 配置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# ============================================================
# 全局配置
# ============================================================

MODELS_DIR = Path('./models')
DATA_DIR = Path('./data')

SYMBOLS = [
    'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'XRPUSDT', 'DOGEUSDT',
    'SOLUSDT', 'LTCUSDT', 'AVAXUSDT', 'LINKUSDT', 'UNIUSDT', 'ATOMUSDT',
    'MATICUSDT', 'DOTUSDT', 'FILUSDT', 'AAVEUSDT', 'OPUSDT', 'ARBUSDT',
    'ALGOUSDT', 'NEARUSDT', 'BCHUSDT', 'ETCUSDT'
]

TIMEFRAMES = ['15m', '1h']

# ============================================================
# 模型管理器
# ============================================================

class ModelManager:
    """管理三層模型的加載和預測"""
    
    def __init__(self):
        self.bb_models = {}  # Position Classifier
        self.validity_models = {}  # Validity Detector
        self.vol_models = {}  # Volatility Predictor
        self.model_cache = {}  # 緩存已加載的模型
        self.load_all_models()
    
    def _get_model_path(self, model_type, symbol, timeframe):
        """獲取模型路徑"""
        base_path = MODELS_DIR / model_type
        return base_path / symbol / timeframe
    
    def _load_model_file(self, filepath, filename):
        """加載單個模型文件"""
        full_path = filepath / filename
        cache_key = str(full_path)
        
        if cache_key in self.model_cache:
            return self.model_cache[cache_key]
        
        if full_path.exists():
            try:
                with open(full_path, 'rb') as f:
                    model = pickle.load(f)
                self.model_cache[cache_key] = model
                logger.info(f"已加載: {cache_key}")
                return model
            except Exception as e:
                logger.error(f"加載失敗 {full_path}: {e}")
                return None
        return None
    
    def load_all_models(self):
        """加載所有模型"""
        logger.info("開始加載所有模型...")
        
        for symbol in SYMBOLS:
            for timeframe in TIMEFRAMES:
                # 加載 BB Position Classifier
                bb_path = self._get_model_path('bb_models', symbol, timeframe)
                if bb_path.exists():
                    self.bb_models[(symbol, timeframe)] = {
                        'model': self._load_model_file(bb_path, 'model.pkl'),
                        'scaler': self._load_model_file(bb_path, 'scaler.pkl'),
                        'label_map': self._load_model_file(bb_path, 'label_map.pkl')
                    }
                
                # 加載 Validity Detector
                validity_path = self._get_model_path('validity_models', symbol, timeframe)
                if validity_path.exists():
                    self.validity_models[(symbol, timeframe)] = {
                        'model': self._load_model_file(validity_path, 'validity_model.pkl'),
                        'scaler': self._load_model_file(validity_path, 'scaler.pkl'),
                        'feature_names': self._load_model_file(validity_path, 'feature_names.pkl')
                    }
                
                # 加載 Volatility Predictor
                vol_path = self._get_model_path('vol_models', symbol, timeframe)
                if vol_path.exists():
                    self.vol_models[(symbol, timeframe)] = {
                        'model': self._load_model_file(vol_path, 'model_regression.pkl'),
                        'scaler': self._load_model_file(vol_path, 'scaler_regression.pkl')
                    }
        
        logger.info(f"模型加載完成: {len(self.bb_models)} BB, {len(self.validity_models)} Validity, {len(self.vol_models)} Vol")
    
    def predict_bb_touch(self, symbol, timeframe, features):
        """
        層級1：預測是否觸碰到軌道
        返回: {'touched': bool, 'touch_type': 'upper'|'lower'|'none', 'confidence': float}
        """
        key = (symbol, timeframe)
        if key not in self.bb_models:
            return None
        
        models = self.bb_models[key]
        if not models['model'] or not models['scaler']:
            return None
        
        try:
            # 特徵縮放
            features_scaled = models['scaler'].transform([features])
            
            # 預測
            prediction = models['model'].predict(features_scaled)[0]
            confidence = max(models['model'].predict_proba(features_scaled)[0])
            
            # 轉換標籤
            label_map = models['label_map'] or {0: 'none', 1: 'upper', 2: 'lower'}
            touch_type = label_map.get(prediction, 'none')
            
            touched = touch_type != 'none'
            
            return {
                'touched': touched,
                'touch_type': touch_type,
                'confidence': float(confidence),
                'prediction': int(prediction)
            }
        except Exception as e:
            logger.error(f"BB觸及預測失敗 {symbol} {timeframe}: {e}")
            return None
    
    def predict_validity(self, symbol, timeframe, features):
        """
        層級2：預測反彈有效性
        返回: {'valid': bool, 'probability': float, 'quality': 'excellent'|'good'|'moderate'|'weak'|'poor'}
        """
        key = (symbol, timeframe)
        if key not in self.validity_models:
            return None
        
        models = self.validity_models[key]
        if not models['model'] or not models['scaler']:
            return None
        
        try:
            # 特徵縮放
            features_scaled = models['scaler'].transform([features])
            
            # 預測概率
            proba = models['model'].predict_proba(features_scaled)[0]
            valid_prob = float(proba[1]) if len(proba) > 1 else 0.5
            
            # 判定有效性等級
            if valid_prob >= 0.75:
                quality = 'excellent'
            elif valid_prob >= 0.65:
                quality = 'good'
            elif valid_prob >= 0.50:
                quality = 'moderate'
            elif valid_prob >= 0.30:
                quality = 'weak'
            else:
                quality = 'poor'
            
            valid = valid_prob >= 0.50
            
            return {
                'valid': valid,
                'probability': valid_prob * 100,  # 轉換為百分比
                'quality': quality,
                'confidence': valid_prob
            }
        except Exception as e:
            logger.error(f"有效性預測失敗 {symbol} {timeframe}: {e}")
            return None
    
    def predict_volatility(self, symbol, timeframe, features):
        """
        層級3：預測波動性
        返回: {'predicted_vol': float, 'will_expand': bool, 'expansion_strength': float}
        """
        key = (symbol, timeframe)
        if key not in self.vol_models:
            return None
        
        models = self.vol_models[key]
        if not models['model'] or not models['scaler']:
            return None
        
        try:
            # 特徵縮放
            features_scaled = models['scaler'].transform([features])
            
            # 預測波動性
            predicted_vol = float(models['model'].predict(features_scaled)[0])
            
            # 判定是否會擴張（以平均波動性為基準）
            # 假設基準波動性為1.0，>1.2表示會擴張
            will_expand = predicted_vol > 1.2
            expansion_strength = max(0, (predicted_vol - 1.0) / 1.0)  # 正規化至0-1
            
            return {
                'predicted_vol': predicted_vol,
                'will_expand': will_expand,
                'expansion_strength': min(1.0, expansion_strength)  # 上限0-1
            }
        except Exception as e:
            logger.error(f"波動性預測失敗 {symbol} {timeframe}: {e}")
            return None


# 全局模型管理器
model_manager = ModelManager()

# ============================================================
# 特徵提取（簡化版本）
# ============================================================

class FeatureExtractor:
    """從原始數據提取特徵"""
    
    @staticmethod
    def extract_features(ohlcv_data):
        """
        提取基本特徵（需根據您的actual實現調整）
        ohlcv_data: {'open', 'high', 'low', 'close', 'volume'}
        """
        features = []
        
        try:
            # 基本特徵
            o, h, l, c, v = ohlcv_data.get('open'), ohlcv_data.get('high'), \
                            ohlcv_data.get('low'), ohlcv_data.get('close'), \
                            ohlcv_data.get('volume')
            
            # 1. 價格相關特徵
            body_ratio = (c - o) / (h - l + 1e-8)  # K棒實體比例
            wick_ratio = min(h - max(o, c), min(o, c) - l) / (h - l + 1e-8)  # 影線比例
            high_low_range = (h - l) / c  # 幅度比例
            close_position = (c - l) / (h - l + 1e-8)  # 收盤位置
            
            features.extend([body_ratio, wick_ratio, high_low_range, close_position])
            
            # 2. 成交量特徵
            vol_norm = v / (1e6 + 1e-8)  # 正規化成交量
            features.append(vol_norm)
            
            # 3. 簡單動能指標
            price_slope = (c - o) / o  # 價格變化率
            features.append(price_slope)
            
            # 4. 時間特徵
            hour = datetime.now().hour
            is_high_volume_time = 1 if (hour >= 8 and hour <= 12) or (hour >= 20 and hour <= 23) else 0
            features.extend([hour, is_high_volume_time])
            
            # 5. 補充特徵至16個（根據您的模型需要調整）
            while len(features) < 16:
                features.append(0.0)
            
            return np.array(features[:16], dtype=np.float32)
        
        except Exception as e:
            logger.error(f"特徵提取失敗: {e}")
            return np.zeros(16, dtype=np.float32)


# ============================================================
# API 端點
# ============================================================

@app.route('/health', methods=['GET'])
def health_check():
    """健康檢查"""
    return jsonify({
        'status': 'ok',
        'timestamp': datetime.now().isoformat(),
        'models_loaded': {
            'bb_models': len(model_manager.bb_models),
            'validity_models': len(model_manager.validity_models),
            'vol_models': len(model_manager.vol_models)
        }
    })


@app.route('/predict', methods=['POST'])
def predict():
    """
    完整的三層預測端點
    
    請求格式:
    {
        "symbol": "BTCUSDT",
        "timeframe": "15m",
        "ohlcv": {"open": 45000, "high": 45500, "low": 44900, "close": 45200, "volume": 1000000}
    }
    
    返回格式:
    {
        "symbol": "BTCUSDT",
        "timeframe": "15m",
        "bb_touch": {...},
        "validity": {...},
        "volatility": {...},
        "signal": "BUY|SELL|NEUTRAL|HOLD"
    }
    """
    try:
        data = request.get_json()
        
        symbol = data.get('symbol', '').upper()
        timeframe = data.get('timeframe', '15m')
        ohlcv = data.get('ohlcv', {})
        
        # 驗證輸入
        if not symbol or symbol not in SYMBOLS:
            return jsonify({'error': f'無效的幣種: {symbol}'}), 400
        
        if timeframe not in TIMEFRAMES:
            return jsonify({'error': f'無效的時間框架: {timeframe}'}), 400
        
        # 提取特徵
        features = FeatureExtractor.extract_features(ohlcv)
        
        # 層級1：檢測BB觸及
        bb_result = model_manager.predict_bb_touch(symbol, timeframe, features)
        if not bb_result or not bb_result['touched']:
            return jsonify({
                'symbol': symbol,
                'timeframe': timeframe,
                'bb_touch': bb_result or {'touched': False},
                'validity': None,
                'volatility': None,
                'signal': 'NEUTRAL'
            })
        
        # 層級2：檢測有效性
        validity_result = model_manager.predict_validity(symbol, timeframe, features)
        
        # 層級3：預測波動性
        vol_result = model_manager.predict_volatility(symbol, timeframe, features)
        
        # 生成交易信號
        signal = generate_signal(bb_result, validity_result, vol_result)
        
        return jsonify({
            'symbol': symbol,
            'timeframe': timeframe,
            'timestamp': datetime.now().isoformat(),
            'bb_touch': bb_result,
            'validity': validity_result,
            'volatility': vol_result,
            'signal': signal,
            'confidence': calculate_confidence(bb_result, validity_result, vol_result)
        })
    
    except Exception as e:
        logger.error(f"預測錯誤: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    """批量預測多個幣種"""
    try:
        data = request.get_json()
        symbols = data.get('symbols', SYMBOLS)
        timeframe = data.get('timeframe', '15m')
        ohlcv_data = data.get('ohlcv_data', {})  # {symbol: ohlcv}
        
        results = []
        for symbol in symbols:
            if symbol not in ohlcv_data:
                continue
            
            features = FeatureExtractor.extract_features(ohlcv_data[symbol])
            
            bb_result = model_manager.predict_bb_touch(symbol, timeframe, features)
            if not bb_result or not bb_result['touched']:
                continue
            
            validity_result = model_manager.predict_validity(symbol, timeframe, features)
            vol_result = model_manager.predict_volatility(symbol, timeframe, features)
            
            signal = generate_signal(bb_result, validity_result, vol_result)
            
            results.append({
                'symbol': symbol,
                'timeframe': timeframe,
                'bb_touch': bb_result,
                'validity': validity_result,
                'volatility': vol_result,
                'signal': signal
            })
        
        return jsonify({
            'timestamp': datetime.now().isoformat(),
            'results': results,
            'count': len(results)
        })
    
    except Exception as e:
        logger.error(f"批量預測錯誤: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


# ============================================================
# 輔助函數
# ============================================================

def generate_signal(bb_result, validity_result, vol_result):
    """
    根據三層模型結果生成交易信號
    """
    if not bb_result or not bb_result['touched']:
        return 'NEUTRAL'
    
    if not validity_result:
        return 'NEUTRAL'
    
    # 基於有效性
    quality_score = {
        'excellent': 4,
        'good': 3,
        'moderate': 2,
        'weak': 1,
        'poor': 0
    }.get(validity_result.get('quality', 'poor'), 0)
    
    # 基於波動性
    vol_score = 0
    if vol_result:
        if vol_result.get('will_expand'):
            vol_score = 2
        else:
            vol_score = 1
    
    total_score = quality_score + vol_score
    
    # 生成信號
    if total_score >= 5:
        return 'STRONG_BUY' if bb_result['touch_type'] == 'lower' else 'STRONG_SELL'
    elif total_score >= 3:
        return 'BUY' if bb_result['touch_type'] == 'lower' else 'SELL'
    elif total_score >= 1:
        return 'HOLD'
    else:
        return 'NEUTRAL'


def calculate_confidence(bb_result, validity_result, vol_result):
    """計算整體信心度（0-1）"""
    if not bb_result:
        return 0.0
    
    confidence = bb_result.get('confidence', 0.5) * 0.3  # BB權重30%
    
    if validity_result:
        confidence += (validity_result.get('confidence', 0.5)) * 0.5  # Validity權重50%
    
    if vol_result:
        confidence += (vol_result.get('expansion_strength', 0.5)) * 0.2  # Vol權重20%
    
    return min(1.0, confidence)


# ============================================================
# 主程序
# ============================================================

if __name__ == '__main__':
    logger.info("=" * 60)
    logger.info("BB反彈ML系統 - 實時服務 V3")
    logger.info("=" * 60)
    logger.info("模型架構：")
    logger.info("  層級1: BB Position Classifier (觸及檢測)")
    logger.info("  層級2: Validity Detector (有效性判別)")
    logger.info("  層級3: Volatility Predictor (波動性預測)")
    logger.info("=" * 60)
    
    # 啟動Flask服務
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=False,
        threaded=True
    )
