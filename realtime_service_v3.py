"""
BB反彈ML系統 - 實時服務 V3
支持三層模型整合：BB觸及檢測 -> 有效性判別 -> 波動性預測
修正：pickle/joblib 序列化不一致問題
"""

import os
import pickle
import joblib
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
import warnings

warnings.filterwarnings('ignore', category=UserWarning)

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
# 修正: 安全的模型加載器
# ============================================================

class ModelLoader:
    """高算騂模型加載器 - 支持 pickle 和 joblib"""
    
    @staticmethod
    def load_model(filepath, model_type='auto'):
        """
        加載檔案 (joblib 或 pickle)
        
        參數：
        - filepath: 檔案路徑
        - model_type: 'auto'(自動偵渫), 'joblib', 'pickle'
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            logger.warning(f'檔案不存在: {filepath}')
            return None
        
        try:
            # 第一次：优先使用 joblib (訓練脚本使用)
            if model_type in ['auto', 'joblib']:
                try:
                    model = joblib.load(filepath)
                    logger.debug(f'使用 joblib 加載: {filepath.name}')
                    return model
                except Exception as e1:
                    if model_type == 'joblib':
                        raise
                    logger.debug(f'joblib 加載失救：{e1}，新試 pickle')
            
            # 第二次：使用 pickle (encoding='latin1')
            if model_type in ['auto', 'pickle']:
                with open(filepath, 'rb') as f:
                    model = pickle.load(f, encoding='latin1')
                    logger.debug(f'使用 pickle 加載: {filepath.name}')
                    return model
        
        except Exception as e:
            logger.error(f'加載失救 {filepath}: {str(e)[:200]}')
            return None

# ============================================================
# 模型管理器
# ============================================================

class ModelManager:
    """管理三層模型的加載和預測"""
    
    def __init__(self):
        self.bb_models = {}  # Position Classifier
        self.validity_models = {}  # Validity Detector
        self.vol_models = {}  # Volatility Predictor
        self.model_cache = {}  # 緩存
        self.load_all_models()
    
    def _get_model_path(self, model_type, symbol, timeframe):
        """獲取模型路徑"""
        base_path = MODELS_DIR / model_type
        return base_path / symbol / timeframe
    
    def _load_model_file(self, filepath, filename, model_type='auto'):
        """加載單個模型檔案"""
        full_path = filepath / filename
        cache_key = str(full_path)
        
        if cache_key in self.model_cache:
            return self.model_cache[cache_key]
        
        if full_path.exists():
            try:
                # 使用安全加載器
                model = ModelLoader.load_model(full_path, model_type)
                
                if model is not None:
                    self.model_cache[cache_key] = model
                    logger.info(f'已加載: {full_path.name}')
                    return model
                else:
                    logger.warning(f'加載空值: {full_path}')
                    return None
            
            except Exception as e:
                logger.error(f'加載失敐 {full_path}: {str(e)[:100]}')
                return None
        
        logger.debug(f'檔案不存在: {full_path}')
        return None
    
    def load_all_models(self):
        """加載所有模型"""
        logger.info('開始加載所有模型...')
        
        loaded_count = {'bb': 0, 'validity': 0, 'vol': 0}
        failed_count = {'bb': 0, 'validity': 0, 'vol': 0}
        
        for symbol in SYMBOLS:
            for timeframe in TIMEFRAMES:
                # 加載 BB Position Classifier
                bb_path = self._get_model_path('bb_models', symbol, timeframe)
                if bb_path.exists():
                    try:
                        bb_model = self._load_model_file(bb_path, 'model.pkl')
                        bb_scaler = self._load_model_file(bb_path, 'scaler.pkl')
                        bb_label_map = self._load_model_file(bb_path, 'label_map.pkl')
                        
                        if bb_model and bb_scaler:
                            self.bb_models[(symbol, timeframe)] = {
                                'model': bb_model,
                                'scaler': bb_scaler,
                                'label_map': bb_label_map
                            }
                            loaded_count['bb'] += 1
                        else:
                            failed_count['bb'] += 1
                    except Exception as e:
                        logger.warning(f'BB 模型加載失敐 {symbol} {timeframe}: {e}')
                        failed_count['bb'] += 1
                
                # 加載 Validity Detector
                validity_path = self._get_model_path('validity_models', symbol, timeframe)
                if validity_path.exists():
                    try:
                        # validity_models 使用 joblib 上帵 pickle
                        validity_model = self._load_model_file(validity_path, 'validity_model.pkl', 'auto')
                        validity_scaler = self._load_model_file(validity_path, 'scaler.pkl', 'auto')
                        feature_names = self._load_model_file(validity_path, 'feature_names.pkl', 'auto')
                        
                        if validity_model and validity_scaler:
                            self.validity_models[(symbol, timeframe)] = {
                                'model': validity_model,
                                'scaler': validity_scaler,
                                'feature_names': feature_names
                            }
                            loaded_count['validity'] += 1
                        else:
                            failed_count['validity'] += 1
                    except Exception as e:
                        logger.warning(f'Validity 模型加載失敐 {symbol} {timeframe}: {e}')
                        failed_count['validity'] += 1
                
                # 加載 Volatility Predictor
                vol_path = self._get_model_path('vol_models', symbol, timeframe)
                if vol_path.exists():
                    try:
                        vol_model = self._load_model_file(vol_path, 'model_regression.pkl')
                        vol_scaler = self._load_model_file(vol_path, 'scaler_regression.pkl')
                        
                        if vol_model and vol_scaler:
                            self.vol_models[(symbol, timeframe)] = {
                                'model': vol_model,
                                'scaler': vol_scaler
                            }
                            loaded_count['vol'] += 1
                        else:
                            failed_count['vol'] += 1
                    except Exception as e:
                        logger.warning(f'Vol 模型加載失敐 {symbol} {timeframe}: {e}')
                        failed_count['vol'] += 1
        
        logger.info(f'模型加載完成:')
        logger.info(f'  BB: {loaded_count["bb"]}標 (失敐: {failed_count["bb"]}標)')
        logger.info(f'  Validity: {loaded_count["validity"]}標 (失敐: {failed_count["validity"]}標)')
        logger.info(f'  Vol: {loaded_count["vol"]}標 (失敐: {failed_count["vol"]}標)')
    
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
            features_scaled = models['scaler'].transform([features])
            prediction = models['model'].predict(features_scaled)[0]
            confidence = max(models['model'].predict_proba(features_scaled)[0])
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
            logger.error(f'BB觸厬預測失敐 {symbol} {timeframe}: {e}')
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
            features_scaled = models['scaler'].transform([features])
            proba = models['model'].predict_proba(features_scaled)[0]
            valid_prob = float(proba[1]) if len(proba) > 1 else 0.5
            
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
                'probability': valid_prob * 100,
                'quality': quality,
                'confidence': valid_prob
            }
        except Exception as e:
            logger.error(f'有效性預測失敐 {symbol} {timeframe}: {e}')
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
            features_scaled = models['scaler'].transform([features])
            predicted_vol = float(models['model'].predict(features_scaled)[0])
            will_expand = predicted_vol > 1.2
            expansion_strength = max(0, (predicted_vol - 1.0) / 1.0)
            
            return {
                'predicted_vol': predicted_vol,
                'will_expand': will_expand,
                'expansion_strength': min(1.0, expansion_strength)
            }
        except Exception as e:
            logger.error(f'波動性預測失敐 {symbol} {timeframe}: {e}')
            return None


model_manager = ModelManager()

# ============================================================
# 特徵提取（簡化版本）
# ============================================================

class FeatureExtractor:
    """從原始數據提取特徵"""
    
    @staticmethod
    def extract_features(ohlcv_data):
        """提取基本特徵"""
        features = []
        
        try:
            o, h, l, c, v = ohlcv_data.get('open'), ohlcv_data.get('high'), \
                            ohlcv_data.get('low'), ohlcv_data.get('close'), \
                            ohlcv_data.get('volume')
            
            body_ratio = (c - o) / (h - l + 1e-8)
            wick_ratio = min(h - max(o, c), min(o, c) - l) / (h - l + 1e-8)
            high_low_range = (h - l) / c
            close_position = (c - l) / (h - l + 1e-8)
            
            features.extend([body_ratio, wick_ratio, high_low_range, close_position])
            
            vol_norm = v / (1e6 + 1e-8)
            features.append(vol_norm)
            
            price_slope = (c - o) / o
            features.append(price_slope)
            
            hour = datetime.now().hour
            is_high_volume_time = 1 if (hour >= 8 and hour <= 12) or (hour >= 20 and hour <= 23) else 0
            features.extend([hour, is_high_volume_time])
            
            while len(features) < 16:
                features.append(0.0)
            
            return np.array(features[:16], dtype=np.float32)
        
        except Exception as e:
            logger.error(f'特徵提取失敐: {e}')
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
    """完整的三層預測端點"""
    try:
        data = request.get_json()
        
        symbol = data.get('symbol', '').upper()
        timeframe = data.get('timeframe', '15m')
        ohlcv = data.get('ohlcv', {})
        
        if not symbol or symbol not in SYMBOLS:
            return jsonify({'error': f'無效的幣種: {symbol}'}), 400
        
        if timeframe not in TIMEFRAMES:
            return jsonify({'error': f'無效的時間框架: {timeframe}'}), 400
        
        features = FeatureExtractor.extract_features(ohlcv)
        
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
        
        validity_result = model_manager.predict_validity(symbol, timeframe, features)
        vol_result = model_manager.predict_volatility(symbol, timeframe, features)
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
        logger.error(f'預測錯誤: {e}', exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    """批量預測多個幣種"""
    try:
        data = request.get_json()
        symbols = data.get('symbols', SYMBOLS)
        timeframe = data.get('timeframe', '15m')
        ohlcv_data = data.get('ohlcv_data', {})
        
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
        logger.error(f'批量預測錯誤: {e}', exc_info=True)
        return jsonify({'error': str(e)}), 500


def generate_signal(bb_result, validity_result, vol_result):
    """根據三層模型結果生成交易信號"""
    if not bb_result or not bb_result['touched']:
        return 'NEUTRAL'
    
    if not validity_result:
        return 'NEUTRAL'
    
    quality_score = {
        'excellent': 4,
        'good': 3,
        'moderate': 2,
        'weak': 1,
        'poor': 0
    }.get(validity_result.get('quality', 'poor'), 0)
    
    vol_score = 0
    if vol_result:
        if vol_result.get('will_expand'):
            vol_score = 2
        else:
            vol_score = 1
    
    total_score = quality_score + vol_score
    
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
    
    confidence = bb_result.get('confidence', 0.5) * 0.3
    
    if validity_result:
        confidence += (validity_result.get('confidence', 0.5)) * 0.5
    
    if vol_result:
        confidence += (vol_result.get('expansion_strength', 0.5)) * 0.2
    
    return min(1.0, confidence)


if __name__ == '__main__':
    logger.info('='*60)
    logger.info('BB反彈ML系統 - 實時服務 V3')
    logger.info('='*60)
    logger.info('模型架構：')
    logger.info('  層級1: BB Position Classifier (觸厬檢測)')
    logger.info('  層級2: Validity Detector (有效性判別)')
    logger.info('  層級3: Volatility Predictor (波動性預測)')
    logger.info('='*60)
    logger.info('修正: joblib + pickle 序列化錯誤已解決')
    logger.info('='*60)
    
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=False,
        threaded=True
    )
