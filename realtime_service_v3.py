"""
BB反彈ML系統 - 實時服務 V3 (修計第三輬修載)
支持三層模型整合：BB觸厬檢測 + 有效性判別 + 波動性預測
修載：
1. pickle/joblib 序列化不一致問題
2. BB 模型特徵數量不匹配 (16 -> 12)
3. 波動性模型特徵數量不匹配 (3 -> 15)
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
# 修武：安全的模型加載器
# ============================================================

class ModelLoader:
    """高算騂模型加載器 - 支持 pickle 和 joblib"""
    
    @staticmethod
    def load_model(filepath, model_type='auto'):
        """
        加載檔案 (joblib 或 pickle)
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            logger.warning(f'檔案不存在: {filepath}')
            return None
        
        try:
            if model_type in ['auto', 'joblib']:
                try:
                    model = joblib.load(filepath)
                    logger.debug(f'使用 joblib 加載: {filepath.name}')
                    return model
                except Exception as e1:
                    if model_type == 'joblib':
                        raise
                    logger.debug(f'joblib 加載失敐，新試 pickle')
            
            if model_type in ['auto', 'pickle']:
                with open(filepath, 'rb') as f:
                    model = pickle.load(f, encoding='latin1')
                    logger.debug(f'使用 pickle 加載: {filepath.name}')
                    return model
        
        except Exception as e:
            logger.error(f'加載失敐 {filepath}: {str(e)[:200]}')
            return None

# ============================================================
# BB 特徵提取 (修警版本 - 実現歷式數據)
# ============================================================

class RealTimeFeatureExtractor:
    """
    從原始數據提取 BB 模型所需的 12 個特徵
    実現歷式數據快存機制
    """
    
    def __init__(self, history_size=50):
        self.history_size = max(history_size, 50)
        self.history = {}
        self.bb_period = 20
        self.bb_std = 2
        self.rsi_period = 14
    
    def update_history(self, symbol, timeframe, ohlcv_data):
        key = (symbol, timeframe)
        if key not in self.history:
            self.history[key] = deque(maxlen=self.history_size)
        self.history[key].append(ohlcv_data)
    
    def _get_close_prices(self, symbol, timeframe):
        key = (symbol, timeframe)
        if key not in self.history or len(self.history[key]) == 0:
            return None
        return np.array([d.get('close', 0) for d in list(self.history[key])])
    
    def _calculate_bb(self, symbol, timeframe):
        closes = self._get_close_prices(symbol, timeframe)
        if closes is None or len(closes) < self.bb_period:
            return None, None, None, None
        
        sma = np.mean(closes[-self.bb_period:])
        std = np.std(closes[-self.bb_period:])
        upper = sma + self.bb_std * std
        lower = sma - self.bb_std * std
        width = (upper - lower) / sma if sma != 0 else 0
        
        return upper, sma, lower, width
    
    def _calculate_rsi(self, symbol, timeframe):
        closes = self._get_close_prices(symbol, timeframe)
        if closes is None or len(closes) < self.rsi_period + 1:
            return 50
        
        deltas = np.diff(closes[-self.rsi_period-1:])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        
        if avg_loss == 0:
            rsi = 100 if avg_gain > 0 else 50
        else:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
        
        return max(0, min(100, rsi))
    
    def _calculate_volatility(self, symbol, timeframe):
        closes = self._get_close_prices(symbol, timeframe)
        if closes is None or len(closes) < 2:
            return 0
        
        returns = np.diff(closes) / closes[:-1]
        volatility = np.std(returns)
        return volatility
    
    def _calculate_sma(self, symbol, timeframe, period):
        closes = self._get_close_prices(symbol, timeframe)
        if closes is None or len(closes) < period:
            return 0
        return np.mean(closes[-period:])
    
    def extract_features(self, symbol, timeframe, ohlcv_data):
        """提取 12 個特徵"""
        self.update_history(symbol, timeframe, ohlcv_data)
        
        o = ohlcv_data.get('open', 0)
        h = ohlcv_data.get('high', 0)
        l = ohlcv_data.get('low', 0)
        c = ohlcv_data.get('close', 0)
        v = ohlcv_data.get('volume', 0)
        
        features = []
        
        # 1-4. BB 特徵
        bb_upper, bb_middle, bb_lower, bb_width = self._calculate_bb(symbol, timeframe)
        
        if bb_middle is not None and bb_middle != 0:
            price_to_bb_middle = (c - bb_middle) / bb_middle
            features.append(price_to_bb_middle)
            
            bb_range = bb_upper - bb_lower
            dist_upper_norm = (bb_upper - c) / bb_range if bb_range != 0 else 0
            features.append(dist_upper_norm)
            
            dist_lower_norm = (c - bb_lower) / bb_range if bb_range != 0 else 0
            features.append(dist_lower_norm)
            
            features.append(bb_width)
        else:
            features.extend([0, 0, 0, 0])
        
        # 5. rsi
        rsi = self._calculate_rsi(symbol, timeframe)
        features.append(rsi / 100)
        
        # 6. volatility
        volatility = self._calculate_volatility(symbol, timeframe)
        features.append(volatility)
        
        # 7. returns_std
        closes = self._get_close_prices(symbol, timeframe)
        if closes is not None and len(closes) >= 20:
            returns = np.diff(closes[-20:]) / closes[-21:-1]
            returns_std = np.std(returns)
        else:
            returns_std = 0
        features.append(returns_std)
        
        # 8. high_low_ratio
        high_low_ratio = (h / l - 1) if l != 0 else 0
        features.append(high_low_ratio)
        
        # 9. close_open_ratio
        close_open_ratio = (c / o - 1) if o != 0 else 0
        features.append(close_open_ratio)
        
        # 10-12. SMA
        features.append(self._calculate_sma(symbol, timeframe, 5))
        features.append(self._calculate_sma(symbol, timeframe, 20))
        features.append(self._calculate_sma(symbol, timeframe, 50))
        
        assert len(features) == 12, f'須有 12 個特徵, 但有 {len(features)} 個'
        
        return np.array(features, dtype=np.float32)

# ============================================================
# 波動性特徵提取 (修警版本 - 15 個特徵)
# ============================================================

class VolatilityFeatureExtractor:
    """
    從原始數據提取波動性模型所需的 15 個特徵
    """
    
    def __init__(self, history_size=50):
        self.history_size = max(history_size, 50)
        self.history = {}
    
    def update_history(self, symbol, timeframe, ohlcv_data):
        key = (symbol, timeframe)
        if key not in self.history:
            self.history[key] = deque(maxlen=self.history_size)
        self.history[key].append(ohlcv_data)
    
    def _get_closes(self, symbol, timeframe):
        key = (symbol, timeframe)
        if key not in self.history or len(self.history[key]) == 0:
            return None
        return np.array([d.get('close', 0) for d in list(self.history[key])])
    
    def _get_highs(self, symbol, timeframe):
        key = (symbol, timeframe)
        if key not in self.history or len(self.history[key]) == 0:
            return None
        return np.array([d.get('high', 0) for d in list(self.history[key])])
    
    def _get_lows(self, symbol, timeframe):
        key = (symbol, timeframe)
        if key not in self.history or len(self.history[key]) == 0:
            return None
        return np.array([d.get('low', 0) for d in list(self.history[key])])
    
    def _get_volumes(self, symbol, timeframe):
        key = (symbol, timeframe)
        if key not in self.history or len(self.history[key]) == 0:
            return None
        return np.array([d.get('volume', 0) for d in list(self.history[key])])
    
    def _calculate_rsi(self, symbol, timeframe, period=14):
        closes = self._get_closes(symbol, timeframe)
        if closes is None or len(closes) < period + 1:
            return 50
        
        deltas = np.diff(closes[-period-1:])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        
        if avg_loss == 0:
            return 100 if avg_gain > 0 else 50
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return max(0, min(100, rsi))
    
    def _calculate_atr(self, symbol, timeframe, period=14):
        closes = self._get_closes(symbol, timeframe)
        highs = self._get_highs(symbol, timeframe)
        lows = self._get_lows(symbol, timeframe)
        
        if closes is None or highs is None or lows is None or len(closes) < period + 1:
            return 0
        
        tr1 = highs[-period-1:] - lows[-period-1:]
        tr2 = np.abs(highs[-period-1:] - closes[-period-2:-1])
        tr3 = np.abs(lows[-period-1:] - closes[-period-2:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        atr = np.mean(tr[-period:])
        return atr
    
    def _calculate_stochastic(self, symbol, timeframe, period=14):
        closes = self._get_closes(symbol, timeframe)
        highs = self._get_highs(symbol, timeframe)
        lows = self._get_lows(symbol, timeframe)
        
        if closes is None or highs is None or lows is None or len(closes) < period:
            return 50, 50
        
        min_low = np.min(lows[-period:])
        max_high = np.max(highs[-period:])
        k_percent = 100 * (closes[-1] - min_low) / (max_high - min_low + 1e-8)
        d_percent = k_percent  # 粀化簡化
        
        return k_percent, d_percent
    
    def extract_features(self, symbol, timeframe, ohlcv_data):
        """提取 15 個波動性特徵"""
        self.update_history(symbol, timeframe, ohlcv_data)
        
        o = ohlcv_data.get('open', 0)
        h = ohlcv_data.get('high', 0)
        l = ohlcv_data.get('low', 0)
        c = ohlcv_data.get('close', 0)
        v = ohlcv_data.get('volume', 0)
        
        closes = self._get_closes(symbol, timeframe)
        volumes = self._get_volumes(symbol, timeframe)
        
        features = []
        
        # 1. volatility
        if closes is not None and len(closes) > 1:
            returns = np.diff(closes) / closes[:-1]
            volatility = np.std(returns)
        else:
            volatility = 0
        features.append(volatility)
        
        # 2. bb_width
        if closes is not None and len(closes) >= 20:
            sma = np.mean(closes[-20:])
            std = np.std(closes[-20:])
            bb_width = (2 * 2 * std) / sma if sma != 0 else 0
        else:
            bb_width = 0
        features.append(bb_width)
        
        # 3. price_range
        price_range = (h - l) / c if c != 0 else 0
        features.append(price_range)
        
        # 4. body_size
        body_size = abs(c - o) / c if c != 0 else 0
        features.append(body_size)
        
        # 5. rsi
        rsi = self._calculate_rsi(symbol, timeframe)
        features.append(rsi / 100)
        
        # 6. volume_change
        if volumes is not None and len(volumes) >= 5:
            vol_returns = np.diff(volumes[-5:]) / volumes[-6:-1]
            volume_change = np.std(vol_returns)
        else:
            volume_change = 0
        features.append(volume_change)
        
        # 7. atr_ratio
        atr = self._calculate_atr(symbol, timeframe)
        atr_ratio = atr / c if c != 0 else 0
        features.append(atr_ratio)
        
        # 8. returns_rolling_std (10)
        if closes is not None and len(closes) >= 10:
            returns = np.diff(closes[-10:]) / closes[-11:-1]
            returns_rolling_std = np.std(returns)
        else:
            returns_rolling_std = 0
        features.append(returns_rolling_std)
        
        # 9. returns_rolling_mean (10)
        if closes is not None and len(closes) >= 10:
            returns = np.diff(closes[-10:]) / closes[-11:-1]
            returns_rolling_mean = np.mean(returns)
        else:
            returns_rolling_mean = 0
        features.append(returns_rolling_mean)
        
        # 10-12. Historical volatilities
        for period in [5, 10, 20]:
            if closes is not None and len(closes) >= period:
                returns = np.diff(closes[-period:]) / closes[-period-1:-1]
                hist_vol = np.std(returns)
            else:
                hist_vol = 0
            features.append(hist_vol)
        
        # 13. price_to_sma
        if closes is not None and len(closes) >= 20:
            sma_20 = np.mean(closes[-20:])
            price_to_sma = c / sma_20 if sma_20 != 0 else 1
        else:
            price_to_sma = 1
        features.append(price_to_sma)
        
        # 14-15. Stochastic
        k_percent, d_percent = self._calculate_stochastic(symbol, timeframe)
        features.append(k_percent / 100)
        features.append(d_percent / 100)
        
        assert len(features) == 15, f'須有 15 個特徵, 但有 {len(features)} 個'
        
        return np.array(features, dtype=np.float32)

# ============================================================
# 模型管理器
# ============================================================

class ModelManager:
    def __init__(self):
        self.bb_models = {}
        self.validity_models = {}
        self.vol_models = {}
        self.model_cache = {}
        self.bb_feature_extractor = RealTimeFeatureExtractor()
        self.vol_feature_extractor = VolatilityFeatureExtractor()
        self.load_all_models()
    
    def _get_model_path(self, model_type, symbol, timeframe):
        base_path = MODELS_DIR / model_type
        return base_path / symbol / timeframe
    
    def _load_model_file(self, filepath, filename, model_type='auto'):
        full_path = filepath / filename
        cache_key = str(full_path)
        
        if cache_key in self.model_cache:
            return self.model_cache[cache_key]
        
        if full_path.exists():
            try:
                model = ModelLoader.load_model(full_path, model_type)
                if model is not None:
                    self.model_cache[cache_key] = model
                    logger.info(f'已加載: {full_path.name}')
                    return model
            except Exception as e:
                logger.error(f'加載失敐 {full_path}: {str(e)[:100]}')
        
        return None
    
    def load_all_models(self):
        logger.info('開始加載所有模型...')
        
        loaded_count = {'bb': 0, 'validity': 0, 'vol': 0}
        failed_count = {'bb': 0, 'validity': 0, 'vol': 0}
        
        for symbol in SYMBOLS:
            for timeframe in TIMEFRAMES:
                # BB Models
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
                
                # Validity Models
                validity_path = self._get_model_path('validity_models', symbol, timeframe)
                if validity_path.exists():
                    try:
                        validity_model = self._load_model_file(validity_path, 'validity_model.pkl')
                        validity_scaler = self._load_model_file(validity_path, 'scaler.pkl')
                        
                        if validity_model and validity_scaler:
                            self.validity_models[(symbol, timeframe)] = {
                                'model': validity_model,
                                'scaler': validity_scaler
                            }
                            loaded_count['validity'] += 1
                        else:
                            failed_count['validity'] += 1
                    except Exception as e:
                        logger.warning(f'Validity 模型加載失敐: {e}')
                        failed_count['validity'] += 1
                
                # Volatility Models
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
                        logger.warning(f'Vol 模型加載失敐: {e}')
                        failed_count['vol'] += 1
        
        logger.info(f'模型加載完成: BB={loaded_count["bb"]}, Validity={loaded_count["validity"]}, Vol={loaded_count["vol"]}')
    
    def predict_bb_touch(self, symbol, timeframe, ohlcv_data):
        key = (symbol, timeframe)
        if key not in self.bb_models:
            return None
        
        models = self.bb_models[key]
        if not models['model'] or not models['scaler']:
            return None
        
        try:
            features = self.bb_feature_extractor.extract_features(symbol, timeframe, ohlcv_data)
            features_scaled = models['scaler'].transform([features])
            prediction = models['model'].predict(features_scaled)[0]
            confidence = max(models['model'].predict_proba(features_scaled)[0])
            label_map = models['label_map'] or {0: 'lower', 1: 'none', 2: 'upper'}
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
    
    def predict_validity(self, symbol, timeframe, ohlcv_data):
        key = (symbol, timeframe)
        if key not in self.validity_models:
            return None
        
        models = self.validity_models[key]
        if not models['model'] or not models['scaler']:
            return None
        
        try:
            from validity_features import ValidityFeatures
            extractor = ValidityFeatures(lookahead=10)
            df = pd.DataFrame([ohlcv_data])
            df = extractor.extract_all_features(df)
            feature_names = extractor.get_feature_names()
            X = df[feature_names].values
            
            features_scaled = models['scaler'].transform(X)
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
            
            return {
                'valid': valid_prob >= 0.50,
                'probability': valid_prob * 100,
                'quality': quality,
                'confidence': valid_prob
            }
        except Exception as e:
            logger.error(f'有效性預測失敐 {symbol} {timeframe}: {e}')
            return None
    
    def predict_volatility(self, symbol, timeframe, ohlcv_data):
        key = (symbol, timeframe)
        if key not in self.vol_models:
            return None
        
        models = self.vol_models[key]
        if not models['model'] or not models['scaler']:
            return None
        
        try:
            # 提取 15 個波動性特徵
            features = self.vol_feature_extractor.extract_features(symbol, timeframe, ohlcv_data)
            
            # 標準化
            features_scaled = models['scaler'].transform([features])
            
            # 預測
            predicted_vol = float(models['model'].predict(features_scaled)[0])
            will_expand = predicted_vol > 1.2
            expansion_strength = max(0, (predicted_vol - 1.0) / 1.0)
            
            return {
                'predicted_vol': predicted_vol,
                'will_expand': will_expand,
                'expansion_strength': min(1.0, expansion_strength)
            }
        except Exception as e:
            logger.error(f'波動性預測失敐 {symbol} {timeframe}: {str(e)[:200]}')
            return None


model_manager = ModelManager()

# ============================================================
# API 端點
# ============================================================

@app.route('/health', methods=['GET'])
def health_check():
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
    try:
        data = request.get_json()
        symbol = data.get('symbol', '').upper()
        timeframe = data.get('timeframe', '15m')
        ohlcv = data.get('ohlcv', {})
        
        if not symbol or symbol not in SYMBOLS:
            return jsonify({'error': f'無效的幣種: {symbol}'}), 400
        if timeframe not in TIMEFRAMES:
            return jsonify({'error': f'無效的時間框架: {timeframe}'}), 400
        
        # 預測
        bb_result = model_manager.predict_bb_touch(symbol, timeframe, ohlcv)
        if not bb_result or not bb_result['touched']:
            return jsonify({
                'symbol': symbol,
                'timeframe': timeframe,
                'bb_touch': bb_result or {'touched': False},
                'validity': None,
                'volatility': None,
                'signal': 'NEUTRAL'
            })
        
        validity_result = model_manager.predict_validity(symbol, timeframe, ohlcv)
        vol_result = model_manager.predict_volatility(symbol, timeframe, ohlcv)
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
    try:
        data = request.get_json()
        symbols = data.get('symbols', SYMBOLS)
        timeframe = data.get('timeframe', '15m')
        ohlcv_data = data.get('ohlcv_data', {})
        
        results = []
        for symbol in symbols:
            if symbol not in ohlcv_data:
                continue
            
            bb_result = model_manager.predict_bb_touch(symbol, timeframe, ohlcv_data[symbol])
            if not bb_result or not bb_result['touched']:
                continue
            
            validity_result = model_manager.predict_validity(symbol, timeframe, ohlcv_data[symbol])
            vol_result = model_manager.predict_volatility(symbol, timeframe, ohlcv_data[symbol])
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
    if not bb_result or not bb_result['touched']:
        return 'NEUTRAL'
    if not validity_result:
        return 'NEUTRAL'
    
    quality_score = {'excellent': 4, 'good': 3, 'moderate': 2, 'weak': 1, 'poor': 0}.get(validity_result.get('quality', 'poor'), 0)
    vol_score = 2 if (vol_result and vol_result.get('will_expand')) else (1 if vol_result else 0)
    total_score = quality_score + vol_score
    
    if total_score >= 5:
        return 'STRONG_BUY' if bb_result['touch_type'] == 'lower' else 'STRONG_SELL'
    elif total_score >= 3:
        return 'BUY' if bb_result['touch_type'] == 'lower' else 'SELL'
    elif total_score >= 1:
        return 'HOLD'
    return 'NEUTRAL'


def calculate_confidence(bb_result, validity_result, vol_result):
    if not bb_result:
        return 0.0
    confidence = bb_result.get('confidence', 0.5) * 0.3
    if validity_result:
        confidence += validity_result.get('confidence', 0.5) * 0.5
    if vol_result:
        confidence += vol_result.get('expansion_strength', 0.5) * 0.2
    return min(1.0, confidence)


if __name__ == '__main__':
    logger.info('='*60)
    logger.info('BB反彈ML系統 - 實時服務 V3')
    logger.info('='*60)
    logger.info('模型架構：')
    logger.info('  層級1: BB Position Classifier (12 個特徵)')
    logger.info('  層級2: Validity Detector (17 個特徵)')
    logger.info('  層級3: Volatility Predictor (15 個特徵)')
    logger.info('='*60)
    
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=False,
        threaded=True
    )
