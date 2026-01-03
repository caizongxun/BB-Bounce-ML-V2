"""
BB反彈ML系統 - 實時服務 V3
支持三層模型整合：BB觸及檢測 -> 有效性判別 -> 波動性預測
修警：
1. pickle/joblib 序列化不一致問題
2. BB 模型特徵數量不匹配
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
                    logger.debug(f'joblib 加載失救，新試 pickle')
            
            if model_type in ['auto', 'pickle']:
                with open(filepath, 'rb') as f:
                    model = pickle.load(f, encoding='latin1')
                    logger.debug(f'使用 pickle 加載: {filepath.name}')
                    return model
        
        except Exception as e:
            logger.error(f'加載失救 {filepath}: {str(e)[:200]}')
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
        """
        參數：
        - history_size: 保留的歷史 K 線數量 (最低 50)
        """
        self.history_size = max(history_size, 50)
        # {(symbol, timeframe): deque([{open, high, low, close, volume}, ...])}
        self.history = {}
        
        # BB 參數
        self.bb_period = 20
        self.bb_std = 2
        
        # RSI 參數
        self.rsi_period = 14
    
    def update_history(self, symbol, timeframe, ohlcv_data):
        """更新歷式數據快存"""
        key = (symbol, timeframe)
        
        if key not in self.history:
            self.history[key] = deque(maxlen=self.history_size)
        
        self.history[key].append(ohlcv_data)
    
    def _get_close_prices(self, symbol, timeframe):
        """取得角位件的所有收盤價"""
        key = (symbol, timeframe)
        if key not in self.history or len(self.history[key]) == 0:
            return None
        
        return np.array([d.get('close', 0) for d in list(self.history[key])])
    
    def _calculate_bb(self, symbol, timeframe):
        """計算 Bollinger Bands"""
        closes = self._get_close_prices(symbol, timeframe)
        if closes is None or len(closes) < self.bb_period:
            return None, None, None, None  # (upper, middle, lower, width)
        
        sma = np.mean(closes[-self.bb_period:])
        std = np.std(closes[-self.bb_period:])
        upper = sma + self.bb_std * std
        lower = sma - self.bb_std * std
        width = (upper - lower) / sma if sma != 0 else 0
        
        return upper, sma, lower, width
    
    def _calculate_rsi(self, symbol, timeframe):
        """計算 RSI"""
        closes = self._get_close_prices(symbol, timeframe)
        if closes is None or len(closes) < self.rsi_period + 1:
            return 50  # 預設中性 RSI
        
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
        """計算波動性"""
        closes = self._get_close_prices(symbol, timeframe)
        if closes is None or len(closes) < 2:
            return 0
        
        returns = np.diff(closes) / closes[:-1]
        volatility = np.std(returns)
        return volatility
    
    def _calculate_sma(self, symbol, timeframe, period):
        """計算 SMA"""
        closes = self._get_close_prices(symbol, timeframe)
        if closes is None or len(closes) < period:
            return 0
        
        return np.mean(closes[-period:])
    
    def extract_features(self, symbol, timeframe, ohlcv_data):
        """
        提取 12 個特徵 (識抄訓練時使用)
        
        特徵順序 (12 個):
        1. price_to_bb_middle
        2. dist_upper_norm
        3. dist_lower_norm
        4. bb_width
        5. rsi
        6. volatility
        7. returns_std (20 根 SMA 收盤率的標準差)
        8. high_low_ratio
        9. close_open_ratio
        10. sma_5
        11. sma_20
        12. sma_50
        """
        
        # 更新歷式數據
        self.update_history(symbol, timeframe, ohlcv_data)
        
        # 取得當前資料
        o = ohlcv_data.get('open', 0)
        h = ohlcv_data.get('high', 0)
        l = ohlcv_data.get('low', 0)
        c = ohlcv_data.get('close', 0)
        v = ohlcv_data.get('volume', 0)
        
        features = []
        
        # 1-4. BB 特徵
        bb_upper, bb_middle, bb_lower, bb_width = self._calculate_bb(symbol, timeframe)
        
        if bb_middle is not None and bb_middle != 0:
            # 1. price_to_bb_middle
            price_to_bb_middle = (c - bb_middle) / bb_middle
            features.append(price_to_bb_middle)
            
            # 2. dist_upper_norm
            bb_range = bb_upper - bb_lower
            dist_upper_norm = (bb_upper - c) / bb_range if bb_range != 0 else 0
            features.append(dist_upper_norm)
            
            # 3. dist_lower_norm
            dist_lower_norm = (c - bb_lower) / bb_range if bb_range != 0 else 0
            features.append(dist_lower_norm)
            
            # 4. bb_width
            features.append(bb_width)
        else:
            features.extend([0, 0, 0, 0])
        
        # 5. rsi
        rsi = self._calculate_rsi(symbol, timeframe)
        features.append(rsi / 100)  # 正規化到 0-1
        
        # 6. volatility
        volatility = self._calculate_volatility(symbol, timeframe)
        features.append(volatility)
        
        # 7. returns_std (回報率的標準差, 20 根)
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
        sma_5 = self._calculate_sma(symbol, timeframe, 5)
        features.append(sma_5)
        
        sma_20 = self._calculate_sma(symbol, timeframe, 20)
        features.append(sma_20)
        
        sma_50 = self._calculate_sma(symbol, timeframe, 50)
        features.append(sma_50)
        
        # 確保正好 12 個特徵
        assert len(features) == 12, f'須有 12 個特徵, 但有 {len(features)} 個'
        
        return np.array(features, dtype=np.float32)

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
        self.feature_extractor = RealTimeFeatureExtractor()  # 新增
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
    
    def predict_bb_touch(self, symbol, timeframe, ohlcv_data):
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
            # 提取特徵
            features = self.feature_extractor.extract_features(symbol, timeframe, ohlcv_data)
            
            # 標準化
            features_scaled = models['scaler'].transform([features])
            
            # 預測
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
            # 特徵提取 (使用 validity_features.py 的方法)
            from validity_features import ValidityFeatures
            extractor = ValidityFeatures(lookahead=10)
            
            # 需要將 ohlcv_data 轉換为 dataframe
            df = pd.DataFrame([ohlcv_data])
            df = extractor.extract_all_features(df)
            
            feature_names = models['feature_names'] or extractor.get_feature_names()
            X = df[feature_names].values
            
            # 標準化
            features_scaled = models['scaler'].transform(X)
            
            # 預測機率
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
    
    def predict_volatility(self, symbol, timeframe, ohlcv_data):
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
            # 需要按照 train_vol_model.py 的標準提取特徵
            # 戲下為簡化識別
            from collections import Counter
            
            # 特徵提取 (簡化簡)
            o = ohlcv_data.get('open', 0)
            h = ohlcv_data.get('high', 0)
            l = ohlcv_data.get('low', 0)
            c = ohlcv_data.get('close', 0)
            v = ohlcv_data.get('volume', 0)
            
            # 粀堵特徵 (需改進)
            features = [h/l if l != 0 else 1, c/o if o != 0 else 1, v/1e6]
            
            # 標準化測試
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
        
        # 預測 BB 觸碰
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
        
        # 預測有效性
        validity_result = model_manager.predict_validity(symbol, timeframe, ohlcv)
        
        # 預測波動性
        vol_result = model_manager.predict_volatility(symbol, timeframe, ohlcv)
        
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
    logger.info('修復:')
    logger.info('  1. joblib + pickle 序列化錯誤')
    logger.info('  2. BB 特徵數量不匹配 (16 → 12) - 実現歷式數據快存')
    logger.info('='*60)
    
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=False,
        threaded=True
    )
