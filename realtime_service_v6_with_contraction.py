#!/usr/bin/env python3
"""
BB 反彈實時監控系統 V6
整合了特徵：
1. V1 有效性模型 (提取 17 個特徵)
2. V2 BB 收縮模型 (提取 16 個特徵)
當两个模型都符合時，策略需要待兒反彈
"""

import numpy as np
import pandas as pd
from pathlib import Path
from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime
import logging
import warnings
import joblib
import pickle
import requests
from functools import lru_cache
import traceback

warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# ============================================================
# 全局配置
# ============================================================

SYMBOLS = [
    'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'XRPUSDT', 'DOGEUSDT',
    'SOLUSDT', 'LTCUSDT', 'AVAXUSDT', 'LINKUSDT', 'UNIUSDT', 'ATOMUSDT',
    'MATICUSDT', 'DOTUSDT', 'FILUSDT', 'AAVEUSDT', 'OPUSDT', 'ARBUSDT',
    'ALGOUSDT', 'NEARUSDT', 'BCHUSDT', 'ETCUSDT'
]

TIMEFRAMES = ['15m', '1h']

# BB 參數
BB_PERIOD = 20
BB_STD = 2

# 接近/接觸閾值
TOUCHED_THRESHOLD = 0.0005      # 0.05%
APPROACHING_DANGER = 0.002      # 0.2%
APPROACHING_WARNING = 0.005     # 0.5%
APPROACHING_CAUTION = 0.015     # 1.5%

# Binance API
BINANCE_API = 'https://api.binance.com/api/v3'

MODELS_DIR = Path('./models')

# ============================================================
# 預測策略
# ============================================================

class PredictionStrategy:
    LATEST_INCOMPLETE = 'latest'
    PREVIOUS_COMPLETE = 'previous'

CURRENT_STRATEGY = PredictionStrategy.PREVIOUS_COMPLETE

# ============================================================
# BB 計算器
# ============================================================

class BBCalculator:
    
    @staticmethod
    def fetch_historical_closes(symbol, timeframe, limit=100):
        try:
            url = f"{BINANCE_API}/klines"
            params = {
                'symbol': symbol,
                'interval': timeframe,
                'limit': limit
            }
            response = requests.get(url, params=params, timeout=5)
            if response.status_code == 200:
                klines = response.json()
                closes = np.array([float(k[4]) for k in klines])
                
                klines_data = [{
                    'open': float(k[1]),
                    'high': float(k[2]),
                    'low': float(k[3]),
                    'close': float(k[4]),
                    'volume': float(k[7]),
                    'close_time': int(k[6])
                } for k in klines]
                
                return closes, klines_data
        except Exception as e:
            logger.warning(f'[Binance] 獲取歷史數據失敖: {symbol} {timeframe} - {e}')
        return None, None
    
    @staticmethod
    def calculate_bb(symbol, timeframe, strategy=PredictionStrategy.PREVIOUS_COMPLETE):
        closes, klines_data = BBCalculator.fetch_historical_closes(symbol, timeframe, limit=100)
        if closes is None or len(closes) < BB_PERIOD:
            logger.warning(f'[BB計算] {symbol} {timeframe}: 數據不足')
            return None, None, None, None, None
        
        if strategy == PredictionStrategy.PREVIOUS_COMPLETE:
            prediction_kline = klines_data[-2] if len(klines_data) >= 2 else klines_data[-1]
            kline_status = 'CLOSED'
        else:
            prediction_kline = klines_data[-1]
            kline_status = 'FORMING'
        
        recent_closes = closes[-BB_PERIOD:]
        sma = np.mean(recent_closes)
        std = np.std(recent_closes)
        upper = sma + BB_STD * std
        lower = sma - BB_STD * std
        
        return float(upper), float(sma), float(lower), prediction_kline, kline_status
    
    @staticmethod
    def analyze_bb_status(symbol, timeframe, strategy=PredictionStrategy.PREVIOUS_COMPLETE):
        bb_upper, bb_middle, bb_lower, prediction_kline, kline_status = BBCalculator.calculate_bb(
            symbol, timeframe, strategy=strategy
        )
        
        if bb_upper is None:
            return {
                'status': 'normal',
                'direction': None,
                'distance_percent': 0,
                'warning_level': 'none',
                'bb_upper': None,
                'bb_middle': None,
                'bb_lower': None,
                'kline_status': 'ERROR',
                'prediction_source': 'none',
                'prediction_kline': None,
            }
        
        pred_close = prediction_kline.get('close', 0)
        pred_high = prediction_kline.get('high', 0)
        pred_low = prediction_kline.get('low', 0)
        
        if pred_close <= 0:
            return {
                'status': 'normal',
                'direction': None,
                'distance_percent': 0,
                'warning_level': 'none',
                'bb_upper': bb_upper,
                'bb_middle': bb_middle,
                'bb_lower': bb_lower,
                'kline_status': kline_status,
                'prediction_source': 'failed',
                'prediction_kline': None,
            }
        
        dist_to_upper = (bb_upper - pred_high) / bb_upper if bb_upper > 0 else 1.0
        dist_to_lower = (pred_low - bb_lower) / bb_lower if bb_lower > 0 else 1.0
        
        status = 'normal'
        direction = None
        distance_percent = 0
        warning_level = 'none'
        
        if dist_to_upper <= TOUCHED_THRESHOLD:
            status = 'touched'
            direction = 'upper'
            distance_percent = dist_to_upper * 100
            warning_level = 'danger'
        elif dist_to_lower <= TOUCHED_THRESHOLD:
            status = 'touched'
            direction = 'lower'
            distance_percent = dist_to_lower * 100
            warning_level = 'danger'
        elif dist_to_upper <= APPROACHING_DANGER:
            status = 'approaching'
            direction = 'upper'
            distance_percent = dist_to_upper * 100
            warning_level = 'danger'
        elif dist_to_lower <= APPROACHING_DANGER:
            status = 'approaching'
            direction = 'lower'
            distance_percent = dist_to_lower * 100
            warning_level = 'danger'
        elif dist_to_upper <= APPROACHING_WARNING:
            status = 'approaching'
            direction = 'upper'
            distance_percent = dist_to_upper * 100
            warning_level = 'warning'
        elif dist_to_lower <= APPROACHING_WARNING:
            status = 'approaching'
            direction = 'lower'
            distance_percent = dist_to_lower * 100
            warning_level = 'warning'
        elif dist_to_upper <= APPROACHING_CAUTION:
            status = 'approaching'
            direction = 'upper'
            distance_percent = dist_to_upper * 100
            warning_level = 'caution'
        elif dist_to_lower <= APPROACHING_CAUTION:
            status = 'approaching'
            direction = 'lower'
            distance_percent = dist_to_lower * 100
            warning_level = 'caution'
        
        return {
            'status': status,
            'direction': direction,
            'distance_percent': distance_percent,
            'warning_level': warning_level,
            'bb_upper': bb_upper,
            'bb_middle': bb_middle,
            'bb_lower': bb_lower,
            'kline_status': kline_status,
            'prediction_source': 'previous_complete' if strategy == PredictionStrategy.PREVIOUS_COMPLETE else strategy,
            'prediction_kline': prediction_kline,
        }

# ============================================================
# BB 收縮特徵提取器
# ============================================================

class BBContractionFeatureExtractor:
    
    @staticmethod
    def calculate_bb_bands(closes, period=20, std_dev=2):
        sma = np.mean(closes[-period:])
        std = np.std(closes[-period:])
        upper = sma + std_dev * std
        lower = sma - std_dev * std
        width = upper - lower
        return upper, sma, lower, width, std
    
    @staticmethod
    def extract_contraction_features(symbol, timeframe, prediction_kline):
        """
        提取 BB 收縮特徵
        """
        try:
            closes, klines_data = BBCalculator.fetch_historical_closes(symbol, timeframe, limit=100)
            
            if closes is None or len(closes) < 40:
                return None
            
            # 計算當前帶狀
            curr_upper, curr_middle, curr_lower, curr_width, curr_std = \
                BBContractionFeatureExtractor.calculate_bb_bands(closes[-20:])
            
            # 計算過去 3 根 K 棒的帶狀
            past_upper, past_middle, past_lower, past_width, past_std = \
                BBContractionFeatureExtractor.calculate_bb_bands(closes[-23:-3])
            
            # 特徵 1：BB 寬度 1 根 K 棒變化
            if len(closes) >= 21:
                prev_width = BBContractionFeatureExtractor.calculate_bb_bands(closes[-21:-1])[3]
                bb_width_change_1bar = (curr_width - prev_width) / (prev_width + 1e-8)
            else:
                bb_width_change_1bar = 0
            
            # 特徵 2：BB 寬度 2 根 K 棒變化
            bb_width_change_2bar = (curr_width - past_width) / (past_width + 1e-8)
            
            # 特徵 3：標準差變化
            std_change = (curr_std - past_std) / (past_std + 1e-8)
            
            # 特徵 4：BB 寬度相對歷史位置
            widths = [BBContractionFeatureExtractor.calculate_bb_bands(closes[i-20:i])[3] for i in range(20, len(closes))]
            widths_array = np.array(widths)
            bb_width_percentile = (curr_width - widths_array.min()) / (widths_array.max() - widths_array.min() + 1e-8)
            
            # 特徵 5：價格相對 BB 位置
            price_bb_position = (prediction_kline['close'] - curr_lower) / (curr_width + 1e-8)
            
            # 特徵 6：RSI
            delta = np.diff(closes[-15:])
            gain = np.where(delta > 0, delta, 0).mean()
            loss = np.where(delta < 0, -delta, 0).mean()
            rs = gain / (loss + 1e-8)
            rsi_14 = 100 - (100 / (1 + rs))
            
            # 特徵 7-8：波動率比
            hist_vol = np.std(np.diff(np.log(closes[-20:])))
            vol_ratio = curr_std / (np.std(closes[-40:-20]) + 1e-8)
            
            # 特徵 9-10：動量
            momentum_5 = (prediction_kline['close'] - closes[-5]) / closes[-5]
            momentum_10 = (prediction_kline['close'] - closes[-10]) / closes[-10]
            
            # 特徵 11：BB 寬度加速度
            if len(closes) >= 23:
                prev2_width = BBContractionFeatureExtractor.calculate_bb_bands(closes[-23:-2])[3]
                bb_width_acceleration = (bb_width_change_1bar - (prev_width - prev2_width) / (prev2_width + 1e-8))
            else:
                bb_width_acceleration = 0
            
            # 特徵 12：成交量比
            if 'volume' in str(klines_data[0]):
                volume_ratio = prediction_kline.get('volume', 1) / (np.mean([k['volume'] for k in klines_data[-20:]]) + 1e-8)
            else:
                volume_ratio = 1.0
            
            # 特徵 13-15：掩展相關
            bb_width_change_3bar = (curr_width - BBContractionFeatureExtractor.calculate_bb_bands(closes[-23:-3])[3]) / \
                                   (BBContractionFeatureExtractor.calculate_bb_bands(closes[-23:-3])[3] + 1e-8)
            bb_distance_change = 0  # 保留來解確保篆康
            historical_vol = hist_vol
            
            features = np.array([
                bb_width_change_1bar,
                bb_width_change_2bar,
                bb_width_change_3bar,
                bb_width_percentile,
                std_change,
                price_bb_position,
                rsi_14,
                momentum_5,
                momentum_10,
                bb_width_acceleration,
                volume_ratio,
                vol_ratio,
                historical_vol,
                bb_distance_change,
                curr_width,  # 原始寬度
                curr_std     # 原始標準差
            ], dtype=np.float32)
            
            return features
        
        except Exception as e:
            logger.error(f'[BB收縮特徵] {symbol} {timeframe} 失敖: {e}')
            return None

# ============================================================
# 模型加載
# ============================================================

class ModelLoader:
    @staticmethod
    def load_model(filepath):
        filepath = Path(filepath)
        if not filepath.exists():
            return None
        try:
            return joblib.load(filepath)
        except:
            try:
                with open(filepath, 'rb') as f:
                    return pickle.load(f, encoding='latin1')
            except:
                return None

class BBContractionPredictor:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.load_all_models()
    
    def load_all_models(self):
        for symbol in SYMBOLS:
            for timeframe in TIMEFRAMES:
                model_path = MODELS_DIR / 'bb_contraction_v2_models' / symbol / timeframe / 'bb_contraction_v2_model.pkl'
                scaler_path = MODELS_DIR / 'bb_contraction_v2_models' / symbol / timeframe / 'bb_contraction_v2_scaler.pkl'
                
                model = ModelLoader.load_model(model_path)
                scaler = ModelLoader.load_model(scaler_path)
                
                if model and scaler:
                    self.models[(symbol, timeframe)] = model
                    self.scalers[(symbol, timeframe)] = scaler
    
    def predict(self, symbol, timeframe, features):
        key = (symbol, timeframe)
        if key not in self.models:
            return None
        
        try:
            model = self.models[key]
            scaler = self.scalers[key]
            
            features_scaled = scaler.transform([features])
            proba = model.predict_proba(features_scaled)[0]
            bounce_valid_prob = float(proba[1]) if len(proba) > 1 else 0.5
            
            logger.debug(f'[BB收縮] {symbol} {timeframe}: {bounce_valid_prob*100:.1f}%')
            
            return {
                'valid': bounce_valid_prob >= 0.50,
                'probability': bounce_valid_prob * 100,
                'quality': self._get_quality(bounce_valid_prob)
            }
        except Exception as e:
            logger.error(f'[BB收縮預測失敖] {e}')
            return None
    
    @staticmethod
    def _get_quality(prob):
        if prob >= 0.80:
            return 'excellent'
        elif prob >= 0.65:
            return 'good'
        elif prob >= 0.50:
            return 'moderate'
        else:
            return 'weak'

# ============================================================
# 初始化
# ============================================================

bb_contraction_predictor = BBContractionPredictor()

# ============================================================
# API 端點
# ============================================================

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'ok',
        'timestamp': datetime.now().isoformat(),
        'description': 'BB 反彈實時監控系統 V6 - 整合版',
        'models': {
            'validity': '有效性模型 (V1)',
            'contraction': 'BB收縮模型 (V2)'
        }
    })


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        symbol = data.get('symbol', '').upper()
        timeframe = data.get('timeframe', '15m')
        
        if not symbol or symbol not in SYMBOLS:
            return jsonify({'error': f'無效的幣種: {symbol}'}), 400
        if timeframe not in TIMEFRAMES:
            return jsonify({'error': f'無效的時間框架: {timeframe}'}), 400
        
        logger.info(f'[請求] {symbol} {timeframe}')
        
        # BB 計算
        bb_result = BBCalculator.analyze_bb_status(symbol, timeframe, strategy=CURRENT_STRATEGY)
        
        # BB 收縮模型 V2
        bb_contraction_result = None
        
        if bb_result['warning_level'] in ['danger', 'warning', 'caution']:
            logger.info(f'[模型] 觸發 V2 模型 (警告={bb_result["warning_level"]})')
            
            prediction_kline = bb_result['prediction_kline']
            if prediction_kline:
                features = BBContractionFeatureExtractor.extract_contraction_features(symbol, timeframe, prediction_kline)
                
                if features is not None:
                    bb_contraction_result = bb_contraction_predictor.predict(symbol, timeframe, features)
        
        return jsonify({
            'symbol': symbol,
            'timeframe': timeframe,
            'timestamp': datetime.now().isoformat(),
            'bb_touch': {
                'status': bb_result['status'],
                'direction': bb_result['direction'],
                'distance_percent': bb_result['distance_percent'],
                'warning_level': bb_result['warning_level'],
                'bb_upper': bb_result['bb_upper'],
                'bb_middle': bb_result['bb_middle'],
                'bb_lower': bb_result['bb_lower'],
            },
            'bb_contraction_v2': bb_contraction_result  # 新增 V2 結果
        })
    
    except Exception as e:
        logger.error(f'預測錯誤: {e}', exc_info=True)
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    try:
        logger.info('=' * 60)
        logger.info('BB 反彈實時監控系統 V6 - 整合版')
        logger.info('=' * 60)
        logger.info('模型 1：V1 有效性 (有效性評分)')
        logger.info('模型 2：V2 BB收縮 (反彈所有效性)')
        logger.info('=' * 60)
        
        logger.info(f'部署地址: 0.0.0.0:5000')
        app.run(
            host='0.0.0.0',
            port=5000,
            debug=False,
            threaded=True,
            use_reloader=False
        )
    except Exception as e:
        logger.error(f'[FATAL] {e}', exc_info=True)
        raise