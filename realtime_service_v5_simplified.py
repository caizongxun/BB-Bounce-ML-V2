#!/usr/bin/env python3
"""
BB反彈ML系統 - 實時服務 V5 (簡化版)
直接計算 BB 通道 - 不再依賴內存 history
使用 Binance 完整歷史數據來計算 BB

改進：使用上一根完全形成的K棒做預測，避免預測閃爍
完整特椅提取：提取完整的17個特椅，與訓練模型一致
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
TOUCHED_THRESHOLD = 0.0005      # 0.05% - 接觸
APPROACHING_DANGER = 0.002      # 0.2%  - 接近危險
APPROACHING_WARNING = 0.005     # 0.5%  - 接近警告
APPROACHING_CAUTION = 0.015     # 1.5%  - 接近注意

# Binance API
BINANCE_API = 'https://api.binance.com/api/v3'

MODELS_DIR = Path('./models')

# ============================================================
# 預測策略配置
# ============================================================

class PredictionStrategy:
    """
    預測策略選擇器
    LATEST_INCOMPLETE: 使用最新K棒（即使未閉合）- 最敏感但有閃爍
    PREVIOUS_COMPLETE: 使用上一根完全K棒 - 穩定但延遲一根
    SMOOTHED: 混合策略 - 平衡敏感度和穩定性
    """
    LATEST_INCOMPLETE = 'latest'      # 實時敏感
    PREVIOUS_COMPLETE = 'previous'    # 穩定可靠
    SMOOTHED = 'smoothed'             # 平衡

# 當前使用的策略（推薦用 PREVIOUS_COMPLETE）
CURRENT_STRATEGY = PredictionStrategy.PREVIOUS_COMPLETE

# ============================================================
# BB 計算器 - 使用 Binance 完整歷史數據
# ============================================================

class BBCalculator:
    """使用 Binance 完整歷史數據計算 BB 通道"""
    
    @staticmethod
    def fetch_historical_closes(symbol, timeframe, limit=100):
        """從 Binance 獲取完整歷史收盤價"""
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
                
                # 同時近回完整K棒信息，用於K棒閉合檢測
                klines_data = [{
                    'open': float(k[1]),
                    'high': float(k[2]),
                    'low': float(k[3]),
                    'close': float(k[4]),
                    'volume': float(k[7]),
                    'close_time': int(k[6])  # K棒閉合時間
                } for k in klines]
                
                return closes, klines_data
        except Exception as e:
            logger.warning(f'[Binance] 獲取歷史數據失敗: {symbol} {timeframe} - {e}')
        return None, None
    
    @staticmethod
    def is_kline_closed(kline_close_time, current_time):
        """檢測K棒是否已閉合"""
        return current_time >= kline_close_time
    
    @staticmethod
    def calculate_bb(symbol, timeframe, strategy=PredictionStrategy.PREVIOUS_COMPLETE):
        """計算 BB 通道值"""
        closes, klines_data = BBCalculator.fetch_historical_closes(symbol, timeframe, limit=100)
        if closes is None or len(closes) < BB_PERIOD:
            actual_length = len(closes) if closes is not None else 0
            logger.warning(f'[BB計算] {symbol} {timeframe}: 數據不足 (只有 {actual_length} 根，需要 {BB_PERIOD} 根)')
            return None, None, None, None, None
        
        # 根據策略選擇用哪根K棍
        if strategy == PredictionStrategy.PREVIOUS_COMPLETE:
            prediction_kline = klines_data[-2] if len(klines_data) >= 2 else klines_data[-1]
            kline_status = 'CLOSED'
            logger.info(f'[K棒選擇] {symbol} {timeframe}: 使用上一根完全K棍 (已閉合)')
        else:
            prediction_kline = klines_data[-1]
            current_time_ms = datetime.now().timestamp() * 1000
            is_closed = BBCalculator.is_kline_closed(prediction_kline['close_time'], current_time_ms)
            kline_status = 'CLOSED' if is_closed else 'FORMING'
            logger.info(f'[K棒選擇] {symbol} {timeframe}: 使用最新K棍 (狀態={kline_status})')
        
        recent_closes = closes[-BB_PERIOD:]
        sma = np.mean(recent_closes)
        std = np.std(recent_closes)
        upper = sma + BB_STD * std
        lower = sma - BB_STD * std
        
        logger.info(f'[BB計算] {symbol} {timeframe}: 上={upper:.2f}, 中={sma:.2f}, 下={lower:.2f}')
        logger.info(f'[預測K棍] {symbol} {timeframe}: close={prediction_kline["close"]:.2f}, high={prediction_kline["high"]:.2f}, low={prediction_kline["low"]:.2f}')
        
        return float(upper), float(sma), float(lower), prediction_kline, kline_status
    
    @staticmethod
    def analyze_bb_status(symbol, timeframe, strategy=PredictionStrategy.PREVIOUS_COMPLETE):
        """
        分析 K 棍是否接近/接觸 BB 軌道
        """
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
        
        logger.info(f'[距離] {symbol} {timeframe}: 上={dist_to_upper*100:.8f}%, 下={dist_to_lower*100:.8f}%')
        
        # 判斷狀態
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
# 完整特椅提取器 - 17個特椅
# ============================================================

class CompleteFeatureExtractor:
    """提取完整的17個特椅符合訓練模型"""
    
    @staticmethod
    def calculate_indicators(closes, highs, lows, volumes):
        """計算基本指標"""
        # BB
        sma = np.mean(closes[-BB_PERIOD:])
        std = np.std(closes[-BB_PERIOD:])
        bb_upper = sma + BB_STD * std
        bb_lower = sma - BB_STD * std
        bb_middle = sma
        bb_width = bb_upper - bb_lower
        
        # RSI
        deltas = np.diff(closes[-14:])
        gains = np.where(deltas > 0, deltas, 0).mean()
        losses = np.where(deltas < 0, -deltas, 0).mean()
        rs = gains / (losses + 1e-8)
        rsi = 100 - (100 / (1 + rs))
        
        # ATR
        tr1 = highs[-14:] - lows[-14:]
        tr2 = np.abs(highs[-14:] - closes[-14:][:-1] if len(closes) > 1 else 0)
        tr3 = np.abs(lows[-14:] - closes[-14:][:-1] if len(closes) > 1 else 0)
        atr = np.max([tr1, tr2, tr3], axis=0).mean()
        
        # EMA
        ema = closes[-5:].mean()  # 粗似 EMA
        ema_prev = closes[-6:-5].mean() if len(closes) >= 6 else closes[-5]
        ema_slope = (ema - ema_prev) / (ema_prev + 1e-8) if ema_prev > 0 else 0
        
        # 成交量比
        avg_volume = np.mean(volumes[-20:])
        volume_ratio = volumes[-1] / (avg_volume + 1e-8)
        
        return {
            'sma': sma,
            'std': std,
            'bb_upper': bb_upper,
            'bb_lower': bb_lower,
            'bb_middle': bb_middle,
            'bb_width': bb_width,
            'rsi': rsi,
            'atr': atr,
            'ema_slope': ema_slope,
            'volume_ratio': volume_ratio,
            'avg_volume': avg_volume,
        }
    
    @staticmethod
    def extract_17_features(symbol, timeframe, prediction_kline):
        """提取完整的17個特椅"""
        try:
            # 獲取 100 根K棍數據
            closes, klines_data = BBCalculator.fetch_historical_closes(symbol, timeframe, limit=100)
            if closes is None or len(closes) < 50:
                logger.warning(f'[7特椅] {symbol}: 數據不足')
                return None
            
            # 提取 OHLCV
            highs = np.array([k['high'] for k in klines_data])
            lows = np.array([k['low'] for k in klines_data])
            volumes = np.array([k['volume'] for k in klines_data])
            
            # 計算基本指標
            indicators = CompleteFeatureExtractor.calculate_indicators(closes, highs, lows, volumes)
            
            # 當前價格
            curr_close = prediction_kline['close']
            curr_high = prediction_kline['high']
            curr_low = prediction_kline['low']
            curr_volume = prediction_kline['volume']
            
            # 提取 17 個特椅
            features = []
            
            # 動量特椅 (3個)
            momentum = np.mean(np.diff(closes[-5:]))
            momentum_prev = np.mean(np.diff(closes[-10:-5]))
            momentum_decay = (momentum_prev - momentum) / (abs(momentum_prev) + 1e-8)
            features.append(momentum_decay)  # 0: momentum_decay_rate
            
            # 比例特椅 (3個)
            bounce_height = (curr_high - curr_close) / (indicators['bb_width'] + 1e-8)
            features.append(bounce_height)  # 1: bounce_height_ratio
            
            time_to_recovery = 0  # 简銀
            features.append(time_to_recovery)  # 2: time_to_recovery
            
            breakout_dist = (curr_high / (indicators['bb_upper'] + 1e-8) - 1) * 100
            features.append(breakout_dist)  # 3: breakout_distance
            
            # RSI相關 (2個)
            features.append(indicators['rsi'])  # 4: rsi_level
            features.append(indicators['volume_ratio'])  # 5: volume_ratio
            
            # 波動率相關 (3個)
            volatility = np.std(np.diff(closes[-20:]))
            avg_volatility = np.std(np.diff(closes[-40:-20]))
            vol_regime = volatility / (avg_volatility + 1e-8)
            features.append(vol_regime)  # 6: volatility_regime
            
            bb_width_ratio = indicators['bb_width'] / np.mean([klines_data[i]['high'] - klines_data[i]['low'] for i in range(-20, 0)]) + 1e-8
            features.append(bb_width_ratio)  # 7: bb_width_ratio
            
            momentum_dir = np.sign(np.mean(np.diff(closes[-5:])))
            features.append(momentum_dir)  # 8: price_momentum_direction
            
            # 优選特椅 (3個)
            price_to_middle = (curr_close - indicators['bb_middle']) / (indicators['bb_middle'] + 1e-8)
            features.append(price_to_middle)  # 9: price_to_bb_middle
            
            dist_lower = (curr_close - indicators['bb_lower']) / (indicators['bb_width'] + 1e-8)
            features.append(dist_lower)  # 10: dist_lower_norm
            
            dist_upper = (indicators['bb_upper'] - curr_close) / (indicators['bb_width'] + 1e-8)
            features.append(dist_upper)  # 11: dist_upper_norm
            
            # 基税指標 (3個)
            features.append(indicators['rsi'])  # 12: rsi
            features.append(indicators['atr'])  # 13: atr
            features.append(indicators['ema_slope'])  # 14: ema_slope
            
            # 反配特椅 (3個)
            reversal_strength = abs(momentum) / (abs(momentum_prev) + 1e-8) if momentum_prev != 0 else 0
            features.append(reversal_strength)  # 15: momentum_reversal_strength
            
            vol_momentum = curr_volume / (indicators['avg_volume'] + 1e-8)
            features.append(vol_momentum)  # 16: volume_momentum_ratio
            
            logger.info(f'[17特椅] {symbol} {timeframe}: 已提取 17 個特椅')
            return np.array(features, dtype=np.float32)
        
        except Exception as e:
            logger.error(f'[17特椅] 提取失敖: {e}')
            return None

# ============================================================
# 有效性 & 波動性模型
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

class ValidityChecker:
    """有效性檢查 - 使用完整的17個特椅"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.load_all_models()
    
    def load_all_models(self):
        for symbol in SYMBOLS:
            for timeframe in TIMEFRAMES:
                model_path = MODELS_DIR / 'validity_models' / symbol / timeframe / 'validity_model.pkl'
                scaler_path = MODELS_DIR / 'validity_models' / symbol / timeframe / 'scaler.pkl'
                
                model = ModelLoader.load_model(model_path)
                scaler = ModelLoader.load_model(scaler_path)
                
                if model and scaler:
                    self.models[(symbol, timeframe)] = model
                    self.scalers[(symbol, timeframe)] = scaler
                    logger.debug(f'已加載: {symbol} {timeframe}')
    
    def predict(self, symbol, timeframe, features):
        """預測有效性"""
        key = (symbol, timeframe)
        if key not in self.models:
            return None
        
        try:
            model = self.models[key]
            scaler = self.scalers[key]
            
            # 標準化特椅
            features_scaled = scaler.transform([features])
            proba = model.predict_proba(features_scaled)[0]
            valid_prob = float(proba[1]) if len(proba) > 1 else 0.5
            
            if valid_prob >= 0.75:
                quality = 'excellent'
            elif valid_prob >= 0.65:
                quality = 'good'
            elif valid_prob >= 0.50:
                quality = 'moderate'
            else:
                quality = 'weak'
            
            logger.info(f'[有效性] {symbol} {timeframe}: {valid_prob*100:.1f}% ({quality})')
            
            return {
                'valid': valid_prob >= 0.50,
                'probability': valid_prob * 100,
                'quality': quality
            }
        except Exception as e:
            logger.debug(f'有效性預測: {e}')
            return None

class VolatilityPredictor:
    """波動性預測 - 使用完整的17個特椅"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.load_all_models()
    
    def load_all_models(self):
        for symbol in SYMBOLS:
            for timeframe in TIMEFRAMES:
                model_path = MODELS_DIR / 'vol_models' / symbol / timeframe / 'model_regression.pkl'
                scaler_path = MODELS_DIR / 'vol_models' / symbol / timeframe / 'scaler_regression.pkl'
                
                model = ModelLoader.load_model(model_path)
                scaler = ModelLoader.load_model(scaler_path)
                
                if model and scaler:
                    self.models[(symbol, timeframe)] = model
                    self.scalers[(symbol, timeframe)] = scaler
                    logger.debug(f'已加載: {symbol} {timeframe}')
    
    def predict(self, symbol, timeframe, features):
        """預測波動性"""
        key = (symbol, timeframe)
        if key not in self.models:
            return None
        
        try:
            model = self.models[key]
            scaler = self.scalers[key]
            
            # 標準化特椅（提取前15個特椅）
            features_vol = features[:15]
            features_scaled = scaler.transform([features_vol])
            predicted_vol = float(model.predict(features_scaled)[0])
            
            will_expand = predicted_vol > 1.2
            expansion_strength = max(0, (predicted_vol - 1.0) / 1.0)
            
            if expansion_strength > 1.5:
                vol_level = 'very_high'
            elif expansion_strength > 1.0:
                vol_level = 'high'
            elif expansion_strength > 0.5:
                vol_level = 'moderate'
            else:
                vol_level = 'low'
            
            logger.info(f'[波動性] {symbol} {timeframe}: {predicted_vol:.2f}x ({vol_level})')
            
            return {
                'predicted_vol': predicted_vol,
                'will_expand': will_expand,
                'expansion_strength': min(1.0, expansion_strength),
                'volatility_level': vol_level
            }
        except Exception as e:
            logger.debug(f'波動性預測: {e}')
            return None

# ============================================================
# 初始化
# ============================================================

validity_checker = ValidityChecker()
volatility_predictor = VolatilityPredictor()

# ============================================================
# API 端點
# ============================================================

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'ok',
        'timestamp': datetime.now().isoformat(),
        'description': 'BB 反彈實時監控系統 V5 (簡化版)',
        'prediction_strategy': CURRENT_STRATEGY,
        'features': '17個完整特椅'
    })


@app.route('/predict', methods=['POST'])
def predict():
    """預測 K 棍是否接近/接觸 BB 軌道"""
    try:
        data = request.get_json()
        symbol = data.get('symbol', '').upper()
        timeframe = data.get('timeframe', '15m')
        
        if not symbol or symbol not in SYMBOLS:
            return jsonify({'error': f'無效的幣種: {symbol}'}), 400
        if timeframe not in TIMEFRAMES:
            return jsonify({'error': f'無效的時間框架: {timeframe}'}), 400
        
        logger.info(f'\n[請求] {symbol} {timeframe}')
        
        # 第一步: 計算 BB
        bb_result = BBCalculator.analyze_bb_status(symbol, timeframe, strategy=CURRENT_STRATEGY)
        
        # 第二步: 只有接近/接觸時才調用模型
        validity_result = None
        volatility_result = None
        
        if bb_result['status'] in ['approaching', 'touched']:
            logger.info(f'[模型] 觸發模型預測 (狀態={bb_result["status"]})')
            
            prediction_kline = bb_result['prediction_kline']
            if prediction_kline:
                # 提取 17 個特椅
                features = CompleteFeatureExtractor.extract_17_features(symbol, timeframe, prediction_kline)
                
                if features is not None:
                    logger.info(f'[特椅] {symbol}: {features[:5]}')
                    validity_result = validity_checker.predict(symbol, timeframe, features)
                    volatility_result = volatility_predictor.predict(symbol, timeframe, features)
                else:
                    logger.warning(f'[警告] 特椅提取失敗')
        
        logger.info(f'[回應] {symbol} {timeframe} - 狀態={bb_result["status"]}, 距離={bb_result["distance_percent"]:.8f}%\n')
        
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
                'kline_status': bb_result['kline_status'],
                'prediction_source': bb_result['prediction_source'],
            },
            'validity': validity_result,
            'volatility': volatility_result
        })
    
    except Exception as e:
        logger.error(f'預測錯誤: {e}', exc_info=True)
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    try:
        logger.info('=' * 60)
        logger.info('BB 反彈實時監控系統 V5 (簡化版)')
        logger.info('=' * 60)
        logger.info('特椅: 17個完整特椅提取')
        logger.info('BB計算: 使用上一根完整K棍')
        logger.info('有效性/波動性: 只有接近時計算')
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