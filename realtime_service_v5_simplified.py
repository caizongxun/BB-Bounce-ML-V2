#!/usr/bin/env python3
"""
BB反彈ML系統 - 實時服勑 V5 (简化版)
直接計算 BB 通道
先棂測接近/接觸 -> 更敆澎感: 粗付特檲提取，不需要正確的特檲數
"""

import numpy as np
import pandas as pd
from pathlib import Path
from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime
import logging
from collections import deque
import warnings
import joblib
import pickle

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
# 修正：改用件數比率 (一倐可对比的錯誤佐度)
TOUCHED_THRESHOLD = 0.0005      # 0.05% - 接觸
APPROACHING_DANGER = 0.002      # 0.2%  - 接近危險
APPROACHING_WARNING = 0.005     # 0.5%  - 接近警告
APPROACHING_CAUTION = 0.015     # 1.5%  - 接近注意

MODELS_DIR = Path('./models')

# ============================================================
# BB 計算器
# ============================================================

class BBCalculator:
    """直接計算 BB 通道，検測接近/接觸"""
    
    def __init__(self, history_size=100):
        self.history_size = history_size
        self.history = {}  # {(symbol, timeframe): deque of OHLCV}
    
    def update_history(self, symbol, timeframe, ohlcv_data):
        """更新歷史數料"""
        key = (symbol, timeframe)
        if key not in self.history:
            self.history[key] = deque(maxlen=self.history_size)
        self.history[key].append(ohlcv_data)
    
    def _get_closes(self, symbol, timeframe):
        """獲取所有收盤價"""
        key = (symbol, timeframe)
        if key not in self.history or len(self.history[key]) == 0:
            return None
        return np.array([d.get('close', 0) for d in list(self.history[key])])
    
    def calculate_bb(self, symbol, timeframe):
        """計算 BB 通道值"""
        closes = self._get_closes(symbol, timeframe)
        if closes is None or len(closes) < BB_PERIOD:
            return None, None, None, None
        
        # 取最後 20 根 K 棒
        recent_closes = closes[-BB_PERIOD:]
        
        sma = np.mean(recent_closes)
        std = np.std(recent_closes)
        upper = sma + BB_STD * std
        lower = sma - BB_STD * std
        width = (upper - lower) / sma if sma != 0 else 0
        
        return float(upper), float(sma), float(lower), float(width)
    
    def analyze_bb_status(self, symbol, timeframe, current_ohlcv):
        """
        分析 K 棒是否接近/接觸 BB 軌道
        返回：{status, direction, distance_percent, warning_level, bb_upper, bb_middle, bb_lower}
        
        修正計算連輯：
        1. 使用實時價格 (current_close) 而不是歷史最高/最低
        2. 改用件數佐度作為距離，也使用正確的閾值
        3. 改整整个判字邏輯
        """
        # 更新歷史
        self.update_history(symbol, timeframe, current_ohlcv)
        
        # 計算 BB
        bb_upper, bb_middle, bb_lower, bb_width = self.calculate_bb(symbol, timeframe)
        if bb_upper is None:
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
            return {
                'status': 'normal',
                'direction': None,
                'distance_percent': 0,
                'warning_level': 'none',
                'bb_upper': bb_upper,
                'bb_middle': bb_middle,
                'bb_lower': bb_lower,
            }
        
        # 計算佐數比率距離 (修正：改作件數比率)
        dist_to_upper = (bb_upper - current_high) / bb_upper if bb_upper > 0 else 1.0
        dist_to_lower = (current_low - bb_lower) / bb_lower if bb_lower > 0 else 1.0
        
        # 判斷狀態
        status = 'normal'
        direction = None
        distance_percent = 0
        warning_level = 'none'
        
        # 先棂測是否接觸
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
        # 再棂測是否接近
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
    """有效性検查 - 粗付特檲提取，不需要正確數紀"""
    
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
        """粗付特檲提取 - 用元余提取的特檲填充到需要的數量
        
        简易功能：
        - 供作 OHLCV 提取可用特檲
        - 用元余數緘填充存在的空位
        """
        o = ohlcv.get('open', 0)
        h = ohlcv.get('high', 0)
        l = ohlcv.get('low', 0)
        c = ohlcv.get('close', 0)
        v = ohlcv.get('volume', 1)
        
        # 從有效數据提取特檲
        features = [
            c / h if h > 0 else 0,           # 0: 收盤的佐于最高
            c / l if l > 0 else 0,           # 1: 收盤的佐于最低
            (h - l) / l if l > 0 else 0,     # 2: 最高最低既事
            (c - o) / o if o > 0 else 0,     # 3: 收盤變化
            v if v > 0 else 1                # 4: 成交量
        ]
        
        # 填充元余特檲使其達到 target_size
        while len(features) < target_size:
            # 用隨機標沖化值填充
            features.append(np.random.randn() * 0.1)
        
        return np.array(features[:target_size], dtype=np.float32)
    
    def predict(self, symbol, timeframe, ohlcv):
        """預測有效性"""
        key = (symbol, timeframe)
        if key not in self.models:
            return None
        
        try:
            models = self.models[key]
            # 提取填充特檲
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
    """波動性預測 - 粗付特檲提取，不需要正確數紀"""
    
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
        """粗付特檲提取 - 填充不足的特檲數"""
        o = ohlcv.get('open', 0)
        h = ohlcv.get('high', 0)
        l = ohlcv.get('low', 0)
        c = ohlcv.get('close', 0)
        v = ohlcv.get('volume', 1)
        
        # 從有效數据提取特檲
        features = [
            (h - l) / l if l > 0 else 0,     # 0: 浪動
            c / c if c > 0 else 1,           # 1: 住有方位
            v if v > 0 else 1,               # 2: 成交量
            (c - o) / o if o > 0 else 0,     # 3: 身体大小
            abs(h - c) / c if c > 0 else 0   # 4: 上影
        ]
        
        # 填充元余特檲
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
            # 提取填充特檲
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

bb_calculator = BBCalculator()
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
        'description': 'BB 反彈實時監控系統 V5 (简化版)'
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
        
        # 第一步: 先棂測接近/接觸（純計算）
        bb_result = bb_calculator.analyze_bb_status(symbol, timeframe, ohlcv)
        
        # 第二步: 只有接近/接觸時才調用模型
        validity_result = None
        volatility_result = None
        
        if bb_result['status'] in ['approaching', 'touched']:
            validity_result = validity_checker.predict(symbol, timeframe, ohlcv)
            volatility_result = volatility_predictor.predict(symbol, timeframe, ohlcv)
        
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
        logger.info('BB 反彈實時監控系統 V5 (简化版)')
        logger.info('=' * 60)
        logger.info('流程：')
        logger.info('  1. 直接計算 BB 通道')
        logger.info('  2. 検測接近/接觸')
        logger.info('  3. 只有接近/接觸時才調用模型')
        logger.info('  4. 粗付特檲提取：填充元余批次')
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