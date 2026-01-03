#!/usr/bin/env python3
"""
BB反彈ML系統 - 實時服務 V5 (簡化版)
直接計算 BB 通道，檢測 K 棒高低點接近程度
無需分類器
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

# 接近閾值
APPROACHING_THRESHOLD = 0.02  # 2%
WARNING_THRESHOLD = 0.05      # 5%
DANGER_THRESHOLD = 0.01       # 1%

# ============================================================
# BB 計算器
# ============================================================

class BBCalculator:
    """直接計算 BB 通道，不使用分類器"""
    
    def __init__(self, history_size=100):
        self.history_size = history_size
        self.history = {}  # {(symbol, timeframe): deque of OHLCV}
    
    def update_history(self, symbol, timeframe, ohlcv_data):
        """更新歷史數據"""
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
    
    def _get_highs(self, symbol, timeframe):
        """獲取所有最高價"""
        key = (symbol, timeframe)
        if key not in self.history or len(self.history[key]) == 0:
            return None
        return np.array([d.get('high', 0) for d in list(self.history[key])])
    
    def _get_lows(self, symbol, timeframe):
        """獲取所有最低價"""
        key = (symbol, timeframe)
        if key not in self.history or len(self.history[key]) == 0:
            return None
        return np.array([d.get('low', 0) for d in list(self.history[key])])
    
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
    
    def get_lookback_highs_lows(self, symbol, timeframe, lookback=20):
        """獲取過去 N 根 K 棒的最高點和最低點"""
        highs = self._get_highs(symbol, timeframe)
        lows = self._get_lows(symbol, timeframe)
        
        if highs is None or lows is None:
            return None, None
        
        if len(highs) < lookback:
            # 如果歷史不足，就用全部
            lookback = len(highs)
        
        lookback_highs = highs[-lookback:]
        lookback_lows = lows[-lookback:]
        
        max_high = float(np.max(lookback_highs))
        min_low = float(np.min(lookback_lows))
        
        return max_high, min_low
    
    def analyze_approaching(self, symbol, timeframe, current_ohlcv):
        """
        分析 K 棒是否接近 BB 軌道
        返回：{approaching, direction, distance_percent, warning_level}
        """
        # 更新歷史
        self.update_history(symbol, timeframe, current_ohlcv)
        
        # 計算 BB
        bb_upper, bb_middle, bb_lower, bb_width = self.calculate_bb(symbol, timeframe)
        if bb_upper is None:
            return {
                'approaching': False,
                'direction': None,
                'distance_percent': 0,
                'warning_level': 'none',
                'bb_upper': None,
                'bb_middle': None,
                'bb_lower': None
            }
        
        # 獲取過去 20 根 K 棒的最高/最低點
        lookback_high, lookback_low = self.get_lookback_highs_lows(symbol, timeframe, lookback=20)
        if lookback_high is None:
            lookback_high = current_ohlcv.get('high', 0)
            lookback_low = current_ohlcv.get('low', 0)
        
        # 計算到上軌的距離
        dist_to_upper = (bb_upper - lookback_high) / (lookback_high + 1e-8) if lookback_high > 0 else 1.0
        
        # 計算到下軌的距離
        dist_to_lower = (lookback_low - bb_lower) / (lookback_low + 1e-8) if lookback_low > 0 else 1.0
        
        # 判斷接近狀態
        approaching = False
        direction = None
        distance_percent = 0
        warning_level = 'none'
        
        # 距離百分比（用於顯示）
        distance_percent = abs(min(dist_to_upper, dist_to_lower)) * 100
        
        # 判斷接近程度
        if dist_to_upper <= DANGER_THRESHOLD:
            # 上軌 - 危險等級
            approaching = True
            direction = 'upper'
            warning_level = 'danger'
        elif dist_to_upper <= APPROACHING_THRESHOLD:
            # 上軌 - 接近等級
            approaching = True
            direction = 'upper'
            warning_level = 'warning'
        elif dist_to_upper <= WARNING_THRESHOLD:
            # 上軌 - 警告等級
            direction = 'upper'
            warning_level = 'caution'
        elif dist_to_lower <= DANGER_THRESHOLD:
            # 下軌 - 危險等級
            approaching = True
            direction = 'lower'
            warning_level = 'danger'
        elif dist_to_lower <= APPROACHING_THRESHOLD:
            # 下軌 - 接近等級
            approaching = True
            direction = 'lower'
            warning_level = 'warning'
        elif dist_to_lower <= WARNING_THRESHOLD:
            # 下軌 - 警告等級
            direction = 'lower'
            warning_level = 'caution'
        
        return {
            'approaching': approaching,
            'direction': direction,
            'distance_percent': distance_percent,
            'warning_level': warning_level,
            'bb_upper': bb_upper,
            'bb_middle': bb_middle,
            'bb_lower': bb_lower,
            'lookback_high': lookback_high,
            'lookback_low': lookback_low
        }

# ============================================================
# 初始化
# ============================================================

bb_calculator = BBCalculator()

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
    """
    預測 K 棒是否接近 BB 軌道
    
    輸入：
    {
        "symbol": "BTCUSDT",
        "timeframe": "15m",
        "ohlcv": {
            "open": 45000,
            "high": 45100,
            "low": 44900,
            "close": 45050,
            "volume": 1000
        }
    }
    """
    try:
        data = request.get_json()
        symbol = data.get('symbol', '').upper()
        timeframe = data.get('timeframe', '15m')
        ohlcv = data.get('ohlcv', {})
        
        if not symbol or symbol not in SYMBOLS:
            return jsonify({'error': f'無效的幣種: {symbol}'}), 400
        if timeframe not in TIMEFRAMES:
            return jsonify({'error': f'無效的時間框架: {timeframe}'}), 400
        
        # 分析接近程度
        result = bb_calculator.analyze_approaching(symbol, timeframe, ohlcv)
        
        return jsonify({
            'symbol': symbol,
            'timeframe': timeframe,
            'timestamp': datetime.now().isoformat(),
            'bb_touch': {
                'approaching': result['approaching'],
                'direction': result['direction'],
                'distance_percent': result['distance_percent'],
                'warning_level': result['warning_level'],
                'bb_upper': result['bb_upper'],
                'bb_middle': result['bb_middle'],
                'bb_lower': result['bb_lower'],
                'lookback_high': result['lookback_high'],
                'lookback_low': result['lookback_low']
            }
        })
    
    except Exception as e:
        logger.error(f'預測錯誤: {e}', exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    """
    批量預測多個幣種
    """
    try:
        data = request.get_json()
        symbols = data.get('symbols', SYMBOLS)
        timeframe = data.get('timeframe', '15m')
        ohlcv_data = data.get('ohlcv_data', {})
        
        results = []
        for symbol in symbols:
            if symbol not in ohlcv_data:
                continue
            
            result = bb_calculator.analyze_approaching(symbol, timeframe, ohlcv_data[symbol])
            
            results.append({
                'symbol': symbol,
                'timeframe': timeframe,
                'bb_touch': {
                    'approaching': result['approaching'],
                    'direction': result['direction'],
                    'distance_percent': result['distance_percent'],
                    'warning_level': result['warning_level'],
                    'bb_upper': result['bb_upper'],
                    'bb_middle': result['bb_middle'],
                    'bb_lower': result['bb_lower']
                }
            })
        
        return jsonify({
            'timestamp': datetime.now().isoformat(),
            'results': results,
            'count': len(results)
        })
    
    except Exception as e:
        logger.error(f'批量預測錯誤: {e}', exc_info=True)
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    try:
        logger.info('=' * 60)
        logger.info('BB 反彈實時監控系統 V5 (簡化版)')
        logger.info('=' * 60)
        logger.info('功能：')
        logger.info('  1. 直接計算 BB 通道')
        logger.info('  2. 檢測 K 棒高低點接近程度')
        logger.info('  3. 三級警告系統 (danger/warning/caution)')
        logger.info('  4. 無需機器學習模型')
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
