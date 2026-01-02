import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from typing import Dict, List, Tuple
import ccxt
import asyncio
import aiohttp
from datetime import datetime, timedelta
from flask import Flask, jsonify, request
from flask_cors import CORS
import threading
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from label_generator import LabelGenerator

class RealtimePredictor:
    def __init__(self, models_dir='models'):
        self.models_dir = Path(models_dir)
        
        # 模型基路徑
        self.bb_models_dir = self.models_dir / 'bb_models'
        self.vol_models_dir = self.models_dir / 'vol_models'
        
        # 模型記戆區
        self.bb_models_cache = {}      # {(symbol, timeframe): model}
        self.bb_scalers_cache = {}     # {(symbol, timeframe): scaler}
        self.vol_models_cache = {}     # {(symbol, timeframe): model}
        self.vol_scalers_cache = {}    # {(symbol, timeframe): scaler}
        
        # 初始化 Binance 客戶端
        self.exchange = ccxt.binance({
            'enableRateLimit': True,
            'rateLimit': 100
        })
        
        self.generator = LabelGenerator()
        
        # 缓存最近的扫泶結果
        self.scan_cache = {}
        self.scan_cache_time = {}
        self.scan_interval = 5  # 秒
    
    def load_symbol_models(self, symbol: str, timeframe: str):
        """
        加輈特定幣種 + timeframe 的模式
        """
        cache_key = (symbol, timeframe)
        
        # 棄末已在記戆區中
        if cache_key in self.bb_models_cache:
            return True
        
        try:
            # BB 模式
            bb_model_dir = self.bb_models_dir / symbol / timeframe
            bb_model_path = bb_model_dir / 'model.pkl'
            bb_scaler_path = bb_model_dir / 'scaler.pkl'
            bb_label_map_path = bb_model_dir / 'label_map.pkl'
            
            if bb_model_path.exists() and bb_scaler_path.exists():
                self.bb_models_cache[cache_key] = joblib.load(bb_model_path)
                self.bb_scalers_cache[cache_key] = joblib.load(bb_scaler_path)
                
                if bb_label_map_path.exists():
                    label_map = joblib.load(bb_label_map_path)
                    inverse_map = {v: k for k, v in label_map.items()}
                else:
                    inverse_map = {0: -1, 1: 0, 2: 1}
                
                self.bb_inverse_maps = getattr(self, 'bb_inverse_maps', {})
                self.bb_inverse_maps[cache_key] = inverse_map
                
                logger.info(f'✅ 已加輈 {symbol} {timeframe} BB 模式')
            else:
                logger.warning(f'❌ {symbol} {timeframe} BB 模式不存在')
                return False
            
            # 波動性模式
            vol_model_dir = self.vol_models_dir / symbol / timeframe
            vol_model_path = vol_model_dir / 'model_regression.pkl'
            vol_scaler_path = vol_model_dir / 'scaler_regression.pkl'
            
            if vol_model_path.exists() and vol_scaler_path.exists():
                self.vol_models_cache[cache_key] = joblib.load(vol_model_path)
                self.vol_scalers_cache[cache_key] = joblib.load(vol_scaler_path)
                logger.info(f'✅ 已加輈 {symbol} {timeframe} 波動性模式')
            else:
                logger.warning(f'❌ {symbol} {timeframe} 波動性模式不存在')
            
            return cache_key in self.bb_models_cache
        
        except Exception as e:
            logger.error(f'加輈 {symbol} {timeframe} 模式失敗: {e}')
            return False
    
    def fetch_klines(self, symbol: str, timeframe: str, limit: int = 120):
        """
        從 Binance 抷取 K 線數據
        """
        try:
            klines = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(
                klines,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.rename(columns={'timestamp': 'time'})
            return df
        except Exception as e:
            logger.error(f'{symbol} {timeframe} 抷取失敗: {e}')
            return None
    
    def create_features_for_prediction(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        產產預測特录
        """
        df = df.copy()
        
        # BB 軌道
        df = self.generator.calculate_bollinger_bands(df)
        df['volatility'] = self.generator.calculate_volatility(df)
        df = self.generator.generate_bb_touch_labels(df, touch_range=0.02)
        
        # 一膳標籤
        close_col = 'close'
        df['price_to_bb_middle'] = (df[close_col] - df['bb_middle']) / df['bb_middle']
        df['dist_upper_norm'] = (df['bb_upper'] - df[close_col]) / (df['bb_upper'] - df['bb_lower'])
        df['dist_lower_norm'] = (df[close_col] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        
        # RSI
        df = self._calculate_rsi(df)
        
        # 其他特录
        df['returns'] = df[close_col].pct_change()
        df['returns_std'] = df['returns'].rolling(window=20).std()
        df['high_low_ratio'] = df['high'] / df['low'] - 1
        df['close_open_ratio'] = df[close_col] / df['open'] - 1
        
        # 移動平均
        df['sma_5'] = df[close_col].rolling(window=5).mean()
        df['sma_20'] = df[close_col].rolling(window=20).mean()
        df['sma_50'] = df[close_col].rolling(window=50).mean()
        
        return df.fillna(method='bfill').fillna(method='ffill')
    
    def _calculate_rsi(self, df: pd.DataFrame, period=14) -> pd.DataFrame:
        """計算 RSI"""
        df = df.copy()
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        df['rsi'] = df['rsi'].fillna(50)
        return df
    
    def predict_bb_signal(self, df: pd.DataFrame, symbol: str, timeframe: str) -> Dict:
        """
        預測 BB 軌道支歴/阻力信号
        """
        cache_key = (symbol, timeframe)
        
        if cache_key not in self.bb_models_cache or len(df) == 0:
            return None
        
        model = self.bb_models_cache[cache_key]
        scaler = self.bb_scalers_cache[cache_key]
        inverse_map = self.bb_inverse_maps[cache_key]
        
        # 單位最新一根
        row = df.iloc[-1:].copy()
        
        feature_cols = [
            'price_to_bb_middle', 'dist_upper_norm', 'dist_lower_norm',
            'bb_width', 'rsi', 'volatility', 'returns_std',
            'high_low_ratio', 'close_open_ratio',
            'sma_5', 'sma_20', 'sma_50'
        ]
        
        X = row[feature_cols].values
        X_scaled = scaler.transform(X)
        
        # 預測統計搩率
        proba = model.predict_proba(X_scaled)[0]
        pred_class_mapped = model.predict(X_scaled)[0]
        
        # 介旧整測標籤
        pred_class = inverse_map[pred_class_mapped]
        
        # 信心度
        confidence = float(np.max(proba))
        
        # 信号映射: -1 = 下軌, 0 = 中間, 1 = 上軌
        signal_map = {-1: 'SUPPORT', 0: 'NEUTRAL', 1: 'RESISTANCE'}
        signal = signal_map[int(pred_class)]
        
        return {
            'signal': signal,
            'confidence': confidence,
            'probabilities': {
                'support': float(proba[0]),
                'neutral': float(proba[1]),
                'resistance': float(proba[2])
            },
            'price': float(row['close'].values[0]),
            'bb_upper': float(row['bb_upper'].values[0]),
            'bb_lower': float(row['bb_lower'].values[0]),
            'bb_middle': float(row['bb_middle'].values[0])
        }
    
    def predict_volatility(self, df: pd.DataFrame, symbol: str, timeframe: str) -> Dict:
        """
        預測未來波動性
        """
        cache_key = (symbol, timeframe)
        
        if cache_key not in self.vol_models_cache or len(df) == 0:
            return None
        
        model = self.vol_models_cache[cache_key]
        scaler = self.vol_scalers_cache[cache_key]
        
        row = df.iloc[-1:].copy()
        
        feature_cols = [
            'volatility', 'bb_width', 'price_range', 'body_size',
            'rsi', 'volume_change', 'atr_ratio',
            'returns_rolling_std', 'returns_rolling_mean',
            'hist_vol_5', 'hist_vol_10', 'hist_vol_20',
            'price_to_sma', 'k_percent', 'd_percent'
        ]
        
        # 擷選合適的特录
        available_cols = [col for col in feature_cols if col in row.columns]
        X = row[available_cols].values
        
        if len(X) == 0:
            return None
        
        X_scaled = scaler.transform(X)
        pred_vol = model.predict(X_scaled)[0]
        
        return {
            'predicted_volatility': float(pred_vol),
            'current_volatility': float(row['volatility'].values[0])
        }
    
    def scan_all_symbols(self, symbols: List[str], timeframe='15m') -> List[Dict]:
        """
        扫描所有幣種的 BB 接近狀態
        """
        results = []
        
        for symbol in symbols:
            # 加輈模式
            if not self.load_symbol_models(symbol, timeframe):
                continue
            
            # 抷取數據
            df = self.fetch_klines(symbol, timeframe)
            if df is None:
                continue
            
            # 產產特录
            df = self.create_features_for_prediction(df)
            
            # 預測
            bb_pred = self.predict_bb_signal(df, symbol, timeframe)
            
            if bb_pred is not None:
                results.append({
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'timestamp': datetime.now().isoformat(),
                    'bb_signal': bb_pred,
                    'distance_to_upper': float(df['dist_to_upper'].iloc[-1]) if 'dist_to_upper' in df.columns else None,
                    'distance_to_lower': float(df['dist_to_lower'].iloc[-1]) if 'dist_to_lower' in df.columns else None
                })
        
        # 按信心度映序
        results.sort(key=lambda x: x['bb_signal']['confidence'], reverse=True)
        
        return results


def create_app(predictor: RealtimePredictor):
    """
    建立 Flask 應用程式
    """
    app = Flask(__name__)
    CORS(app)
    
    # 所有幣種
    all_symbols = [
        'AAVEUSDT', 'ADAUSDT', 'ALGOUSDT', 'ARBUSDT', 'ATOMUSDT',
        'AVAXUSDT', 'BCHUSDT', 'BNBUSDT', 'BTCUSDT', 'DOGEUSDT',
        'DOTUSDT', 'ETCUSDT', 'ETHUSDT', 'FILUSDT', 'LINKUSDT',
        'LTCUSDT', 'MATICUSDT', 'NEARUSDT', 'OPUSDT', 'SOLUSDT',
        'UNIUSDT', 'XRPUSDT'
    ]
    
    @app.route('/api/focus', methods=['POST'])
    def focus():
        """
        业注授棧幣種的實時推理
        """
        try:
            data = request.json
            symbol = data.get('symbol', 'BTCUSDT')
            timeframe = data.get('timeframe', '15m')
            
            # 加輈模式
            if not predictor.load_symbol_models(symbol, timeframe):
                return jsonify({'error': f'模式不存在: {symbol} {timeframe}'}), 400
            
            # 抷取數據
            df = predictor.fetch_klines(symbol, timeframe)
            if df is None:
                return jsonify({'error': '抷取數據失敗'}), 400
            
            # 產產特录
            df = predictor.create_features_for_prediction(df)
            
            # 預測
            bb_pred = predictor.predict_bb_signal(df, symbol, timeframe)
            vol_pred = predictor.predict_volatility(df, symbol, timeframe)
            
            return jsonify({
                'symbol': symbol,
                'timeframe': timeframe,
                'timestamp': datetime.now().isoformat(),
                'bb_signal': bb_pred,
                'vol_signal': vol_pred
            })
        
        except Exception as e:
            logger.error(f'Focus 路徑錯誤: {e}')
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/scan', methods=['GET'])
    def scan():
        """
        扫描所有幣種
        """
        try:
            timeframe = request.args.get('timeframe', '15m')
            limit = int(request.args.get('limit', 10))
            
            results = predictor.scan_all_symbols(all_symbols, timeframe)
            
            # 過濾储斈鏨隱扷幓接近的
            nearby = [r for r in results if r['bb_signal']['confidence'] > 0.5]
            
            return jsonify({
                'timestamp': datetime.now().isoformat(),
                'timeframe': timeframe,
                'total_scanned': len(results),
                'nearby_symbols': nearby[:limit]
            })
        
        except Exception as e:
            logger.error(f'Scan 路徑錯誤: {e}')
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/health', methods=['GET'])
    def health():
        """
        模式櫪功騎檢查
        """
        return jsonify({
            'status': 'ok',
            'models_loaded': len(predictor.bb_models_cache),
            'timestamp': datetime.now().isoformat()
        })
    
    return app


if __name__ == '__main__':
    predictor = RealtimePredictor()
    app = create_app(predictor)
    
    logger.info('🚀 實時推理服務正在啟動...')
    app.run(host='127.0.0.1', port=5000, debug=False, threaded=True)
