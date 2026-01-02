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
import sys

# 設定 UTF-8 編碼
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

logging.basicConfig(level=logging.INFO, encoding='utf-8')
logger = logging.getLogger(__name__)

from label_generator import LabelGenerator

class RealtimePredictor:
    def __init__(self, models_dir='models'):
        self.models_dir = Path(models_dir)
        
        # 模型基路徑
        self.bb_models_dir = self.models_dir / 'bb_models'
        self.vol_models_dir = self.models_dir / 'vol_models'
        
        # 模型記分區
        self.bb_models_cache = {}      # {(symbol, timeframe): model}
        self.bb_scalers_cache = {}     # {(symbol, timeframe): scaler}
        self.bb_inverse_maps = {}      # {(symbol, timeframe): inverse_map}
        self.vol_models_cache = {}     # {(symbol, timeframe): model}
        self.vol_scalers_cache = {}    # {(symbol, timeframe): scaler}
        
        # 初始化 Binance 客戶端
        self.exchange = ccxt.binance({
            'enableRateLimit': True,
            'rateLimit': 100
        })
        
        self.generator = LabelGenerator()
        
        # 緩存最近的掃描結果
        self.scan_cache = {}
        self.scan_cache_time = {}
        self.scan_interval = 5  # 秒
        
        # 掃描所有可用的幣種
        self._available_symbols = self._scan_available_symbols()
    
    def _scan_available_symbols(self) -> List[str]:
        """
        掃描 models/bb_models 目錄來找出所有訓練過的幣種
        """
        symbols = []
        
        if self.bb_models_dir.exists():
            for symbol_dir in self.bb_models_dir.iterdir():
                if symbol_dir.is_dir():
                    symbols.append(symbol_dir.name)
        
        logger.info(f'發現 {len(symbols)} 個已訓練的幣種: {sorted(symbols)}')
        return sorted(symbols)
    
    def get_available_symbols(self) -> List[str]:
        """
        返回所有可用的幣種
        """
        return self._available_symbols
    
    def get_available_timeframes(self, symbol: str) -> List[str]:
        """
        返回特定幣種的可用時間框架
        """
        timeframes = []
        symbol_dir = self.bb_models_dir / symbol
        
        if symbol_dir.exists():
            for tf_dir in symbol_dir.iterdir():
                if tf_dir.is_dir():
                    timeframes.append(tf_dir.name)
        
        return sorted(timeframes)
    
    def load_symbol_models(self, symbol: str, timeframe: str):
        """
        加載特定幣種 + timeframe 的模型
        """
        cache_key = (symbol, timeframe)
        
        # 已在記分區中
        if cache_key in self.bb_models_cache:
            return True
        
        try:
            # BB 模型
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
                
                self.bb_inverse_maps[cache_key] = inverse_map
                logger.info(f'已加載 {symbol} {timeframe} BB 模型')
            else:
                logger.warning(f'{symbol} {timeframe} BB 模型不存在')
                return False
            
            # 波動性模型
            vol_model_dir = self.vol_models_dir / symbol / timeframe
            vol_model_path = vol_model_dir / 'model_regression.pkl'
            vol_scaler_path = vol_model_dir / 'scaler_regression.pkl'
            
            if vol_model_path.exists() and vol_scaler_path.exists():
                self.vol_models_cache[cache_key] = joblib.load(vol_model_path)
                self.vol_scalers_cache[cache_key] = joblib.load(vol_scaler_path)
                logger.info(f'已加載 {symbol} {timeframe} 波動性模型')
            else:
                logger.warning(f'{symbol} {timeframe} 波動性模型不存在')
            
            return cache_key in self.bb_models_cache
        
        except Exception as e:
            logger.error(f'加載 {symbol} {timeframe} 模型失敗: {e}')
            return False
    
    def fetch_klines(self, symbol: str, timeframe: str, limit: int = 120):
        """
        從 Binance 擷取 K 線數據
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
            logger.error(f'{symbol} {timeframe} 擷取失敗: {e}')
            return None
    
    def create_features_for_prediction(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        產生預測特徵
        """
        df = df.copy()
        
        # BB 軌道
        df = self.generator.calculate_bollinger_bands(df)
        df['volatility'] = self.generator.calculate_volatility(df)
        
        # 一些標籤
        close_col = 'close'
        df['price_to_bb_middle'] = (df[close_col] - df['bb_middle']) / df['bb_middle']
        df['dist_upper_norm'] = (df['bb_upper'] - df[close_col]) / (df['bb_upper'] - df['bb_lower'])
        df['dist_lower_norm'] = (df[close_col] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        
        # RSI
        df = self._calculate_rsi(df)
        
        # 其他特徵
        df['returns'] = df[close_col].pct_change()
        df['returns_std'] = df['returns'].rolling(window=20).std()
        df['high_low_ratio'] = df['high'] / df['low'] - 1
        df['close_open_ratio'] = df[close_col] / df['open'] - 1
        
        # 移動平均
        df['sma_5'] = df[close_col].rolling(window=5).mean()
        df['sma_20'] = df[close_col].rolling(window=20).mean()
        df['sma_50'] = df[close_col].rolling(window=50).mean()
        
        # 波動性特徵
        df['price_range'] = (df['high'] - df['low']) / df[close_col]
        df['body_size'] = (df[close_col] - df['open']).abs() / df[close_col]
        df['volume_change'] = df['volume'].pct_change().rolling(window=5).std()
        
        # ATR
        df['atr_14'] = self._calculate_atr(df, period=14)
        df['atr_ratio'] = df['atr_14'] / df[close_col]
        
        # 歷史波動性
        df['hist_vol_5'] = df[close_col].pct_change().rolling(window=5).std()
        df['hist_vol_10'] = df[close_col].pct_change().rolling(window=10).std()
        df['hist_vol_20'] = df[close_col].pct_change().rolling(window=20).std()
        
        # 價格相對於 SMA
        df['price_to_sma'] = df[close_col] / df['sma_20']
        
        # 隨機指標
        df = self._calculate_stochastic(df)
        
        # 日期相關特徵
        df['returns_rolling_std'] = df['returns'].rolling(window=10).std()
        df['returns_rolling_mean'] = df['returns'].rolling(window=10).mean()
        
        # 使用 ffill 和 bfill 替代 fillna(method=...)
        df = df.ffill().bfill()
        
        return df
    
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
    
    def _calculate_atr(self, df: pd.DataFrame, period=14) -> pd.Series:
        """計算 ATR (Average True Range)"""
        close_col = 'close'
        high = df['high']
        low = df['low']
        
        tr1 = high - low
        tr2 = (high - df[close_col].shift()).abs()
        tr3 = (low - df[close_col].shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        return atr
    
    def _calculate_stochastic(self, df: pd.DataFrame, period=14) -> pd.DataFrame:
        """計算隨機指標"""
        df = df.copy()
        
        high = df['high']
        low = df['low']
        close_col = 'close'
        
        min_low = low.rolling(window=period).min()
        max_high = high.rolling(window=period).max()
        
        df['k_percent'] = 100 * ((df[close_col] - min_low) / (max_high - min_low))
        df['d_percent'] = df['k_percent'].rolling(window=3).mean()
        
        return df
    
    def predict_bb_signal(self, df: pd.DataFrame, symbol: str, timeframe: str) -> Dict:
        """
        預測 BB 軌道支撐/阻力信號
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
        
        # 預測統計概率
        proba = model.predict_proba(X_scaled)[0]
        pred_class_mapped = model.predict(X_scaled)[0]
        
        # 介紹整測標籤
        pred_class = inverse_map[pred_class_mapped]
        
        # 信心度
        confidence = float(np.max(proba))
        
        # 信號映射: -1 = 下軌, 0 = 中軸, 1 = 上軌
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
        
        # 波動性模型的完整特徵列表
        feature_cols = [
            'volatility', 'bb_width', 'price_range', 'body_size',
            'rsi', 'volume_change', 'atr_ratio',
            'returns_rolling_std', 'returns_rolling_mean',
            'hist_vol_5', 'hist_vol_10', 'hist_vol_20',
            'price_to_sma', 'k_percent', 'd_percent'
        ]
        
        # 確保所有特徵都存在
        X = row[feature_cols].values
        
        if X.shape[1] != len(feature_cols):
            logger.warning(f'特徵數量不匹配: {X.shape[1]} vs {len(feature_cols)}')
            return None
        
        X_scaled = scaler.transform(X)
        pred_vol = model.predict(X_scaled)[0]
        
        return {
            'predicted_volatility': float(pred_vol),
            'current_volatility': float(row['volatility'].values[0])
        }
    
    def scan_all_symbols(self, symbols: List[str], timeframe='15m') -> List[Dict]:
        """
        掃描所有幣種的 BB 接近狀態
        """
        results = []
        
        for symbol in symbols:
            # 加載模型
            if not self.load_symbol_models(symbol, timeframe):
                continue
            
            # 擷取數據
            df = self.fetch_klines(symbol, timeframe)
            if df is None:
                continue
            
            # 產生特徵
            df = self.create_features_for_prediction(df)
            
            # 預測
            bb_pred = self.predict_bb_signal(df, symbol, timeframe)
            
            if bb_pred is not None:
                results.append({
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'timestamp': datetime.now().isoformat(),
                    'bb_signal': bb_pred,
                })
        
        # 按信心度排序
        results.sort(key=lambda x: x['bb_signal']['confidence'], reverse=True)
        
        return results


def create_app(predictor: RealtimePredictor):
    """
    建立 Flask 應用程式
    """
    app = Flask(__name__)
    CORS(app)
    
    @app.route('/api/symbols', methods=['GET'])
    def get_symbols():
        """
        返回所有可用的幣種
        """
        try:
            symbols = predictor.get_available_symbols()
            return jsonify({
                'symbols': symbols,
                'count': len(symbols),
                'timestamp': datetime.now().isoformat()
            })
        except Exception as e:
            logger.error(f'Get symbols 錯誤: {e}')
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/timeframes/<symbol>', methods=['GET'])
    def get_timeframes(symbol):
        """
        返回特定幣種的可用時間框架
        """
        try:
            timeframes = predictor.get_available_timeframes(symbol)
            return jsonify({
                'symbol': symbol,
                'timeframes': timeframes,
                'timestamp': datetime.now().isoformat()
            })
        except Exception as e:
            logger.error(f'Get timeframes 錯誤: {e}')
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/focus', methods=['POST'])
    def focus():
        """
        業注授掛幣種的實時推理
        """
        try:
            data = request.json
            symbol = data.get('symbol', 'BTCUSDT')
            timeframe = data.get('timeframe', '15m')
            
            # 加載模型
            if not predictor.load_symbol_models(symbol, timeframe):
                return jsonify({'error': f'模型不存在: {symbol} {timeframe}'}), 400
            
            # 擷取數據
            df = predictor.fetch_klines(symbol, timeframe)
            if df is None:
                return jsonify({'error': '擷取數據失敗'}), 400
            
            # 產生特徵
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
            import traceback
            traceback.print_exc()
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/scan', methods=['GET'])
    def scan():
        """
        掃描所有幣種
        """
        try:
            timeframe = request.args.get('timeframe', '15m')
            limit = int(request.args.get('limit', 10))
            
            symbols = predictor.get_available_symbols()
            results = predictor.scan_all_symbols(symbols, timeframe)
            
            # 過濾鄰近接近上/下軌的
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
        模型狀功能檢查
        """
        return jsonify({
            'status': 'ok',
            'models_loaded': len(predictor.bb_models_cache),
            'available_symbols': len(predictor.get_available_symbols()),
            'timestamp': datetime.now().isoformat()
        })
    
    return app


if __name__ == '__main__':
    predictor = RealtimePredictor()
    app = create_app(predictor)
    
    logger.info('啟動實時推理服務...')
    app.run(host='127.0.0.1', port=5000, debug=False, threaded=True)
