import os
import json
import pickle
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import deque
import traceback


class RealtimeBBDetectorV2Enhanced:
    """
    企業級實時 BB Bounce 檢測器 V2 增強版 - 三層架構 + 多時間框架 + 波動預測
    
    層級 1: 分類器 (快速掃描, 10-20ms)
        - 輸入: K線數據 (open, high, low, close, volume, bb_upper, bb_lower)
        - 輸出: 0 = 不接近軌道 / 1 = 接近上軌 / 2 = 接近下軌
    
    層級 2: 有效性模型 (按需驗證, 50-100ms)
        - 輸入: 若層級1 == 1 或 2 才運行
        - 輸出: 是否為有效的支撐/阻力信號 (0-1 概率)
    
    層級 3: 波動預測模型 (預測未來波動, 20-50ms)
        - 輸入: 技術指標
        - 輸出: 未來 24h 預測波動率 (年化或百分比)
    
    特性:
    - 支持 15m 和 1h 時間框架
    - 每個幣種可同時加載多個時間框架的模型
    - 波動預測集成，用於風險管理
    - 組合信心度 (validity_prob × vol_confidence)
    """

    def __init__(
        self,
        symbols: List[str] = None,
        model_dir: str = "models",
        device: str = "cpu",
        history_window: int = 100,
        supported_timeframes: List[str] = None,
    ):
        # 完整的 22 個幣種列表
        self.symbols = symbols or [
            "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
            "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "TONUSDT", "LINKUSDT",
            "OPUSDT", "ARBUSDT", "SUIUSDT", "APEUSDT", "MATICUSDT",
            "LTCUSDT", "TRXUSDT", "FTMUSDT", "INJUSDT", "SEIUSDT",
            "TIAUSDT", "ORDIUSDT",
        ]
        
        # 支持的時間框架
        self.supported_timeframes = supported_timeframes or ["15m", "1h"]
        
        self.model_dir = model_dir
        self.device = device
        self.history_window = history_window
        
        # 分類器模型 (層級1) - 檢測是否接近 BB 上下軌
        # 結構: {symbol: {timeframe: classifier_model}}
        self.classifiers = {}
        
        # 有效性模型 (層級2) - 檢測是否為有效支撐/阻力
        # 結構: {symbol: {timeframe: validity_model}}
        self.validity_models = {}
        
        # Scaler for validity models
        # 結構: {symbol: {timeframe: scaler}}
        self.validity_scalers = {}
        
        # Feature names for validity models
        # 結構: {symbol: {timeframe: feature_names}}
        self.validity_feature_names = {}
        
        # 波動預測模型 (層級3) - 預測未來波動大小
        # 結構: {symbol: vol_model} (通常只有 1h)
        self.vol_models = {}
        
        # Scaler for vol models
        self.vol_scalers = {}
        
        # 歷史數據緩衝區 (用來計算技術指標)
        # 結構: {symbol: {timeframe: deque of candles}}
        self.candle_buffer = {}
        for sym in self.symbols:
            self.candle_buffer[sym] = {}
            for tf in self.supported_timeframes:
                self.candle_buffer[sym][tf] = deque(maxlen=history_window)
        
        # 上次信號時間戳 (避免重複推送)
        self.last_signal_time = {}
        
        # 加載模型
        self._load_models()

    def _load_models(self):
        """從 models 目錄加載所有分類器、有效性模型和波動預測模型
        
        支持的目錄結構:
        1. 分類器 (BB Classifier)：
           models/bb_models/SYMBOL/15m/model.pkl
           models/bb_models/SYMBOL/1h/model.pkl
        
        2. 有效性模型 (Validity)：
           models/validity_models/SYMBOL/15m/validity_model.pkl
           models/validity_models/SYMBOL/15m/scaler.pkl
           models/validity_models/SYMBOL/15m/feature_names.pkl
           models/validity_models/SYMBOL/1h/validity_model.pkl
           models/validity_models/SYMBOL/1h/scaler.pkl
           models/validity_models/SYMBOL/1h/feature_names.pkl
        
        3. 波動預測模型 (Volatility)：
           models/vol_models/SYMBOL/1h/model_regression.pkl
           models/vol_models/SYMBOL/1h/scaler_regression.pkl
        """
        if not os.path.exists(self.model_dir):
            print(f"[RealtimeBBDetectorV2Enhanced] Warning: models dir not found: {self.model_dir}")
            return
        
        for sym in self.symbols:
            # 初始化該 symbol 的字典
            self.classifiers[sym] = {}
            self.validity_models[sym] = {}
            self.validity_scalers[sym] = {}
            self.validity_feature_names[sym] = {}
            
            # ===== 加載分類器模型 (BB Classifier) =====
            for timeframe in self.supported_timeframes:
                classifier_path = os.path.join(
                    self.model_dir, "bb_models", sym, timeframe, "model.pkl"
                )
                if os.path.exists(classifier_path):
                    try:
                        with open(classifier_path, "rb") as f:
                            self.classifiers[sym][timeframe] = pickle.load(f)
                        print(f"[RealtimeBBDetectorV2Enhanced] Loaded classifier for {sym} {timeframe}")
                    except Exception as e:
                        print(f"[RealtimeBBDetectorV2Enhanced] Failed to load classifier {sym} {timeframe}: {e}")
            
            # ===== 加載有效性模型 (Validity Model) =====
            for timeframe in self.supported_timeframes:
                validity_path = os.path.join(
                    self.model_dir, "validity_models", sym, timeframe, "validity_model.pkl"
                )
                scaler_path = os.path.join(
                    self.model_dir, "validity_models", sym, timeframe, "scaler.pkl"
                )
                feature_names_path = os.path.join(
                    self.model_dir, "validity_models", sym, timeframe, "feature_names.pkl"
                )
                
                if os.path.exists(validity_path):
                    try:
                        with open(validity_path, "rb") as f:
                            self.validity_models[sym][timeframe] = pickle.load(f)
                        
                        if os.path.exists(scaler_path):
                            with open(scaler_path, "rb") as f:
                                self.validity_scalers[sym][timeframe] = pickle.load(f)
                        
                        if os.path.exists(feature_names_path):
                            with open(feature_names_path, "rb") as f:
                                self.validity_feature_names[sym][timeframe] = pickle.load(f)
                        
                        print(f"[RealtimeBBDetectorV2Enhanced] Loaded validity model for {sym} {timeframe}")
                    except Exception as e:
                        print(f"[RealtimeBBDetectorV2Enhanced] Failed to load validity model {sym} {timeframe}: {e}")
            
            # ===== 加載波動預測模型 (Volatility Model) =====
            vol_path = os.path.join(
                self.model_dir, "vol_models", sym, "1h", "model_regression.pkl"
            )
            vol_scaler_path = os.path.join(
                self.model_dir, "vol_models", sym, "1h", "scaler_regression.pkl"
            )
            
            if os.path.exists(vol_path):
                try:
                    with open(vol_path, "rb") as f:
                        self.vol_models[sym] = pickle.load(f)
                    
                    if os.path.exists(vol_scaler_path):
                        with open(vol_scaler_path, "rb") as f:
                            self.vol_scalers[sym] = pickle.load(f)
                    
                    print(f"[RealtimeBBDetectorV2Enhanced] Loaded volatility model for {sym}")
                except Exception as e:
                    print(f"[RealtimeBBDetectorV2Enhanced] Failed to load volatility model {sym}: {e}")

    def add_candle(self, symbol: str, candle: Dict, timeframe: str = "15m"):
        """
        將新 K線添加到緩衝區
        
        candle 格式:
        {
            "timestamp": 1735880000000,
            "open": 42500.0,
            "high": 42800.0,
            "low": 42200.0,
            "close": 42600.0,
            "volume": 120.5,
            "bb_upper": 43000.0,
            "bb_middle": 42500.0,
            "bb_lower": 42000.0,
            "bb_width": 1000.0,
            "rsi": 68.5,
            "adx": 24.1,
            "atr": 400.0
        }
        """
        if symbol in self.candle_buffer:
            if timeframe in self.candle_buffer[symbol]:
                self.candle_buffer[symbol][timeframe].append(candle)

    def _extract_layer1_features(self, candles: List[Dict]) -> Optional[np.ndarray]:
        """
        提取層級1 (分類器) 特徵
        
        返回: shape (1, n_features) 或 None
        """
        if len(candles) < 5:
            return None
        
        last = candles[-1]
        
        try:
            # 基礎特徵
            bb_upper = last.get("bb_upper", 0)
            bb_lower = last.get("bb_lower", 0)
            bb_middle = last.get("bb_middle", 0)
            close = last.get("close", 0)
            
            if bb_upper == 0 or bb_lower == 0:
                return None
            
            bb_width = bb_upper - bb_lower
            if bb_width == 0:
                return None
            
            # 距離上下軌的百分比距離
            dist_to_upper = (bb_upper - close) / bb_width
            dist_to_lower = (close - bb_lower) / bb_width
            
            # BB Position 標準化 (0-1, 0.5 = 中軸)
            bb_position = (close - bb_lower) / bb_width
            bb_position = np.clip(bb_position, 0, 1)
            
            # 近期波動
            recent_closes = np.array([c.get("close", 0) for c in candles[-5:]])
            close_volatility = np.std(recent_closes) / np.mean(recent_closes) if np.mean(recent_closes) != 0 else 0
            
            # 成交量變化
            recent_vols = np.array([c.get("volume", 0) for c in candles[-5:]])
            vol_avg = np.mean(recent_vols) if len(recent_vols) > 0 else 1
            vol_ratio = last.get("volume", 1) / vol_avg if vol_avg != 0 else 1
            
            # RSI
            rsi = last.get("rsi", 50) / 100.0
            
            # ADX
            adx = last.get("adx", 20) / 50.0
            
            # 價格變化動量
            price_change = (close - candles[-2].get("close", close)) / close if close != 0 else 0
            
            features = np.array([
                dist_to_upper,
                dist_to_lower,
                bb_position,
                close_volatility,
                vol_ratio,
                rsi,
                adx,
                price_change,
            ]).reshape(1, -1)
            
            return features
        
        except Exception as e:
            print(f"[extract_layer1_features] Error: {e}")
            return None

    def _extract_layer2_features(self, candles: List[Dict], layer1_result: int) -> Optional[Tuple[np.ndarray, str]]:
        """
        提取層級2 (有效性模型) 特徵
        
        層級1結果:
            0 = 不接近軌道 (跳過)
            1 = 接近上軌 → side="short" (反彈做空)
            2 = 接近下軌 → side="long" (反彈做多)
        
        返回: (features, side) 或 (None, None)
        """
        if layer1_result == 0 or len(candles) < 10:
            return None, None
        
        side = "short" if layer1_result == 1 else "long"
        
        try:
            last = candles[-1]
            
            # 近期高低點
            recent_closes = np.array([c.get("close", 0) for c in candles[-10:]])
            recent_highs = np.array([c.get("high", 0) for c in candles[-10:]])
            recent_lows = np.array([c.get("low", 0) for c in candles[-10:]])
            
            # 支撐/阻力強度指標
            if side == "long":
                # 下軌支撐強度: 過去是否多次在此反彈
                bb_lower = last.get("bb_lower", 0)
                touches_lower = np.sum(recent_lows <= bb_lower * 1.01) / len(recent_lows)
                support_strength = touches_lower
            else:
                # 上軌阻力強度: 過去是否多次在此回落
                bb_upper = last.get("bb_upper", 0)
                touches_upper = np.sum(recent_highs >= bb_upper * 0.99) / len(recent_highs)
                support_strength = touches_upper
            
            # RSI 指標 (離extremes越遠越好)
            rsi = last.get("rsi", 50)
            if side == "long":
                rsi_indicator = (100.0 - rsi) / 100.0 if rsi > 30 else 0.5
            else:
                rsi_indicator = rsi / 100.0 if rsi < 70 else 0.5
            
            # ADX 趨勢強度 (越高越好)
            adx = last.get("adx", 20)
            adx_indicator = min(adx / 40.0, 1.0)
            
            # ATR 波動性
            atr = last.get("atr", 0)
            recent_close_mean = np.mean(recent_closes)
            atr_ratio = atr / recent_close_mean if recent_close_mean != 0 else 0.5
            
            # 近期是否有強勢反彈信號
            close = last.get("close", 0)
            prev_close = candles[-2].get("close", close)
            momentum = abs(close - prev_close) / prev_close if prev_close != 0 else 0
            
            # 收盤相對位置 (在 BB 上下軌中的位置)
            bb_upper = last.get("bb_upper", 0)
            bb_lower = last.get("bb_lower", 0)
            bb_width = bb_upper - bb_lower
            if bb_width != 0:
                close_position = (close - bb_lower) / bb_width
                close_position = np.clip(close_position, 0, 1)
            else:
                close_position = 0.5
            
            features = np.array([
                support_strength,
                rsi_indicator,
                adx_indicator,
                atr_ratio,
                momentum,
                close_position,
            ]).reshape(1, -1)
            
            return features, side
        
        except Exception as e:
            print(f"[extract_layer2_features] Error: {e}")
            return None, None

    def _extract_vol_features(self, candles: List[Dict]) -> Optional[np.ndarray]:
        """
        提取層級3 (波動預測模型) 特徵
        
        返回: shape (1, n_features) 或 None
        """
        if len(candles) < 20:
            return None
        
        try:
            last = candles[-1]
            
            # 計算歷史波動率 (過去 20 根 K 線的日均波動)
            closes = np.array([c.get("close", 0) for c in candles[-20:]])
            returns = np.diff(closes) / closes[:-1]
            historical_vol = np.std(returns)
            
            # RSI
            rsi = last.get("rsi", 50) / 100.0
            
            # ADX
            adx = last.get("adx", 20) / 50.0
            
            # ATR 相對於價格
            atr = last.get("atr", 0)
            close = last.get("close", 0)
            atr_ratio = atr / close if close != 0 else 0
            
            # BB 寬度相對於價格
            bb_upper = last.get("bb_upper", 0)
            bb_lower = last.get("bb_lower", 0)
            bb_width = bb_upper - bb_lower
            bb_width_ratio = bb_width / close if close != 0 else 0
            
            # 成交量變化
            recent_vols = np.array([c.get("volume", 0) for c in candles[-5:]])
            vol_avg = np.mean(recent_vols)
            vol_ratio = last.get("volume", vol_avg) / vol_avg if vol_avg != 0 else 1
            
            features = np.array([
                historical_vol,
                rsi,
                adx,
                atr_ratio,
                bb_width_ratio,
                vol_ratio,
            ]).reshape(1, -1)
            
            return features
        
        except Exception as e:
            print(f"[extract_vol_features] Error: {e}")
            return None

    def scan(self, symbol: str, timeframe: str = "15m") -> Optional[Dict]:
        """
        對單個幣種執行二層掃描 (可選波動預測)
        
        返回:
        {
            "symbol": "BTCUSDT",
            "timeframe": "15m",
            "side": "long" or "short",
            "bb_position_label": "Lower" or "Upper",
            "layer1_class": 0/1/2,
            "validity_prob": 0.82,
            "confidence": 0.89,
            "predicted_volatility": 0.045,
            "predicted_volatility_unit": "daily_pct_change",
            "rsi": 68.5,
            "adx": 24.1,
            "vol_ratio": 1.45,
            "timestamp": 1735880000000,
        } 或 None
        """
        
        if symbol not in self.candle_buffer:
            return None
        
        if timeframe not in self.candle_buffer[symbol]:
            return None
        
        candles = list(self.candle_buffer[symbol][timeframe])
        if len(candles) == 0:
            return None
        
        try:
            # ========== 層級 1: 分類器 ==========
            layer1_features = self._extract_layer1_features(candles)
            if layer1_features is None:
                return None
            
            if symbol in self.classifiers and timeframe in self.classifiers[symbol]:
                try:
                    layer1_class = self.classifiers[symbol][timeframe].predict(layer1_features)[0]
                except:
                    layer1_class = 0
            else:
                # 啟發式判斷
                last = candles[-1]
                close = last.get("close", 0)
                bb_upper = last.get("bb_upper", 0)
                bb_lower = last.get("bb_lower", 0)
                bb_width = bb_upper - bb_lower
                
                if bb_width == 0:
                    return None
                
                bb_position = (close - bb_lower) / bb_width
                
                if bb_position > 0.9:
                    layer1_class = 1
                elif bb_position < 0.1:
                    layer1_class = 2
                else:
                    layer1_class = 0
            
            # 若層級1 = 0，直接返回無信號
            if layer1_class == 0:
                return None
            
            # ========== 層級 2: 有效性模型 ==========
            layer2_features, side = self._extract_layer2_features(candles, layer1_class)
            
            if layer2_features is None:
                return None
            
            if symbol in self.validity_models and timeframe in self.validity_models[symbol]:
                try:
                    # 先 scale 特徵
                    if timeframe in self.validity_scalers.get(symbol, {}):
                        layer2_features_scaled = self.validity_scalers[symbol][timeframe].transform(layer2_features)
                    else:
                        layer2_features_scaled = layer2_features
                    
                    # 預測
                    if hasattr(self.validity_models[symbol][timeframe], "predict_proba"):
                        proba = self.validity_models[symbol][timeframe].predict_proba(layer2_features_scaled)[0]
                        validity_prob = proba[1] if len(proba) > 1 else 0.5
                    else:
                        pred = self.validity_models[symbol][timeframe].predict(layer2_features_scaled)[0]
                        validity_prob = 0.8 if pred == 1 else 0.3
                except Exception as e:
                    print(f"[scan] Validity model prediction error for {symbol} {timeframe}: {e}")
                    validity_prob = 0.5
            else:
                # 沒有有效性模型，取中等信心度
                validity_prob = 0.65
            
            confidence = validity_prob
            
            last = candles[-1]
            
            # 構建信號
            signal = {
                "symbol": symbol,
                "timeframe": timeframe,
                "side": side,
                "bb_position_label": "Upper" if layer1_class == 1 else "Lower",
                "layer1_class": int(layer1_class),
                "validity_prob": float(validity_prob),
                "confidence": float(confidence),
                "rsi": float(last.get("rsi", 50)),
                "adx": float(last.get("adx", 20)),
                "vol_ratio": float(last.get("volume", 0) / max(1, last.get("volume", 1))),
                "timestamp": last.get("timestamp", 0),
            }
            
            # ========== 層級 3: 波動預測 (可選) ==========
            # 若有 vol_models，則進行波動預測
            if symbol in self.vol_models:
                volatility = self.predict_volatility(symbol)
                if volatility is not None:
                    signal["predicted_volatility"] = float(volatility)
                    signal["predicted_volatility_unit"] = "daily_pct_change"
            
            return signal
        
        except Exception as e:
            print(f"[scan] Error for {symbol} {timeframe}: {e}")
            traceback.print_exc()
            return None

    def predict_volatility(self, symbol: str) -> Optional[float]:
        """
        預測該幣種未來 24 小時的波動率
        
        返回: 預測的日均波動率 (小數形式，如 0.045 = 4.5%)
        """
        if symbol not in self.candle_buffer:
            return None
        
        # 優先使用 1h 數據
        timeframe = "1h"
        if timeframe not in self.candle_buffer[symbol]:
            return None
        
        candles = list(self.candle_buffer[symbol][timeframe])
        if len(candles) == 0:
            return None
        
        try:
            # 提取特徵
            vol_features = self._extract_vol_features(candles)
            if vol_features is None:
                return None
            
            # 若有 vol_model，使用預測
            if symbol in self.vol_models:
                try:
                    # 先 scale
                    if symbol in self.vol_scalers:
                        vol_features_scaled = self.vol_scalers[symbol].transform(vol_features)
                    else:
                        vol_features_scaled = vol_features
                    
                    # 預測 (假設模型輸出為日均波動率)
                    predicted_vol = self.vol_models[symbol].predict(vol_features_scaled)[0]
                    return float(predicted_vol)
                except Exception as e:
                    print(f"[predict_volatility] Model prediction error for {symbol}: {e}")
                    return None
            else:
                return None
        
        except Exception as e:
            print(f"[predict_volatility] Error for {symbol}: {e}")
            return None

    def scan_all(self, timeframe: str = "15m") -> List[Dict]:
        """
        掃描所有幣種，返回有效信號列表
        """
        signals = []
        for symbol in self.symbols:
            signal = self.scan(symbol, timeframe)
            if signal is not None:
                signals.append(signal)
        return signals

    def get_symbol_state(self, symbol: str, timeframe: str = "15m") -> Dict:
        """
        取得幣種目前狀態 (用於儀表板左側列表)
        """
        if symbol not in self.candle_buffer:
            return {"symbol": symbol, "status": "unknown"}
        
        if timeframe not in self.candle_buffer[symbol]:
            return {"symbol": symbol, "timeframe": timeframe, "status": "no_timeframe"}
        
        candles = list(self.candle_buffer[symbol][timeframe])
        if len(candles) == 0:
            return {"symbol": symbol, "timeframe": timeframe, "status": "no_data"}
        
        last = candles[-1]
        signal = self.scan(symbol, timeframe)
        
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "last_signal": signal,
            "timestamp": last.get("timestamp", 0),
            "close": float(last.get("close", 0)),
            "rsi": float(last.get("rsi", 50)),
            "status": "has_signal" if signal else "idle",
        }

    def get_all_symbols_state(self, timeframe: str = "15m") -> Dict[str, Dict]:
        """
        一次取得所有幣種的狀態 (用於初始化儀表板列表)
        """
        states = {}
        for symbol in self.symbols:
            states[symbol] = self.get_symbol_state(symbol, timeframe)
        return states
