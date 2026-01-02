import pandas as pd
import numpy as np
from typing import List, Dict

class ValidityFeatures:
    """
    為軌道有效性模型提取特徵
    """
    
    def __init__(self, 
                 bb_period=20, 
                 bb_std=2,
                 lookahead=10):
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.lookahead = lookahead
    
    def calculate_bollinger_bands(self, df: pd.DataFrame) -> pd.DataFrame:
        """計算 BB 軌道"""
        df = df.copy()
        close_col = 'close' if 'close' in df.columns else 'Close'
        
        df['sma'] = df[close_col].rolling(window=self.bb_period).mean()
        df['std'] = df[close_col].rolling(window=self.bb_period).std()
        df['bb_upper'] = df['sma'] + (self.bb_std * df['std'])
        df['bb_lower'] = df['sma'] - (self.bb_std * df['std'])
        df['bb_middle'] = df['sma']
        df['bb_width'] = df['bb_upper'] - df['bb_lower']
        
        return df
    
    def calculate_rsi(self, df: pd.DataFrame, period=14) -> pd.DataFrame:
        """計算 RSI"""
        df = df.copy()
        close_col = 'close' if 'close' in df.columns else 'Close'
        
        delta = df[close_col].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        df['rsi'] = df['rsi'].fillna(50)
        
        return df
    
    def calculate_atr(self, df: pd.DataFrame, period=14) -> pd.Series:
        """計算 ATR"""
        high = df['high'] if 'high' in df.columns else df['High']
        low = df['low'] if 'low' in df.columns else df['Low']
        close_col = 'close' if 'close' in df.columns else 'Close'
        
        tr1 = high - low
        tr2 = (high - df[close_col].shift()).abs()
        tr3 = (low - df[close_col].shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        return atr
    
    # ============ 一組：動量特徵 ============
    
    def momentum_decay_rate(self, df: pd.DataFrame, window: int = 3) -> pd.Series:
        """
        動量衰減率
        描述: 動量在觸碰時是否在衰減
        公式: (prev_momentum - curr_momentum) / |prev_momentum|
        較高值 = 動量衰減黛著
        """
        close_col = 'close' if 'close' in df.columns else 'Close'
        momentum = df[close_col].diff(window)
        momentum_prev = momentum.shift(1)
        
        decay = np.zeros(len(df))
        mask = momentum_prev.abs() > 1e-8
        decay[mask] = (momentum_prev[mask] - momentum[mask]) / momentum_prev[mask].abs()
        
        return pd.Series(decay, index=df.index, name='momentum_decay_rate')
    
    def momentum_reversal_strength(self, df: pd.DataFrame) -> pd.Series:
        """
        反向動量強度
        描述: 反向動量相對於原動量的大小
        公式: |reversal_momentum| / |original_momentum|
        較高值 = 反向力量強
        """
        close_col = 'close' if 'close' in df.columns else 'Close'
        
        # 計算 1根剈的動量
        returns = df[close_col].pct_change()
        
        # 反向力量
        reversal_strength = np.zeros(len(df))
        
        for i in range(1, len(df)):
            if returns.iloc[i] * returns.iloc[i-1] < 0:  # 反向
                reversal_strength[i] = abs(returns.iloc[i]) / (abs(returns.iloc[i-1]) + 1e-8)
        
        return pd.Series(reversal_strength, index=df.index, name='momentum_reversal_strength')
    
    def volume_momentum_ratio(self, df: pd.DataFrame, window: int = 5) -> pd.Series:
        """
        成交量有效性
        描述: 反向時成交量是否增加
        公式: 反向成交量 / 平均成交量
        """
        close_col = 'close' if 'close' in df.columns else 'Close'
        returns = df[close_col].pct_change()
        avg_volume = df['volume'].rolling(window=window).mean()
        
        ratio = np.zeros(len(df))
        for i in range(1, len(df)):
            if returns.iloc[i] * returns.iloc[i-1] < 0:  # 反向
                ratio[i] = df['volume'].iloc[i] / (avg_volume.iloc[i] + 1e-8)
        
        return pd.Series(ratio, index=df.index, name='volume_momentum_ratio')
    
    # ============ 二組：反彈強度特徵 ============
    
    def bounce_height_ratio(self, df: pd.DataFrame, window: int = None) -> pd.Series:
        """
        反彈高度比
        描述: 未來 window 根 K 棒的反彈高度 vs BB寶寬
        公式: (未來最高 - 當前) / BB_width
        較高值 = 反彈力度強
        """
        if window is None:
            window = self.lookahead
        
        close_col = 'close' if 'close' in df.columns else 'Close'
        high_col = 'high' if 'high' in df.columns else 'High'
        
        ratio = np.zeros(len(df))
        
        for i in range(len(df) - window):
            future_high = df[high_col].iloc[i:i+window].max()
            current_price = df[close_col].iloc[i]
            bb_width = df['bb_width'].iloc[i]
            
            if bb_width > 1e-8:
                ratio[i] = (future_high - current_price) / bb_width
        
        return pd.Series(ratio, index=df.index, name='bounce_height_ratio')
    
    def time_to_recovery(self, df: pd.DataFrame, window: int = None) -> pd.Series:
        """
        恢複时間
        描述: 需要多少根 K 棒恢複到軌道
        公式: 第一次恢複的覢标 (K 数)
        """
        if window is None:
            window = self.lookahead
        
        close_col = 'close' if 'close' in df.columns else 'Close'
        
        recovery_time = np.zeros(len(df))
        
        for i in range(len(df) - window):
            bb_middle = df['bb_middle'].iloc[i]
            
            # 找到第一次回到中軸的位置
            for j in range(i, i + window):
                if df[close_col].iloc[j] > bb_middle:
                    recovery_time[i] = j - i
                    break
        
        return pd.Series(recovery_time, index=df.index, name='time_to_recovery')
    
    def breakout_distance(self, df: pd.DataFrame, window: int = None) -> pd.Series:
        """
        突破距離
        描述: 未來是否突破軌道及突破距離
        公式: 最高價 / 上軌 - 1 或 下軌 / 最低價 - 1
        較高值 = 突破力墨著
        """
        if window is None:
            window = self.lookahead
        
        close_col = 'close' if 'close' in df.columns else 'Close'
        high_col = 'high' if 'high' in df.columns else 'High'
        low_col = 'low' if 'low' in df.columns else 'Low'
        
        distance = np.zeros(len(df))
        
        for i in range(len(df) - window):
            future_high = df[high_col].iloc[i:i+window].max()
            future_low = df[low_col].iloc[i:i+window].min()
            bb_upper = df['bb_upper'].iloc[i]
            bb_lower = df['bb_lower'].iloc[i]
            
            # 上軌突破
            if future_high > bb_upper:
                distance[i] = (future_high / bb_upper - 1) * 100
            # 下軌突破
            elif future_low < bb_lower:
                distance[i] = -(bb_lower / future_low - 1) * 100
        
        return pd.Series(distance, index=df.index, name='breakout_distance')
    
    # ============ 三組：確認特徵 ============
    
    def rsi_at_touch(self, df: pd.DataFrame) -> pd.Series:
        """
        觸碰時 RSI 值
        描述: 放粗控宇索指旋rsi_level：控宇數字 0-100】4】5了日値。
        較低 RSI = 支撐強, 較高 RSI = 阻力強
        """
        if 'rsi' not in df.columns:
            df = self.calculate_rsi(df)
        
        return df['rsi'].copy()
    
    def volume_at_touch(self, df: pd.DataFrame) -> pd.Series:
        """
        觸碰時成交量
        描述: 放粗控宇索指數，比較平均成交量
        公式: 當前成交量 / 平均成交量
        """
        avg_volume = df['volume'].rolling(window=20).mean()
        volume_ratio = df['volume'] / (avg_volume + 1e-8)
        
        return volume_ratio.copy().rename('volume_at_touch')
    
    def ema_slope(self, df: pd.DataFrame, period: int = 5) -> pd.Series:
        """
        短期 EMA 斜率
        描述: EMA 的斜率代表趨勢強度
        公式: (curr_ema - prev_ema) / prev_ema
        較高值 = 上升趨勢強
        """
        close_col = 'close' if 'close' in df.columns else 'Close'
        ema = df[close_col].ewm(span=period).mean()
        slope = (ema - ema.shift(1)) / (ema.shift(1) + 1e-8)
        
        return slope.copy().rename('ema_slope')
    
    # ============ 四組：風險特徵 ============
    
    def volatility_regime(self, df: pd.DataFrame, window: int = 20) -> pd.Series:
        """
        波動率氧確
        描述: 當前波動率 vs 平均波動率
        公式: 當前波動 / 平均波動
        較高值 = 波動率較高（既存風險）
        """
        close_col = 'close' if 'close' in df.columns else 'Close'
        returns = df[close_col].pct_change()
        volatility = returns.rolling(window=window).std()
        avg_volatility = volatility.rolling(window=window).mean()
        
        regime = volatility / (avg_volatility + 1e-8)
        
        return regime.copy().rename('volatility_regime')
    
    def bb_width_ratio(self, df: pd.DataFrame, window: int = 20) -> pd.Series:
        """
        BB 寶寬比
        描述: 當前 BB 寶寬 vs 平均 BB 寶寬
        公式: curr_bb_width / avg_bb_width
        較高值 = 當前波動率較高
        """
        avg_bb_width = df['bb_width'].rolling(window=window).mean()
        ratio = df['bb_width'] / (avg_bb_width + 1e-8)
        
        return ratio.copy().rename('bb_width_ratio')
    
    def consecutive_touches(self, df: pd.DataFrame, touch_col: str = 'touch', window: int = 10) -> pd.Series:
        """
        連續觸碰數量
        描述: 過去 N 根 K 棒中，觸碰的數量
        較高值 = 觸碰频筁(較弱的信詞)
        """
        consecutive = np.zeros(len(df))
        
        for i in range(len(df)):
            start = max(0, i - window)
            consecutive[i] = (df[touch_col].iloc[start:i] != 0).sum()
        
        return pd.Series(consecutive, index=df.index, name='consecutive_touches')
    
    # ============ 綜合特徵一次提取 ============
    
    def extract_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        一次提取所有特徵
        """
        df = df.copy()
        
        # 基础指標
        df = self.calculate_bollinger_bands(df)
        df = self.calculate_rsi(df)
        df['atr'] = self.calculate_atr(df)
        
        # 動量特徵
        df['momentum_decay_rate'] = self.momentum_decay_rate(df)
        df['momentum_reversal_strength'] = self.momentum_reversal_strength(df)
        df['volume_momentum_ratio'] = self.volume_momentum_ratio(df)
        
        # 反彈強度特徵
        df['bounce_height_ratio'] = self.bounce_height_ratio(df)
        df['time_to_recovery'] = self.time_to_recovery(df)
        df['breakout_distance'] = self.breakout_distance(df)
        
        # 確認特徵
        df['rsi_level'] = self.rsi_at_touch(df)
        df['volume_ratio'] = self.volume_at_touch(df)
        df['ema_slope'] = self.ema_slope(df)
        
        # 風險特徵
        df['volatility_regime'] = self.volatility_regime(df)
        df['bb_width_ratio'] = self.bb_width_ratio(df)
        df['consecutive_touches'] = self.consecutive_touches(df)
        
        # 此外澳幹一些优雛特徵
        df['price_to_bb_middle'] = (df['close'] - df['bb_middle']) / df['bb_middle']
        df['dist_lower_norm'] = (df['close'] - df['bb_lower']) / df['bb_width']
        df['dist_upper_norm'] = (df['bb_upper'] - df['close']) / df['bb_width']
        
        # 清理 NaN
        df = df.ffill().bfill()
        
        return df
    
    def get_feature_names(self) -> List[str]:
        """
        返回所有特徵名稱
        """
        features = [
            # 動量特徵
            'momentum_decay_rate',
            'momentum_reversal_strength',
            'volume_momentum_ratio',
            
            # 反彈強度特徵
            'bounce_height_ratio',
            'time_to_recovery',
            'breakout_distance',
            
            # 確認特徵
            'rsi_level',
            'volume_ratio',
            'ema_slope',
            
            # 風險特徵
            'volatility_regime',
            'bb_width_ratio',
            'consecutive_touches',
            
            # 优雛特徵
            'price_to_bb_middle',
            'dist_lower_norm',
            'dist_upper_norm',
            'rsi',
            'atr',
        ]
        
        return features


if __name__ == '__main__':
    # 測試
    import ccxt
    
    print('\n正在下載測試數據...')
    exchange = ccxt.binance({'enableRateLimit': True})
    klines = exchange.fetch_ohlcv('BTCUSDT', '1h', limit=500)
    
    df = pd.DataFrame(
        klines,
        columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
    )
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.rename(columns={'timestamp': 'time'})
    
    # 提取特徵
    print('\n提取特徵...')
    extractor = ValidityFeatures()
    df_features = extractor.extract_all_features(df)
    
    # 顯示特徵
    feature_names = extractor.get_feature_names()
    print(f'\n✅ 已提取 {len(feature_names)} 個特徵：')
    for i, name in enumerate(feature_names, 1):
        print(f'  {i:2d}. {name}')
    
    print(f'\n特徵異樣值：')
    print(df_features[feature_names].tail(10))
