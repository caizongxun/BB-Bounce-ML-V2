import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

class ValidityLabelGenerator:
    """
    生成軌道有效性標籤
    判斷觸碰上/下軌的反彈是否有效
    """
    
    def __init__(self, 
                 lookahead=10,           # 向前看 N 根 K 棒判斷有效性
                 min_bounce_pct=0.5,    # 最小反彈幅度 0.5%
                 momentum_decay_thresh=0.3,  # 動量衰減閾值 30%
                 bb_period=20,
                 bb_std=2):
        
        self.lookahead = lookahead
        self.min_bounce_pct = min_bounce_pct
        self.momentum_decay_thresh = momentum_decay_thresh
        self.bb_period = bb_period
        self.bb_std = bb_std
    
    def calculate_bollinger_bands(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        計算 Bollinger Bands
        """
        df = df.copy()
        close_col = 'close' if 'close' in df.columns else 'Close'
        
        df['sma'] = df[close_col].rolling(window=self.bb_period).mean()
        df['std'] = df[close_col].rolling(window=self.bb_period).std()
        df['bb_upper'] = df['sma'] + (self.bb_std * df['std'])
        df['bb_lower'] = df['sma'] - (self.bb_std * df['std'])
        df['bb_middle'] = df['sma']
        df['bb_width'] = df['bb_upper'] - df['bb_lower']
        
        return df
    
    def calculate_momentum(self, df: pd.DataFrame, period=3) -> pd.Series:
        """
        計算動量 (MOM) = 當前收盤 - N 根前收盤
        """
        close_col = 'close' if 'close' in df.columns else 'Close'
        momentum = df[close_col].diff(period)
        return momentum
    
    def calculate_returns(self, df: pd.DataFrame) -> pd.Series:
        """
        計算收益率
        """
        close_col = 'close' if 'close' in df.columns else 'Close'
        returns = df[close_col].pct_change()
        return returns
    
    def detect_touch(self, df: pd.DataFrame, touch_range: float = 0.02) -> np.ndarray:
        """
        檢測價格是否觸碰 BB 軌道
        返回：-1 (下軌), 0 (無), 1 (上軌)
        """
        close_col = 'close' if 'close' in df.columns else 'Close'
        
        touch = np.zeros(len(df), dtype=int)
        
        for i in range(len(df)):
            price = df[close_col].iloc[i]
            bb_upper = df['bb_upper'].iloc[i]
            bb_lower = df['bb_lower'].iloc[i]
            bb_middle = df['bb_middle'].iloc[i]
            
            # 下軌觸碰 (在下軌 ±2%)
            if abs(price - bb_lower) / bb_lower < touch_range:
                touch[i] = -1
            # 上軌觸碰
            elif abs(price - bb_upper) / bb_upper < touch_range:
                touch[i] = 1
        
        return touch
    
    def calculate_momentum_decay(self, df: pd.DataFrame, window: int = 5) -> pd.Series:
        """
        計算動量衰減率
        momentum_decay = (prev_momentum - curr_momentum) / |prev_momentum|
        正值表示動量在衰減
        """
        momentum = self.calculate_momentum(df, period=1)
        momentum_prev = momentum.shift(1)
        
        # 避免除以零
        decay = np.zeros(len(df))
        mask = momentum_prev.abs() > 1e-8
        decay[mask] = (momentum_prev[mask] - momentum[mask]) / momentum_prev[mask].abs()
        decay[~mask] = 0
        
        return pd.Series(decay, index=df.index)
    
    def calculate_reversal_strength(self, df: pd.DataFrame, window: int = 3) -> pd.Series:
        """
        反彈強度 = 反彈方向的力度
        計算方式：未來 N 根的最高價 - 當前價
        """
        close_col = 'close' if 'close' in df.columns else 'Close'
        high_col = 'high' if 'high' in df.columns else 'High'
        
        strength = np.zeros(len(df))
        
        for i in range(len(df) - window):
            future_max = df[high_col].iloc[i:i+window].max()
            current_price = df[close_col].iloc[i]
            strength[i] = (future_max - current_price) / current_price
        
        return pd.Series(strength, index=df.index)
    
    def is_valid_support(self, df: pd.DataFrame, touch_idx: int) -> bool:
        """
        判斷在 touch_idx 處的下軌觸碰是否是有效支撐
        """
        close_col = 'close' if 'close' in df.columns else 'Close'
        
        if touch_idx + self.lookahead >= len(df):
            return False  # 數據不足
        
        touch_price = df[close_col].iloc[touch_idx]
        bb_lower = df['bb_lower'].iloc[touch_idx]
        bb_upper = df['bb_upper'].iloc[touch_idx]
        bb_width = bb_upper - bb_lower
        
        # 檢查未來 lookahead 根 K 棒
        future_prices = df[close_col].iloc[touch_idx:touch_idx+self.lookahead]
        future_high = future_prices.max()
        future_low = future_prices.min()
        
        # 條件 1: 動量衰減檢查
        momentum_decay = self.calculate_momentum_decay(df).iloc[touch_idx]
        if momentum_decay < self.momentum_decay_thresh:
            return False  # 動量未衰減，無效
        
        # 條件 2: 反彈幅度檢查
        bounce_pct = (future_high - touch_price) / touch_price
        if bounce_pct < self.min_bounce_pct / 100:
            return False  # 反彈不足
        
        # 條件 3: 未再次觸碰下軌
        retouch_lower = (future_low < bb_lower * (1 + 0.01))  # 再次接近下軌
        if retouch_lower:
            return False  # 再次跌破，支撐無效
        
        # 條件 4: RSI 確認 (可選)
        if 'rsi' in df.columns:
            rsi_at_touch = df['rsi'].iloc[touch_idx]
            if rsi_at_touch > 40:  # 觸碰時 RSI 過高，支撐弱
                return False
        
        return True  # 所有條件滿足，有效支撐
    
    def is_valid_resistance(self, df: pd.DataFrame, touch_idx: int) -> bool:
        """
        判斷在 touch_idx 處的上軌觸碰是否是有效阻力
        """
        close_col = 'close' if 'close' in df.columns else 'Close'
        
        if touch_idx + self.lookahead >= len(df):
            return False  # 數據不足
        
        touch_price = df[close_col].iloc[touch_idx]
        bb_upper = df['bb_upper'].iloc[touch_idx]
        bb_lower = df['bb_lower'].iloc[touch_idx]
        bb_width = bb_upper - bb_lower
        
        # 檢查未來 lookahead 根 K 棒
        future_prices = df[close_col].iloc[touch_idx:touch_idx+self.lookahead]
        future_high = future_prices.max()
        future_low = future_prices.min()
        
        # 條件 1: 動量衰減檢查（向上動量衰減）
        momentum_decay = self.calculate_momentum_decay(df).iloc[touch_idx]
        if momentum_decay < self.momentum_decay_thresh:
            return False  # 上升動量未衰減，無效
        
        # 條件 2: 回檔幅度檢查
        pullback_pct = (touch_price - future_low) / touch_price
        if pullback_pct < self.min_bounce_pct / 100:
            return False  # 回檔不足
        
        # 條件 3: 未再次觸碰上軌
        retouch_upper = (future_high > bb_upper * (1 - 0.01))  # 再次接近上軌
        if retouch_upper:
            return False  # 再次突破，阻力無效
        
        # 條件 4: RSI 確認
        if 'rsi' in df.columns:
            rsi_at_touch = df['rsi'].iloc[touch_idx]
            if rsi_at_touch < 60:  # 觸碰時 RSI 過低，阻力弱
                return False
        
        return True  # 所有條件滿足，有效阻力
    
    def generate_validity_labels(self, df: pd.DataFrame, touch_range: float = 0.02) -> pd.DataFrame:
        """
        生成完整的有效性標籤
        返回 DataFrame，包含：
          - touch: 觸碰位置 (-1/0/1)
          - is_valid_support: 是否有效支撐 (0/1)
          - is_valid_resistance: 是否有效阻力 (0/1)
          - validity_label: 綜合標籤
        """
        df = df.copy()
        close_col = 'close' if 'close' in df.columns else 'Close'
        
        # 計算必要指標
        df = self.calculate_bollinger_bands(df)
        if 'rsi' not in df.columns:
            df = self._calculate_rsi(df)
        
        # 檢測觸碰
        touch = self.detect_touch(df, touch_range)
        df['touch'] = touch
        
        # 初始化標籤列
        df['is_valid_support'] = 0
        df['is_valid_resistance'] = 0
        df['validity_label'] = 0  # 0: 無效, 1: 有效支撐, -1: 有效阻力
        
        # 為每個觸碰點判斷有效性
        for i in range(len(df)):
            if touch[i] == -1:  # 下軌觸碰
                if self.is_valid_support(df, i):
                    df.loc[i, 'is_valid_support'] = 1
                    df.loc[i, 'validity_label'] = 1
            
            elif touch[i] == 1:  # 上軌觸碰
                if self.is_valid_resistance(df, i):
                    df.loc[i, 'is_valid_resistance'] = 1
                    df.loc[i, 'validity_label'] = -1
        
        return df
    
    def _calculate_rsi(self, df: pd.DataFrame, period=14) -> pd.DataFrame:
        """
        計算 RSI (Relative Strength Index)
        """
        df = df.copy()
        close_col = 'close' if 'close' in df.columns else 'Close'
        
        delta = df[close_col].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        df['rsi'] = df['rsi'].fillna(50)
        
        return df
    
    def get_validity_statistics(self, df: pd.DataFrame) -> Dict:
        """
        統計有效性信息
        """
        total_supports = (df['touch'] == -1).sum()
        valid_supports = (df['is_valid_support'] == 1).sum()
        
        total_resistances = (df['touch'] == 1).sum()
        valid_resistances = (df['is_valid_resistance'] == 1).sum()
        
        return {
            'total_support_touches': int(total_supports),
            'valid_supports': int(valid_supports),
            'support_validity_rate': valid_supports / max(total_supports, 1),
            'total_resistance_touches': int(total_resistances),
            'valid_resistances': int(valid_resistances),
            'resistance_validity_rate': valid_resistances / max(total_resistances, 1),
            'overall_validity_rate': (valid_supports + valid_resistances) / max(total_supports + total_resistances, 1)
        }


if __name__ == '__main__':
    # 測試用例
    import ccxt
    from datetime import datetime, timedelta
    
    print('\n正在下載測試數據...')
    exchange = ccxt.binance({'enableRateLimit': True})
    klines = exchange.fetch_ohlcv('BTCUSDT', '1h', limit=500)
    
    df = pd.DataFrame(
        klines,
        columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
    )
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.rename(columns={'timestamp': 'time'})
    
    # 生成有效性標籤
    generator = ValidityLabelGenerator(
        lookahead=10,
        min_bounce_pct=0.5,
        momentum_decay_thresh=0.3
    )
    
    print('\n生成有效性標籤...')
    df_labeled = generator.generate_validity_labels(df)
    
    # 統計結果
    stats = generator.get_validity_statistics(df_labeled)
    
    print('\n✅ 有效性統計結果:')
    print(f'  總下軌觸碰數: {stats["total_support_touches"]}')
    print(f'  有效支撐數: {stats["valid_supports"]}')
    print(f'  下軌有效率: {stats["support_validity_rate"]*100:.1f}%')
    print(f'  總上軌觸碰數: {stats["total_resistance_touches"]}')
    print(f'  有效阻力數: {stats["valid_resistances"]}')
    print(f'  上軌有效率: {stats["resistance_validity_rate"]*100:.1f}%')
    print(f'  整體有效率: {stats["overall_validity_rate"]*100:.1f}%')
    
    # 顯示有效標籤
    valid_touches = df_labeled[df_labeled['touch'] != 0].copy()
    print(f'\n有效觸碰點 (前 20 個):')
    print(valid_touches[['time', 'close', 'touch', 'is_valid_support', 'is_valid_resistance']].head(20))
