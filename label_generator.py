import pandas as pd
import numpy as np
from typing import Tuple

class LabelGenerator:
    def __init__(self, period=20, std_dev=2):
        """
        標籤生成器
        - period: BB 週期
        - std_dev: 標準侯差
        """
        self.period = period
        self.std_dev = std_dev
    
    def calculate_bollinger_bands(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        計算 Bollinger Bands
        """
        df = df.copy()
        
        # 標準化列名
        close_col = 'close' if 'close' in df.columns else 'Close'
        
        # 計算中軸線（SMA）
        df['bb_middle'] = df[close_col].rolling(window=self.period).mean()
        
        # 計算標準侯差
        df['bb_std'] = df[close_col].rolling(window=self.period).std()
        
        # 計算上軌和下軌
        df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * self.std_dev)
        df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * self.std_dev)
        
        return df
    
    def calculate_volatility(self, df: pd.DataFrame, window=20) -> pd.Series:
        """
        計算波動性（會話美時間ボラティリティ）
        """
        close_col = 'close' if 'close' in df.columns else 'Close'
        returns = df[close_col].pct_change()
        volatility = returns.rolling(window=window).std() * np.sqrt(252 * 24)  # 年化
        return volatility
    
    def generate_bb_touch_labels(self, df: pd.DataFrame, touch_range=0.02) -> pd.DataFrame:
        """
        生成 BB 軌道距離標籤
        
        接近範圍：距離上軌 < touch_range 或 距離下軌 < touch_range
        
        標籤：
        - 1: 接近上軌（可能需要阻力）
        - -1: 接近下軌（可能需要支撐）
        - 0: 中軸或不接近
        
        Args:
            df: DataFrame with OHLCV data
            touch_range: 距離上/下軌的閾值（例如 0.01 = 1%, 0.02 = 2%）
        """
        df = df.copy()
        close_col = 'close' if 'close' in df.columns else 'Close'
        
        # 計算距離上軌和下軌的距離（百分比）
        # dist_to_upper = (upper - close) / upper
        # dist_to_lower = (close - lower) / lower
        df['dist_to_upper'] = (df['bb_upper'] - df[close_col]) / df['bb_upper']
        df['dist_to_lower'] = (df[close_col] - df['bb_lower']) / df['bb_lower']
        
        # 生成標籤
        df['bb_touch_label'] = 0
        df.loc[df['dist_to_upper'] < touch_range, 'bb_touch_label'] = 1   # 接近上軌
        df.loc[df['dist_to_lower'] < touch_range, 'bb_touch_label'] = -1  # 接近下軌
        
        return df
    
    def generate_volatility_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        生成波動性標籤
        
        三世級：
        - High: 波動性 > 75位數
        - Medium: 25-75位數
        - Low: 波動性 < 25位數
        """
        df = df.copy()
        
        # 計算波動性位數
        df['volatility_pct'] = df['volatility'].rank(pct=True) * 100
        
        # 生成標籤
        df['volatility_label'] = 'medium'
        df.loc[df['volatility_pct'] > 75, 'volatility_label'] = 'high'
        df.loc[df['volatility_pct'] < 25, 'volatility_label'] = 'low'
        
        # 數值標籤
        df['volatility_numeric'] = df['volatility_label'].map({'low': 0, 'medium': 1, 'high': 2})
        
        return df
    
    def generate_future_volatility(self, df: pd.DataFrame, lookahead=5) -> pd.Series:
        """
        生成未來 N 根的波動性（用於訓練目標）
        """
        close_col = 'close' if 'close' in df.columns else 'Close'
        future_returns = df[close_col].shift(-lookahead) / df[close_col] - 1
        future_volatility = future_returns.abs().rolling(window=lookahead).mean()
        return future_volatility
    
    def create_training_dataset(self, df: pd.DataFrame, lookahead=5, touch_range=0.02) -> pd.DataFrame:
        """
        合成一完整的訓練數據集
        """
        df = df.copy()
        
        # 1. BB 軌道
        df = self.calculate_bollinger_bands(df)
        
        # 2. 波動性
        df['volatility'] = self.calculate_volatility(df)
        
        # 3. BB 軌道距離標籤
        df = self.generate_bb_touch_labels(df, touch_range=touch_range)
        
        # 4. 波動性標籤
        df = self.generate_volatility_labels(df)
        
        # 5. 未來波動性（預測目標）
        df['future_volatility'] = self.generate_future_volatility(df, lookahead=lookahead)
        
        # 只保留有效訓練數據
        df = df.dropna(subset=['bb_middle', 'volatility', 'bb_touch_label', 'future_volatility'])
        
        return df


if __name__ == '__main__':
    # 測試
    import pandas as pd
    
    # 建立梨數據
    dates = pd.date_range('2024-01-01', periods=1000, freq='15min')
    data = {
        'time': dates,
        'close': np.random.randn(1000).cumsum() + 50000
    }
    df = pd.DataFrame(data)
    
    # 生成標籤
    generator = LabelGenerator(period=20, std_dev=2)
    df_labeled = generator.create_training_dataset(df, lookahead=5, touch_range=0.02)
    
    print(f'✅ 生成了 {len(df_labeled)} 根有效的訓練數據')
    print(f'\n標籤分布:')
    print(df_labeled[['bb_touch_label', 'volatility_label']].value_counts())
    print(f'\n樣本:')
    print(df_labeled[['close', 'bb_upper', 'bb_lower', 'dist_to_upper', 'dist_to_lower', 'bb_touch_label', 'volatility_label']].head(20))
