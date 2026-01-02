import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from data_loader import CryptoDataLoader
from label_generator import LabelGenerator

class PredictionAnalyzer:
    def __init__(self, models_dir='models'):
        self.models_dir = Path(models_dir)
        self.bb_models_dir = self.models_dir / 'bb_models'
        self.loader = CryptoDataLoader()
        self.generator = LabelGenerator(period=20, std_dev=2)
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        從 K 線數據製作特徵
        """
        df = df.copy()
        close_col = 'close' if 'close' in df.columns else 'Close'
        
        if 'open' not in df.columns and 'Open' in df.columns:
            df['open'] = df['Open']
            df['high'] = df['High']
            df['low'] = df['Low']
        
        # 1. 價格位置（相對於 BB 中軸）
        df['price_to_bb_middle'] = (df[close_col] - df['bb_middle']) / df['bb_middle']
        
        # 2. 價格距離上/下軌
        df['dist_upper_norm'] = (df['bb_upper'] - df[close_col]) / (df['bb_upper'] - df['bb_lower'])
        df['dist_lower_norm'] = (df[close_col] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # 3. BB 寶寬（BBW: Bollinger Bands Width）
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        
        # 4. RSI
        df = self.calculate_rsi(df)
        
        # 5. 波動性
        df['volatility'] = df['volatility'].fillna(df['volatility'].mean())
        
        # 6. 價格動量（日幾何幣率）
        df['returns'] = df[close_col].pct_change()
        df['returns_std'] = df['returns'].rolling(window=20).std()
        
        # 7. 價格走勢
        df['high_low_ratio'] = df['high'] / df['low'] - 1 if 'high' in df.columns else 0
        df['close_open_ratio'] = df[close_col] / df['open'] - 1 if 'open' in df.columns else 0
        
        # 8. 移動平均
        df['sma_5'] = df[close_col].rolling(window=5).mean()
        df['sma_20'] = df[close_col].rolling(window=20).mean()
        df['sma_50'] = df[close_col].rolling(window=50).mean()
        
        # 樣本數據正證化
        df = df.ffill().bfill()
        
        return df
    
    def calculate_rsi(self, df: pd.DataFrame, period=14) -> pd.DataFrame:
        """計算 RSI (Relative Strength Index)"""
        df = df.copy()
        close_col = 'close' if 'close' in df.columns else 'Close'
        
        delta = df[close_col].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        df['rsi'] = df['rsi'].fillna(50)
        
        return df
    
    def analyze_symbol(self, symbol: str, timeframe: str):
        """
        分析特定幣種的預測信心度分布
        """
        print(f'\n{"="*60}')
        print(f'分析 {symbol} {timeframe} 模型')
        print(f'{"="*60}')
        
        try:
            # 1. 加載模型
            model_path = self.bb_models_dir / symbol / timeframe / 'model.pkl'
            scaler_path = self.bb_models_dir / symbol / timeframe / 'scaler.pkl'
            
            if not model_path.exists():
                print(f'❌ 模型不存在: {model_path}')
                return
            
            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)
            
            print(f'✅ 已加載模型')
            
            # 2. 下載數據
            df = self.loader.download_symbol_data(symbol, timeframe)
            if df is None:
                print(f'❌ 下載失敗')
                return
            
            print(f'✅ 下載 {len(df)} 根 K 棒')
            
            # 3. 產生標籤
            df = self.generator.create_training_dataset(df, lookahead=5, touch_range=0.02)
            
            # 4. 產生特徵
            df = self.create_features(df)
            
            # 5. 準備特徵
            feature_cols = [
                'price_to_bb_middle', 'dist_upper_norm', 'dist_lower_norm',
                'bb_width', 'rsi', 'volatility', 'returns_std',
                'high_low_ratio', 'close_open_ratio',
                'sma_5', 'sma_20', 'sma_50'
            ]
            
            X = df[feature_cols].ffill().bfill()
            
            # 6. 進行預測（概率）
            X_scaled = scaler.transform(X)
            probabilities = model.predict_proba(X_scaled)
            
            # 7. 分析信心度分布
            print(f'\n📊 信心度分布分析：')
            print(f'  總預測數: {len(probabilities)}')
            
            # 計算每個類別的最大概率（信心度）
            confidences = np.max(probabilities, axis=1)
            
            print(f'\n🔍 信心度統計：')
            print(f'  最小信心度: {np.min(confidences):.4f} ({np.min(confidences)*100:.2f}%)')
            print(f'  最大信心度: {np.max(confidences):.4f} ({np.max(confidences)*100:.2f}%)')
            print(f'  平均信心度: {np.mean(confidences):.4f} ({np.mean(confidences)*100:.2f}%)')
            print(f'  中位數信心度: {np.median(confidences):.4f} ({np.median(confidences)*100:.2f}%)')
            print(f'  標準差: {np.std(confidences):.4f}')
            
            # 信心度分佈百分比
            print(f'\n📈 信心度分佈：')
            bins = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 1.00]
            for i, bin_val in enumerate(bins):
                if i == 0:
                    count = np.sum(confidences < bin_val)
                    pct = count / len(confidences) * 100
                    print(f'  < {bin_val:.0%}: {count:6d} ({pct:5.1f}%)')
                else:
                    count = np.sum((confidences >= bins[i-1]) & (confidences < bin_val))
                    pct = count / len(confidences) * 100
                    print(f'  {bins[i-1]:.0%} - {bin_val:.0%}: {count:6d} ({pct:5.1f}%)')
            
            # 檢查是否有問題
            count_100_pct = np.sum(confidences >= 0.99)
            pct_100 = count_100_pct / len(confidences) * 100
            
            if pct_100 > 50:
                print(f'\n⚠️ 警告：有 {pct_100:.1f}% 的預測信心度 >= 99%')
                print(f'   這可能表示模型過度擬合或決策邊界太極端')
            elif pct_100 > 10:
                print(f'\n🔔 注意：有 {pct_100:.1f}% 的預測信心度 >= 99%')
                print(f'   應該監控模型表現')
            else:
                print(f'\n✅ 正常：信心度分布合理')
            
            # 分析各類別的預測分布
            print(f'\n📊 各類別預測分布：')
            predictions = model.predict(X_scaled)
            label_map = {0: '下軌支撐', 1: '中軸中立', 2: '上軌阻力'}
            
            for class_idx in range(3):
                count = np.sum(predictions == class_idx)
                pct = count / len(predictions) * 100
                avg_conf = np.mean(confidences[predictions == class_idx]) if count > 0 else 0
                print(f'  {label_map[class_idx]}: {count:6d} ({pct:5.1f}%) - 平均信心度: {avg_conf:.2%}')
        
        except Exception as e:
            print(f'❌ 分析失敗: {e}')
            import traceback
            traceback.print_exc()
    
    def analyze_all_symbols(self):
        """
        分析所有幣種的預測信心度
        """
        print('\n🚀 開始分析所有模型的預測信心度...')
        
        for symbol in self.loader.symbols:
            for timeframe in self.loader.timeframes:
                self.analyze_symbol(symbol, timeframe)


if __name__ == '__main__':
    analyzer = PredictionAnalyzer()
    
    # 分析單個幣種
    analyzer.analyze_symbol('BTCUSDT', '15m')
    
    # 分析所有幣種
    # analyzer.analyze_all_symbols()
