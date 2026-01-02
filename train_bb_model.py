import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from xgboost import XGBClassifier
import warnings

warnings.filterwarnings('ignore')

from data_loader import CryptoDataLoader
from label_generator import LabelGenerator

class BBModelTrainer:
    def __init__(self, output_dir='models'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.loader = CryptoDataLoader()
        self.generator = LabelGenerator(period=20, std_dev=2)
        
        self.model = None
        self.scaler = None
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        從 K 線數據製作特录
        """
        df = df.copy()
        close_col = 'close' if 'close' in df.columns else 'Close'
        
        # 基礎 OHLCV
        if 'open' not in df.columns and 'Open' in df.columns:
            df['open'] = df['Open']
            df['high'] = df['High']
            df['low'] = df['Low']
        
        # 1. 價格位置（相對於 BB 中軸）
        df['price_to_bb_middle'] = (df[close_col] - df['bb_middle']) / df['bb_middle']
        
        # 2. 價格距離上/下軌
        df['dist_upper_norm'] = (df['bb_upper'] - df[close_col]) / (df['bb_upper'] - df['bb_lower'])
        df['dist_lower_norm'] = (df[close_col] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # 3. BB 寶予（BBW: Bollinger Bands Width）
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        
        # 4. RSI
        df = self.calculate_rsi(df)
        
        # 5. 勘動性
        df['volatility'] = df['volatility'].fillna(df['volatility'].mean())
        
        # 6. 價格動埸（日幾何幣率）
        df['returns'] = df[close_col].pct_change()
        df['returns_std'] = df['returns'].rolling(window=20).std()
        
        # 7. 價格跑勢
        df['high_low_ratio'] = df['high'] / df['low'] - 1 if 'high' in df.columns else 0
        df['close_open_ratio'] = df[close_col] / df['open'] - 1 if 'open' in df.columns else 0
        
        # 8. 移動平均
        df['sma_5'] = df[close_col].rolling(window=5).mean()
        df['sma_20'] = df[close_col].rolling(window=20).mean()
        df['sma_50'] = df[close_col].rolling(window=50).mean()
        
        # 樣子據沙正証化
        df = df.fillna(method='bfill').fillna(method='ffill')
        
        return df
    
    def calculate_rsi(self, df: pd.DataFrame, period=14) -> pd.DataFrame:
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
    
    def load_and_prepare_data(self, touch_range=0.02):
        """
        載入整理整個訓練數據集
        """
        print('🚀 開始下載 22 種幣種的整個訓練數據駕...')
        
        all_dfs = []
        
        for symbol in self.loader.symbols:
            try:
                print(f'  ⬇️  {symbol}...', end=' ', flush=True)
                
                # 下載所有 timeframe
                symbol_dfs = []
                for tf in self.loader.timeframes:
                    df = self.loader.download_symbol_data(symbol, tf)
                    if df is not None:
                        # 產生標籤
                        df = self.generator.create_training_dataset(df, lookahead=5, touch_range=touch_range)
                        df['symbol'] = symbol
                        df['timeframe'] = tf
                        symbol_dfs.append(df)
                
                if symbol_dfs:
                    combined = pd.concat(symbol_dfs, ignore_index=True)
                    all_dfs.append(combined)
                    print(f'✅ {len(combined)} 根')
                else:
                    print(f'❌')
            
            except Exception as e:
                print(f'❌ {e}')
        
        # 整合所有訓練數據
        if all_dfs:
            full_df = pd.concat(all_dfs, ignore_index=True)
            print(f'\n✅ 整合後: {len(full_df)} 根訓練數據')
            return full_df
        else:
            raise ValueError('沒有成功加載任何訓練數據')
    
    def train(self, X_train, y_train, X_test=None, y_test=None):
        """
        訓練 XGBClassifier
        """
        print(f'\n📚 訓練 BB 標籤分類器...')
        
        # 新延伸化
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # 訓練模式
        self.model = XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='mlogloss',
            verbosity=0
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        # 驗證
        if X_test is not None and y_test is not None:
            X_test_scaled = self.scaler.transform(X_test)
            y_pred = self.model.predict(X_test_scaled)
            
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            print(f'  上作: {acc:.4f}')
            print(f'  F1 分數: {f1:.4f}')
            print(f'\n箕科頖戶號農象：')
            print(classification_report(y_test, y_pred, target_names=['下軌', '中間', '上軌']))
    
    def save_model(self):
        """
        保存模式
        """
        model_path = self.output_dir / 'bb_model.pkl'
        scaler_path = self.output_dir / 'bb_scaler.pkl'
        
        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)
        
        print(f'\n💾 模式已保存:')
        print(f'  {model_path}')
        print(f'  {scaler_path}')
    
    def run_full_pipeline(self, touch_range=0.02, test_size=0.2):
        """
        執行完整訓練流程
        """
        # 1. 加載整理數據
        df = self.load_and_prepare_data(touch_range=touch_range)
        
        # 2. 產產特彛
        print(f'\n🔧 產產特录...')
        df = self.create_features(df)
        
        # 3. 捲選特彛
        feature_cols = [
            'price_to_bb_middle', 'dist_upper_norm', 'dist_lower_norm',
            'bb_width', 'rsi', 'volatility', 'returns_std',
            'high_low_ratio', 'close_open_ratio',
            'sma_5', 'sma_20', 'sma_50'
        ]
        
        # 离鸓或 nan 數據
        X = df[feature_cols].fillna(method='ffill').fillna(method='bfill')
        y = df['bb_touch_label']
        
        # 4. 傳分訓練/測試集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        print(f'  訓練集: {len(X_train)} 根')
        print(f'  測試集: {len(X_test)} 根')
        
        # 5. 訓練模式
        self.train(X_train.values, y_train.values, X_test.values, y_test.values)
        
        # 6. 保存模式
        self.save_model()
        
        print(f'\n✅ 訓練完成！')


if __name__ == '__main__':
    trainer = BBModelTrainer()
    trainer.run_full_pipeline(touch_range=0.02, test_size=0.2)
