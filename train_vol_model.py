import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, classification_report, accuracy_score
from xgboost import XGBRegressor, XGBClassifier
import warnings

warnings.filterwarnings('ignore')

from data_loader import CryptoDataLoader
from label_generator import LabelGenerator

class VolatilityModelTrainer:
    def __init__(self, output_dir='models'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.loader = CryptoDataLoader()
        self.generator = LabelGenerator(period=20, std_dev=2)
        
        self.model = None
        self.scaler = None
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        產產波動性預測特彛
        """
        df = df.copy()
        close_col = 'close' if 'close' in df.columns else 'Close'
        
        # 基礎 OHLCV
        if 'open' not in df.columns and 'Open' in df.columns:
            df['open'] = df['Open']
            df['high'] = df['High']
            df['low'] = df['Low']
        
        # 1. 當前波動性
        df['volatility'] = df['volatility'].fillna(df['volatility'].mean())
        
        # 2. 上下軌寶予 (BBW)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        
        # 3. 價格車羪率輕渥
        df['price_range'] = (df['high'] - df['low']) / df[close_col] if 'high' in df.columns else 0
        df['body_size'] = (df[close_col] - df['open']).abs() / df[close_col] if 'open' in df.columns else 0
        
        # 4. RSI 和 勘動性第 (Volume Volatility)
        df = self.calculate_rsi(df)
        df['volume_change'] = df['volume'].pct_change().rolling(window=5).std() if 'volume' in df.columns else 0
        
        # 5. 平均僅感譠地区間
        df['atr_14'] = self.calculate_atr(df, period=14)
        df['atr_ratio'] = df['atr_14'] / df[close_col]
        
        # 6. 價格路旁的谺度
        df['returns'] = df[close_col].pct_change()
        df['returns_rolling_std'] = df['returns'].rolling(window=10).std()
        df['returns_rolling_mean'] = df['returns'].rolling(window=10).mean()
        
        # 7. 歷史波動性 (Historical Volatility)
        df['hist_vol_5'] = df[close_col].pct_change().rolling(window=5).std()
        df['hist_vol_10'] = df[close_col].pct_change().rolling(window=10).std()
        df['hist_vol_20'] = df[close_col].pct_change().rolling(window=20).std()
        
        # 8. 三稀線
        df['sma_5'] = df[close_col].rolling(window=5).mean()
        df['sma_20'] = df[close_col].rolling(window=20).mean()
        df['price_to_sma'] = df[close_col] / df['sma_20']
        
        # 9. 碩 (Stochastic)
        df = self.calculate_stochastic(df)
        
        # 填仅 NaN
        df = df.fillna(method='bfill').fillna(method='ffill')
        
        return df
    
    def calculate_rsi(self, df: pd.DataFrame, period=14) -> pd.DataFrame:
        """
        計算 RSI
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
    
    def calculate_atr(self, df: pd.DataFrame, period=14) -> pd.Series:
        """
        計算 ATR (Average True Range)
        """
        close_col = 'close' if 'close' in df.columns else 'Close'
        
        high = df['high'] if 'high' in df.columns else df[close_col]
        low = df['low'] if 'low' in df.columns else df[close_col]
        
        tr1 = high - low
        tr2 = (high - df[close_col].shift()).abs()
        tr3 = (low - df[close_col].shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        return atr
    
    def calculate_stochastic(self, df: pd.DataFrame, period=14) -> pd.DataFrame:
        """
        計算併稀緑線
        """
        df = df.copy()
        
        high = df['high'] if 'high' in df.columns else df['close']
        low = df['low'] if 'low' in df.columns else df['close']
        close_col = 'close' if 'close' in df.columns else 'Close'
        
        min_low = low.rolling(window=period).min()
        max_high = high.rolling(window=period).max()
        
        df['k_percent'] = 100 * ((df[close_col] - min_low) / (max_high - min_low))
        df['d_percent'] = df['k_percent'].rolling(window=3).mean()
        
        return df
    
    def load_and_prepare_data(self, touch_range=0.02):
        """
        加載、整理訓練數據
        """
        print('🚀 開始下載整理訓練數據...')
        
        all_dfs = []
        
        for symbol in self.loader.symbols:
            try:
                print(f'  ⬇️  {symbol}...', end=' ', flush=True)
                
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
        
        if all_dfs:
            full_df = pd.concat(all_dfs, ignore_index=True)
            print(f'\n✅ 整合後: {len(full_df)} 根訓練數據')
            return full_df
        else:
            raise ValueError('沒有成功加載任何訓練數據')
    
    def train_regression(self, X_train, y_train, X_test=None, y_test=None):
        """
        訓練回歸模式（預測未來波動性數值）
        """
        print(f'\n📚 訓練波動性回歸預測模型...')
        
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        self.model = XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=0
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        if X_test is not None and y_test is not None:
            X_test_scaled = self.scaler.transform(X_test)
            y_pred = self.model.predict(X_test_scaled)
            
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            print(f'  MSE: {mse:.6f}')
            print(f'  RMSE: {rmse:.6f}')
            print(f'  MAE: {mae:.6f}')
            print(f'  R²: {r2:.4f}')
    
    def train_classification(self, X_train, y_train, X_test=None, y_test=None):
        """
        訓練分類模式（低/中/高波動性）
        """
        print(f'\n📚 訓練波動性分類模型...')
        
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        
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
        
        if X_test is not None and y_test is not None:
            X_test_scaled = self.scaler.transform(X_test)
            y_pred = self.model.predict(X_test_scaled)
            
            acc = accuracy_score(y_test, y_pred)
            
            print(f'  上作: {acc:.4f}')
            print(f'\n箕科頖戶號農象：')
            print(classification_report(y_test, y_pred, target_names=['低波', '中波', '高波']))
    
    def save_model(self, model_suffix='regression'):
        """
        保存模式
        """
        model_path = self.output_dir / f'vol_model_{model_suffix}.pkl'
        scaler_path = self.output_dir / f'vol_scaler_{model_suffix}.pkl'
        
        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)
        
        print(f'\n💾 模式已保存:')
        print(f'  {model_path}')
        print(f'  {scaler_path}')
    
    def run_full_pipeline(self, touch_range=0.02, test_size=0.2, model_type='regression'):
        """
        執行完整訓練流程
        
        model_type: 'regression' (預測波動性) 或 'classification' (分類低/中/高)
        """
        # 1. 加載整理数据
        df = self.load_and_prepare_data(touch_range=touch_range)
        
        # 2. 產產特彛
        print(f'\n🔧 產產特彔...')
        df = self.create_features(df)
        
        # 3. 捲選特彛
        feature_cols = [
            'volatility', 'bb_width', 'price_range', 'body_size',
            'rsi', 'volume_change', 'atr_ratio',
            'returns_rolling_std', 'returns_rolling_mean',
            'hist_vol_5', 'hist_vol_10', 'hist_vol_20',
            'price_to_sma', 'k_percent', 'd_percent'
        ]
        
        X = df[feature_cols].fillna(method='ffill').fillna(method='bfill')
        
        if model_type == 'regression':
            y = df['future_volatility']
            y = y[y.notna()]
            X = X.loc[y.index]
        else:  # classification
            y = df['volatility_numeric']
        
        # 4. 分隔訓練/測試集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        print(f'  訓練集: {len(X_train)} 根')
        print(f'  測試集: {len(X_test)} 根')
        
        # 5. 訓練模式
        if model_type == 'regression':
            self.train_regression(X_train.values, y_train.values, X_test.values, y_test.values)
            self.save_model('regression')
        else:
            self.train_classification(X_train.values, y_train.values, X_test.values, y_test.values)
            self.save_model('classification')
        
        print(f'\n✅ 訓練完成！')


if __name__ == '__main__':
    # 訓練回歸模式（預測波動性數值）
    trainer = VolatilityModelTrainer()
    trainer.run_full_pipeline(touch_range=0.02, test_size=0.2, model_type='regression')
    
    # 訓練分類模式（低/中/高）
    # trainer2 = VolatilityModelTrainer()
    # trainer2.run_full_pipeline(touch_range=0.02, test_size=0.2, model_type='classification')
