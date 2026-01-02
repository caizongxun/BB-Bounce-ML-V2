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
        
        # 為每個幣種 + timeframe 建立目錄
        self.models_base_dir = self.output_dir / 'bb_models'
        self.models_base_dir.mkdir(parents=True, exist_ok=True)
        
        self.loader = CryptoDataLoader()
        self.generator = LabelGenerator(period=20, std_dev=2)
        
        # 標籤對應
        self.label_map = {-1: 0, 0: 1, 1: 2}  # support -> 0, neutral -> 1, resistance -> 2
        self.inverse_label_map = {0: -1, 1: 0, 2: 1}
    
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
    
    def check_overfitting(self, train_acc, test_acc):
        """
        検查過據合佐（Overfitting）
        """
        gap = train_acc - test_acc
        
        print(f'\n🔍 過據合佐検查：')
        print(f'  訓練精準度: {train_acc:.4f} ({train_acc*100:.2f}%)')
        print(f'  測試精準度: {test_acc:.4f} ({test_acc*100:.2f}%)')
        print(f'  精準度寶: {gap:.4f} ({gap*100:.2f}%)')
        
        if gap < 0.01:  # 精準度寶 < 1%
            print(f'  ✅ 模型帷貌！沒有過據合佐')
            return 'good'
        elif gap < 0.05:  # 精準度寶 < 5%
            print(f'  ⚠️ 輕微過據合佐，但可以接受')
            return 'acceptable'
        elif gap < 0.10:  # 精準度寶 < 10%
            print(f'  👁 中等過據合佐，b鰧詰枣殆建議授出')
            return 'warning'
        else:  # 精準度寶 >= 10%
            print(f'  ❌ 嚴重過據合佐！議誮重新訓練')
            return 'bad'
    
    def train_single_symbol(self, symbol: str, timeframe: str, touch_range=0.02, test_size=0.2):
        """
        為單個幣種 + timeframe 訓練模型
        """
        separator = '='*60
        print(f'\n{separator}')
        print(f'🎯 訓練 {symbol} {timeframe} 模型')
        print(f'{separator}')
        
        try:
            # 1. 下載數據
            df = self.loader.download_symbol_data(symbol, timeframe)
            if df is None:
                print(f'❌ {symbol} {timeframe} 下載失敗')
                return False
            
            # 2. 產生標籤
            print(f'🔧 產生標籤...')
            df = self.generator.create_training_dataset(df, lookahead=5, touch_range=touch_range)
            
            # 3. 產產特录
            print(f'🔧 產產特录...')
            df = self.create_features(df)
            
            # 4. 選擇特录
            feature_cols = [
                'price_to_bb_middle', 'dist_upper_norm', 'dist_lower_norm',
                'bb_width', 'rsi', 'volatility', 'returns_std',
                'high_low_ratio', 'close_open_ratio',
                'sma_5', 'sma_20', 'sma_50'
            ]
            
            # 離棄或 nan 數據
            X = df[feature_cols].fillna(method='ffill').fillna(method='bfill')
            y = df['bb_touch_label']
            
            # 5. 分割訓練/測試集
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )
            
            print(f'  訓練集: {len(X_train)} 根')
            print(f'  測試集: {len(X_test)} 根')
            
            # 6. 訓練模型
            print(f'📚 訓練模型...')
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # 標籤轉換: -1 -> 0, 0 -> 1, 1 -> 2
            y_train_mapped = np.array([self.label_map[int(label)] for label in y_train])
            y_test_mapped = np.array([self.label_map[int(label)] for label in y_test])
            
            model = XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                eval_metric='mlogloss',
                verbosity=0,
                num_class=3
            )
            
            model.fit(X_train_scaled, y_train_mapped)
            
            # 7. 統計邟訓練集精準度
            y_train_pred = model.predict(X_train_scaled)
            train_acc = accuracy_score(y_train_mapped, y_train_pred)
            
            # 8. 統計測試集精準度
            y_test_pred = model.predict(X_test_scaled)
            test_acc = accuracy_score(y_test_mapped, y_test_pred)
            test_f1 = f1_score(y_test_mapped, y_test_pred, average='weighted')
            
            # 9. 検查過據合佐
            overfitting_status = self.check_overfitting(train_acc, test_acc)
            
            # 10. 轉避過據合佐，只顯示測試精準度
            print(f'\n📈 主要指標：')
            print(f'  測試精準度: {test_acc:.4f} ({test_acc*100:.2f}%)')
            print(f'  測試 F1 分數: {test_f1:.4f}')
            
            print(f'\n分類報告：')
            label_names = ['下軌支撐', '中軸中立', '上軌阻力']
            print(classification_report(y_test_mapped, y_test_pred, target_names=label_names))
            
            # 11. 保存模型
            symbol_dir = self.models_base_dir / symbol / timeframe
            symbol_dir.mkdir(parents=True, exist_ok=True)
            
            model_path = symbol_dir / 'model.pkl'
            scaler_path = symbol_dir / 'scaler.pkl'
            label_map_path = symbol_dir / 'label_map.pkl'
            
            joblib.dump(model, model_path)
            joblib.dump(scaler, scaler_path)
            joblib.dump(self.label_map, label_map_path)
            
            print(f'\n📦 模型已保存:')
            print(f'  {model_path}')
            print(f'  {scaler_path}')
            print(f'  {label_map_path}')
            
            # 如果有严重過據合佐，傳回 False 以跟蹤
            return overfitting_status != 'bad'
        
        except Exception as e:
            print(f'❌ 訓練失敗: {e}')
            import traceback
            traceback.print_exc()
            return False
    
    def run_full_pipeline(self, touch_range=0.02, test_size=0.2):
        """
        為所有幣種 + timeframe 訓練模型
        """
        print('\n🚀 開始為所有幣種訓練模型...')
        
        success_count = 0
        warning_count = 0
        total_count = len(self.loader.symbols) * len(self.loader.timeframes)
        
        for symbol in self.loader.symbols:
            for timeframe in self.loader.timeframes:
                if self.train_single_symbol(symbol, timeframe, touch_range, test_size):
                    success_count += 1
        
        separator = '='*60
        print(f'\n{separator}')
        print(f'✅ 訓練完成！成功: {success_count}/{total_count}')
        print(f'{separator}')
        print(f'模型保存位置: {self.models_base_dir}')
        print(f'結構：models/bb_models/<SYMBOL>/<TIMEFRAME>/')


if __name__ == '__main__':
    trainer = BBModelTrainer()
    trainer.run_full_pipeline(touch_range=0.02, test_size=0.2)
