import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from data_loader import CryptoDataLoader
from label_generator import LabelGenerator
import matplotlib.pyplot as plt

class DeepDiagnosis:
    def __init__(self, models_dir='models'):
        self.models_dir = Path(models_dir)
        self.bb_models_dir = self.models_dir / 'bb_models'
        self.loader = CryptoDataLoader()
        self.generator = LabelGenerator(period=20, std_dev=2)
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """從 K 線數據製作特徵"""
        df = df.copy()
        close_col = 'close' if 'close' in df.columns else 'Close'
        
        if 'open' not in df.columns and 'Open' in df.columns:
            df['open'] = df['Open']
            df['high'] = df['High']
            df['low'] = df['Low']
        
        df['price_to_bb_middle'] = (df[close_col] - df['bb_middle']) / df['bb_middle']
        df['dist_upper_norm'] = (df['bb_upper'] - df[close_col]) / (df['bb_upper'] - df['bb_lower'])
        df['dist_lower_norm'] = (df[close_col] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df = self.calculate_rsi(df)
        df['volatility'] = df['volatility'].fillna(df['volatility'].mean())
        df['returns'] = df[close_col].pct_change()
        df['returns_std'] = df['returns'].rolling(window=20).std()
        df['high_low_ratio'] = df['high'] / df['low'] - 1 if 'high' in df.columns else 0
        df['close_open_ratio'] = df[close_col] / df['open'] - 1 if 'open' in df.columns else 0
        df['sma_5'] = df[close_col].rolling(window=5).mean()
        df['sma_20'] = df[close_col].rolling(window=20).mean()
        df['sma_50'] = df[close_col].rolling(window=50).mean()
        df = df.ffill().bfill()
        return df
    
    def calculate_rsi(self, df: pd.DataFrame, period=14) -> pd.DataFrame:
        df = df.copy()
        close_col = 'close' if 'close' in df.columns else 'Close'
        delta = df[close_col].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        df['rsi'] = df['rsi'].fillna(50)
        return df
    
    def diagnose_symbol(self, symbol: str, timeframe: str):
        """
        深度診斷特定幣種
        """
        print(f'\n{"="*70}')
        print(f'🔬 深度診斷 {symbol} {timeframe}')
        print(f'{"="*70}')
        
        try:
            # 1. 加載模型
            model_path = self.bb_models_dir / symbol / timeframe / 'model.pkl'
            scaler_path = self.bb_models_dir / symbol / timeframe / 'scaler.pkl'
            
            if not model_path.exists():
                print(f'❌ 模型不存在')
                return
            
            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)
            
            # 2. 下載數據
            df = self.loader.download_symbol_data(symbol, timeframe)
            if df is None:
                print(f'❌ 下載失敗')
                return
            
            print(f'✅ 已加載 {len(df)} 根 K 棒')
            
            # 3. 生成標籤和特徵
            df = self.generator.create_training_dataset(df, lookahead=5, touch_range=0.02)
            df = self.create_features(df)
            
            # 4. 分割訓練/測試集
            feature_cols = [
                'price_to_bb_middle', 'dist_upper_norm', 'dist_lower_norm',
                'bb_width', 'rsi', 'volatility', 'returns_std',
                'high_low_ratio', 'close_open_ratio',
                'sma_5', 'sma_20', 'sma_50'
            ]
            
            X = df[feature_cols].ffill().bfill()
            y = df['bb_touch_label']
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # 5. 分析訓練 vs 測試數據分布
            print(f'\n📊 數據分布分析：')
            print(f'  訓練集標籤分布：')
            for label in [-1, 0, 1]:
                count = np.sum(y_train == label)
                pct = count / len(y_train) * 100
                print(f'    {label:2d}: {count:6d} ({pct:5.1f}%)')
            
            print(f'  測試集標籤分布：')
            for label in [-1, 0, 1]:
                count = np.sum(y_test == label)
                pct = count / len(y_test) * 100
                print(f'    {label:2d}: {count:6d} ({pct:5.1f}%)')
            
            # 6. 分析特徵統計
            print(f'\n📈 特徵統計（訓練集）：')
            print(f'  特徵名稱                  | 最小值   | 最大值   | 平均值   | 標準差')
            print(f'  {"─"*70}')
            for col in feature_cols:
                min_val = X_train[col].min()
                max_val = X_train[col].max()
                mean_val = X_train[col].mean()
                std_val = X_train[col].std()
                print(f'  {col:25s} | {min_val:8.4f} | {max_val:8.4f} | {mean_val:8.4f} | {std_val:8.4f}')
            
            # 7. 計算模型在訓練集的性能
            print(f'\n🎯 訓練集性能：')
            X_train_scaled = scaler.fit_transform(X_train)
            
            # 轉換標籤
            label_map = {-1: 0, 0: 1, 1: 2}
            y_train_mapped = np.array([label_map[int(label)] for label in y_train])
            
            train_proba = model.predict_proba(X_train_scaled)
            train_predictions = model.predict(X_train_scaled)
            train_confidences = np.max(train_proba, axis=1)
            
            train_accuracy = np.mean(train_predictions == y_train_mapped)
            
            print(f'  訓練集精準度: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)')
            print(f'  訓練集平均信心度: {np.mean(train_confidences):.4f} ({np.mean(train_confidences)*100:.2f}%)')
            print(f'  訓練集最小信心度: {np.min(train_confidences):.4f} ({np.min(train_confidences)*100:.2f}%)')
            print(f'  訓練集 >= 99% 的比例: {np.mean(train_confidences >= 0.99)*100:.2f}%')
            
            # 8. 計算模型在測試集的性能
            print(f'\n🔍 測試集性能：')
            X_test_scaled = scaler.transform(X_test)
            y_test_mapped = np.array([label_map[int(label)] for label in y_test])
            
            test_proba = model.predict_proba(X_test_scaled)
            test_predictions = model.predict(X_test_scaled)
            test_confidences = np.max(test_proba, axis=1)
            
            test_accuracy = np.mean(test_predictions == y_test_mapped)
            
            print(f'  測試集精準度: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)')
            print(f'  測試集平均信心度: {np.mean(test_confidences):.4f} ({np.mean(test_confidences)*100:.2f}%)')
            print(f'  測試集最小信心度: {np.min(test_confidences):.4f} ({np.min(test_confidences)*100:.2f}%)')
            print(f'  測試集 >= 99% 的比例: {np.mean(test_confidences >= 0.99)*100:.2f}%')
            
            # 9. 過擬合檢查
            print(f'\n⚠️ 過擬合檢查：')
            acc_gap = train_accuracy - test_accuracy
            conf_gap = np.mean(train_confidences) - np.mean(test_confidences)
            
            print(f'  精準度差: {acc_gap:.4f} ({acc_gap*100:.2f}%)')
            print(f'  信心度差: {conf_gap:.4f} ({conf_gap*100:.2f}%)')
            
            if acc_gap < 0.01 and conf_gap < 0.01:
                print(f'  ✅ 沒有過擬合跡象')
            elif acc_gap < 0.05:
                print(f'  ⚠️  輕微過擬合，但可接受')
            else:
                print(f'  ❌ 中等過擬合')
            
            # 10. 決策邊界分析
            print(f'\n🔬 決策邊界分析：')
            print(f'  檢查是否某個特徵主導決策...')
            
            # 計算特徵重要性
            feature_importance = model.feature_importances_
            sorted_idx = np.argsort(feature_importance)[::-1]
            
            print(f'\n  前 5 個最重要的特徵：')
            for i in range(min(5, len(feature_cols))):
                idx = sorted_idx[i]
                importance = feature_importance[idx]
                print(f'    {i+1}. {feature_cols[idx]:25s}: {importance:.4f}')
            
            top_importance_sum = np.sum(feature_importance[sorted_idx[:3]]) / np.sum(feature_importance)
            print(f'\n  前 3 個特徵佔比: {top_importance_sum*100:.1f}%')
            
            if top_importance_sum > 0.7:
                print(f'  ⚠️  警告: 模型決策過度依賴少數特徵')
            else:
                print(f'  ✅ 模型使用多個特徵進行決策')
        
        except Exception as e:
            print(f'❌ 診斷失敗: {e}')
            import traceback
            traceback.print_exc()

if __name__ == '__main__':
    diagnosis = DeepDiagnosis()
    
    # 診斷單個幣種
    diagnosis.diagnose_symbol('BTCUSDT', '15m')
    diagnosis.diagnose_symbol('ETHUSDT', '1h')
