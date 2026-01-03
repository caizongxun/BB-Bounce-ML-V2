#!/usr/bin/env python3
"""
BB 反彈 V2 模型訓練器

核心理論：
- 無效反彈：BB 通道向外擴張（波動率上升）→ 市場混亂 → 反彈失敗
- 有效反彈：BB 通道向內縮小（波動率下降）→ 市場秩序恢復 → 反彈成功

新增特徵：
1. bb_width_change - BB 寬度變化率（最重要）
2. bb_width_percentile - BB 寬度相對歷史位置
3. std_change - 標準差變化率
4. upper_lower_distance_change - 上下軌靠近速度
5. width_acceleration - BB 寬度變化加速度
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score
)
from xgboost import XGBClassifier
import warnings

warnings.filterwarnings('ignore')

from data_loader import CryptoDataLoader

class BBContractionFeatureExtractor:
    """提取 BB 通道收縮相關特徵"""
    
    @staticmethod
    def calculate_bb_bands(closes, period=20, std_dev=2):
        """計算布林通道"""
        sma = np.mean(closes[-period:])
        std = np.std(closes[-period:])
        upper = sma + std_dev * std
        lower = sma - std_dev * std
        width = upper - lower
        return upper, sma, lower, width, std
    
    @staticmethod
    def create_features(df: pd.DataFrame, lookahead=5) -> pd.DataFrame:
        """
        建立特徵，重點放在 BB 通道收縮特徵
        """
        df = df.copy()
        close_col = 'close' if 'close' in df.columns else 'Close'
        
        # 計算 BB 帶狀
        bb_period = 20
        uppers = []
        middles = []
        lowers = []
        widths = []
        stds = []
        
        for i in range(len(df)):
            if i < bb_period:
                uppers.append(np.nan)
                middles.append(np.nan)
                lowers.append(np.nan)
                widths.append(np.nan)
                stds.append(np.nan)
            else:
                closes_window = df[close_col].iloc[i-bb_period:i].values
                upper, middle, lower, width, std = BBContractionFeatureExtractor.calculate_bb_bands(closes_window)
                uppers.append(upper)
                middles.append(middle)
                lowers.append(lower)
                widths.append(width)
                stds.append(std)
        
        df['bb_upper'] = uppers
        df['bb_middle'] = middles
        df['bb_lower'] = lowers
        df['bb_width'] = widths
        df['bb_std'] = stds
        
        # ========================================
        # 特徵 1：BB 寬度變化
        # ========================================
        
        # 短期寬度變化
        df['bb_width_change_1bar'] = df['bb_width'].pct_change(1)
        df['bb_width_change_2bar'] = df['bb_width'].pct_change(2)
        df['bb_width_change_3bar'] = df['bb_width'].pct_change(3)
        df['bb_width_change_5bar'] = df['bb_width'].pct_change(5)
        
        # BB 寬度在歷史中的相對位置
        df['bb_width_percentile'] = df['bb_width'].rolling(window=20).apply(
            lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min() + 1e-8),
            raw=False
        )
        
        # 標準差變化
        df['std_change'] = df['bb_std'].pct_change()
        df['std_change_3bar'] = df['bb_std'].pct_change(3)
        
        # 上下軌靠近速度
        df['bb_distance'] = df['bb_upper'] - df['bb_lower']
        df['bb_distance_change'] = df['bb_distance'].pct_change()
        
        # BB 寬度變化加速度
        df['bb_width_acceleration'] = df['bb_width_change_1bar'].diff()
        
        # ========================================
        # 其他特徵
        # ========================================
        
        # RSI
        delta = df[close_col].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-8)
        df['rsi_14'] = 100 - (100 / (1 + rs))
        
        # 價格相對 BB 位置
        df['price_bb_position'] = (df[close_col] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-8)
        
        # 成交量比
        if 'volume' in df.columns:
            df['volume_ratio'] = df['volume'] / df['volume'].rolling(window=20).mean()
        else:
            df['volume_ratio'] = 1.0
        
        # 波動率比
        df['historical_vol'] = df[close_col].pct_change().rolling(window=20).std()
        df['vol_ratio'] = df['bb_std'] / (df['bb_std'].rolling(window=40).mean() + 1e-8)
        
        # 價格動量
        df['momentum_5'] = df[close_col].pct_change(5)
        df['momentum_10'] = df[close_col].pct_change(10)
        
        # 填充 NaN
        df = df.fillna(method='bfill').fillna(method='ffill')
        
        # ========================================
        # 生成標籤（改進版）
        # ========================================
        
        df['label_bounce_valid'] = -1  # 預設為 -1（忽略）
        
        for i in range(len(df) - lookahead):
            # 條件 1：當前 K 棒觸及或接近下軌
            price_to_lower = (df[close_col].iloc[i] - df['bb_lower'].iloc[i]) / (df['bb_width'].iloc[i] + 1e-8)
            is_near_lower = price_to_lower < 0.20  # 在下軌附近 20% 內
            
            if not is_near_lower:
                continue
            
            # ========================================
            # 核心邏輯：比較前後的 BB 寬度變化
            # ========================================
            
            # 過去 3 根 K 棒的平均 BB 寬度
            past_widths = df['bb_width'].iloc[max(0, i-3):i].values
            past_avg_width = np.mean(past_widths) if len(past_widths) > 0 else 0
            
            # 接下來 lookahead 根 K 棒的平均 BB 寬度
            future_widths = df['bb_width'].iloc[i:i+lookahead].values
            future_avg_width = np.mean(future_widths) if len(future_widths) > 0 else 0
            
            if past_avg_width == 0:
                continue
            
            # BB 寬度變化比率
            width_change_ratio = (future_avg_width - past_avg_width) / past_avg_width
            
            # 接下來 lookahead 根 K 棒的價格變化
            future_prices = df[close_col].iloc[i:i+lookahead].values
            future_price_change = (future_prices[-1] - future_prices[0]) / (future_prices[0] + 1e-8)
            
            # ========================================
            # 改進的標籤邏輯（相對比較而非絕對閾值）
            # ========================================
            
            # 情況 1：BB 寬度收縮 + 價格上升 = 有效反彈（最理想）
            if width_change_ratio < -0.10 and future_price_change > 0.003:
                df.loc[i, 'label_bounce_valid'] = 1
            
            # 情況 2：BB 寬度收縮 + 價格穩定 = 有效反彈（次理想）
            elif width_change_ratio < -0.05 and -0.005 < future_price_change < 0.010:
                df.loc[i, 'label_bounce_valid'] = 1
            
            # 情況 3：BB 寬度擴張 + 價格下跌 = 無效反彈（最差）
            elif width_change_ratio > 0.15 and future_price_change < -0.005:
                df.loc[i, 'label_bounce_valid'] = 0
            
            # 情況 4：BB 寬度擴張 + 價格混亂 = 無效反彈（次差）
            elif width_change_ratio > 0.05 and future_price_change < 0.002:
                df.loc[i, 'label_bounce_valid'] = 0
            
            # 其他情況：中性，忽略
            # (保持 -1，後面會篩選掉)
        
        return df


class BBContractionModelTrainer:
    def __init__(self, output_dir='models'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.models_dir = self.output_dir / 'bb_contraction_v2_models'
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        self.loader = CryptoDataLoader()
    
    def train_single_symbol(self, symbol: str, timeframe: str):
        """
        為單個幣種 + 時間框架訓練 BB 收縮模型
        """
        separator = '=' * 70
        print(f'\n{separator}')
        print(f'訓練 {symbol} {timeframe} - BB 收縮 V2 模型')
        print(f'{separator}')
        
        try:
            # 1. 下載數據
            print(f'下載 {symbol} {timeframe} 數據...')
            df = self.loader.download_symbol_data(symbol, timeframe)
            if df is None or len(df) < 100:
                print(f'✗ {symbol} {timeframe} 數據不足')
                return False
            
            # 2. 特徵工程
            print(f'提取 BB 收縮特徵...')
            extractor = BBContractionFeatureExtractor()
            df = extractor.create_features(df, lookahead=5)
            
            # 3. 篩選有效標籤（-1 的行被丟棄）
            df_labeled = df[df['label_bounce_valid'] != -1].copy()
            
            if len(df_labeled) < 50:
                print(f'✗ 有效標籤樣本過少：{len(df_labeled)} 個')
                return False
            
            label_counts = df_labeled['label_bounce_valid'].value_counts()
            print(f'\n標籤分布：')
            print(f'  有效反彈 (1): {label_counts.get(1, 0)} 個 ({label_counts.get(1, 0)/len(df_labeled)*100:.1f}%)')
            print(f'  無效反彈 (0): {label_counts.get(0, 0)} 個 ({label_counts.get(0, 0)/len(df_labeled)*100:.1f}%)')
            
            if label_counts.get(1, 0) < 10 or label_counts.get(0, 0) < 10:
                print(f'⚠️ 某類別樣本過少，訓練可能不穩定')
            
            # 4. 選擇特徵
            feature_cols = [
                # BB 收縮特徵（最重要）
                'bb_width_change_1bar', 'bb_width_change_2bar', 'bb_width_change_3bar', 'bb_width_change_5bar',
                'bb_width_percentile', 'std_change', 'std_change_3bar',
                'bb_distance_change', 'bb_width_acceleration',
                
                # 動量和價格特徵
                'rsi_14', 'price_bb_position', 'momentum_5', 'momentum_10',
                
                # 成交量和波動率
                'volume_ratio', 'vol_ratio', 'historical_vol'
            ]
            
            X = df_labeled[feature_cols]
            y = df_labeled['label_bounce_valid']
            
            # 移除 NaN
            mask = ~X.isna().any(axis=1)
            X = X[mask]
            y = y[mask]
            
            if len(X) < 30:
                print(f'✗ 有效樣本過少：{len(X)} 個')
                return False
            
            print(f'\n特徵數：{len(feature_cols)}')
            print(f'有效樣本數：{len(X)}')
            print(f'類別比例：{y.value_counts().to_dict()}')
            
            # 5. 分割訓練/測試集
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )
            except:
                print(f'⚠️ 樣本不足以分層抽樣，使用普通分割')
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )
            
            # 6. 標準化
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # 7. 訓練模型
            print(f'\n訓練 XGBoost 分類器...')
            model = XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                eval_metric='logloss',
                verbosity=0,
                scale_pos_weight=len(y[y==0]) / len(y[y==1]) if len(y[y==1]) > 0 else 1  # 處理類別不平衡
            )
            model.fit(X_train_scaled, y_train)
            
            # 8. 評估
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            auc = roc_auc_score(y_test, y_pred_proba) if len(set(y_test)) > 1 else 0.5
            
            print(f'\n測試集性能：')
            print(f'  準確率: {accuracy:.4f} ({accuracy*100:.2f}%)')
            print(f'  精準度: {precision:.4f}')
            print(f'  召回率: {recall:.4f}')
            print(f'  F1 分數: {f1:.4f}')
            print(f'  AUC: {auc:.4f}')
            
            print(f'\n混淆矩陣：')
            print(confusion_matrix(y_test, y_pred))
            
            print(f'\n分類報告：')
            print(classification_report(y_test, y_pred, target_names=['無效反彈', '有效反彈']))
            
            # 9. 特徵重要性
            feature_importance = pd.DataFrame({
                'feature': feature_cols,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print(f'\n前 5 重要特徵：')
            for idx, row in feature_importance.head(5).iterrows():
                print(f'  {row["feature"]}: {row["importance"]:.4f}')
            
            # 10. 保存模型
            symbol_dir = self.models_dir / symbol / timeframe
            symbol_dir.mkdir(parents=True, exist_ok=True)
            
            model_path = symbol_dir / 'bb_contraction_v2_model.pkl'
            scaler_path = symbol_dir / 'bb_contraction_v2_scaler.pkl'
            features_path = symbol_dir / 'bb_contraction_v2_features.json'
            
            joblib.dump(model, model_path)
            joblib.dump(scaler, scaler_path)
            
            import json
            with open(features_path, 'w') as f:
                json.dump(feature_cols, f, indent=2)
            
            print(f'\n✅ 模型已保存：')
            print(f'  {model_path}')
            print(f'  {scaler_path}')
            
            return True
        
        except Exception as e:
            print(f'✗ 訓練失敗: {e}')
            import traceback
            traceback.print_exc()
            return False
    
    def run_full_pipeline(self):
        """
        為所有幣種和時間框架訓練
        """
        print(f'\n🚀 開始訓練 BB 收縮 V2 模型（測試版）...')
        
        success_count = 0
        total_count = len(self.loader.symbols) * len(self.loader.timeframes)
        
        for symbol in self.loader.symbols:
            for timeframe in self.loader.timeframes:
                if self.train_single_symbol(symbol, timeframe):
                    success_count += 1
        
        separator = '=' * 70
        print(f'\n{separator}')
        print(f'✅ 訓練完成！成功: {success_count}/{total_count}')
        print(f'{separator}')
        print(f'模型保存位置: {self.models_dir}')


if __name__ == '__main__':
    trainer = BBContractionModelTrainer()
    
    # 先訓練主要幣種測試
    symbols = ['BTCUSDT', 'ETHUSDT']
    timeframes = ['15m', '1h']
    
    print('開始訓練 BB 收縮 V2 模型（測試版）...')
    print(f'幣種: {symbols}')
    print(f'時間框架: {timeframes}')
    
    for symbol in symbols:
        for timeframe in timeframes:
            trainer.train_single_symbol(symbol, timeframe)
