#!/usr/bin/env python3
"""
BB 反彈 V2 模型訓練器 - 優化版

改進點：
1. 標籤邏輯優化：更嚴格的反彈條件 + 回撤限制
2. 新增特徵：BB 寬度趨勢 + 成交量強度 + 動量融合 + 擠壓評分
3. 超參調整：更深的樹 + 更低的學習率 + 正則化
4. 全幣種訓練：22 個幣種 × 2 個時框 = 44 個模型
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
import logging
from datetime import datetime

warnings.filterwarnings('ignore')

# 日誌設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    from data_loader import CryptoDataLoader
except ImportError:
    logger.error('找不到 data_loader，請確保在正確的目錄')
    exit(1)

# ============================================================
# 優化的特徵提取器
# ============================================================

class BBContractionFeatureExtractorV3:
    """改進版特徵提取 - 加入新的 BB 收縮指標"""
    
    @staticmethod
    def calculate_bb_bands(closes, period=20, std_dev=2):
        """計算 BB 帶"""
        if len(closes) < period:
            return None, None, None, None, None
        
        sma = np.mean(closes[-period:])
        std = np.std(closes[-period:])
        upper = sma + std_dev * std
        lower = sma - std_dev * std
        width = upper - lower
        return upper, sma, lower, width, std
    
    @staticmethod
    def create_features(df: pd.DataFrame, timeframe: str, lookahead=5) -> pd.DataFrame:
        """
        建立特徵並生成標籤
        
        Args:
            df: 原始 OHLCV 數據
            timeframe: '15m' 或 '1h'
            lookahead: 向前看的 K 棒數
        """
        df = df.copy()
        close_col = 'close' if 'close' in df.columns else 'Close'
        
        # ========================================
        # 第 1 步：計算 Bollinger Bands
        # ========================================
        
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
                result = BBContractionFeatureExtractorV3.calculate_bb_bands(closes_window)
                if result[0] is not None:
                    upper, middle, lower, width, std = result
                    uppers.append(upper)
                    middles.append(middle)
                    lowers.append(lower)
                    widths.append(width)
                    stds.append(std)
                else:
                    uppers.append(np.nan)
                    middles.append(np.nan)
                    lowers.append(np.nan)
                    widths.append(np.nan)
                    stds.append(np.nan)
        
        df['bb_upper'] = uppers
        df['bb_middle'] = middles
        df['bb_lower'] = lowers
        df['bb_width'] = widths
        df['bb_std'] = stds
        
        # ========================================
        # 第 2 步：計算 BB 寬度變化特徵
        # ========================================
        
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
        
        # 上下軌距離變化
        df['bb_distance'] = df['bb_upper'] - df['bb_lower']
        df['bb_distance_change'] = df['bb_distance'].pct_change()
        
        # BB 寬度加速度
        df['bb_width_acceleration'] = df['bb_width_change_1bar'].diff()
        
        # ========================================
        # 第 3 步：新增特徵 - BB 收縮指標
        # ========================================
        
        # 1. BB 寬度趨勢 (10 根 K 棒線性回歸斜率)
        df['bb_width_trend'] = df['bb_width'].rolling(window=10).apply(
            lambda x: np.polyfit(np.arange(len(x)), x, 1)[0] if len(x) > 1 else 0,
            raw=False
        )
        
        # 2. BB 擠壓分數 (寬度百分位 - 寬度變化幅度)
        df['bb_squeeze_score'] = df['bb_width_percentile'] - df['bb_width_change_2bar'].abs()
        
        # 3. 成交量強度
        if 'volume' in df.columns:
            df['volume_strength'] = df['volume'] / df['volume'].rolling(window=30).mean()
        else:
            df['volume_strength'] = 1.0
        
        # 4. 動量融合 (RSI 正規化 + 價格動量)
        delta = df[close_col].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-8)
        df['rsi_14'] = 100 - (100 / (1 + rs))
        
        rsi_normalized = (df['rsi_14'] - 50) / 50
        df['momentum_5'] = df[close_col].pct_change(5)
        df['momentum_10'] = df[close_col].pct_change(10)
        df['momentum_confluence'] = (rsi_normalized + df['momentum_5'] * 100) / 2
        
        # ========================================
        # 第 4 步：其他特徵
        # ========================================
        
        # 價格相對 BB 位置
        df['price_bb_position'] = (df[close_col] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-8)
        
        # 波動率比
        df['historical_vol'] = df[close_col].pct_change().rolling(window=20).std()
        df['vol_ratio'] = df['bb_std'] / (df['bb_std'].rolling(window=40).mean() + 1e-8)
        
        if 'volume' in df.columns:
            df['volume_ratio'] = df['volume'] / df['volume'].rolling(window=20).mean()
        else:
            df['volume_ratio'] = 1.0
        
        # 填充 NaN
        df = df.fillna(method='bfill').fillna(method='ffill')
        
        # ========================================
        # 第 5 步：生成優化的標籤
        # ========================================
        
        df['label_bounce_valid'] = -1  # 預設為忽略
        
        # 根據時框設定不同的反彈閾值
        if timeframe == '15m':
            min_rebound = 0.008  # 0.8%
            max_drawdown = -0.005  # -0.5%
        else:  # 1h
            min_rebound = 0.015  # 1.5%
            max_drawdown = -0.010  # -1%
        
        for i in range(len(df) - lookahead):
            # 條件 1：價格接近下軌
            if pd.isna(df['bb_lower'].iloc[i]) or pd.isna(df['bb_upper'].iloc[i]):
                continue
            
            price_to_lower = (df[close_col].iloc[i] - df['bb_lower'].iloc[i]) / (df['bb_width'].iloc[i] + 1e-8)
            is_near_lower = price_to_lower < 0.20
            
            if not is_near_lower:
                continue
            
            # 計算未來 lookahead 根 K 棒的統計
            future_closes = df[close_col].iloc[i:i+lookahead].values
            future_widths = df['bb_width'].iloc[i:i+lookahead].values
            
            if len(future_closes) < lookahead or len(future_widths) < lookahead:
                continue
            
            # 過去 3 根 K 棒的平均寬度
            past_widths = df['bb_width'].iloc[max(0, i-3):i].values
            if len(past_widths) == 0:
                continue
            
            past_avg_width = np.mean(past_widths)
            future_avg_width = np.mean(future_widths)
            
            if past_avg_width <= 0:
                continue
            
            # 條件 2：計算 BB 寬度變化
            width_change_ratio = (future_avg_width - past_avg_width) / past_avg_width
            
            # 條件 3：計算價格變化和最大回撤
            future_price_change = (future_closes[-1] - future_closes[0]) / (future_closes[0] + 1e-8)
            max_dd = (np.min(future_closes) / future_closes[0] - 1) if future_closes[0] > 0 else 0
            
            # 條件 4：計算標準差變化
            past_std = df['bb_std'].iloc[max(0, i-3):i].mean() if i >= 3 else df['bb_std'].iloc[i]
            future_std = df['bb_std'].iloc[i:i+lookahead].mean()
            std_change = (future_std - past_std) / (past_std + 1e-8)
            
            # 條件 5：成交量確認
            current_volume = df['volume'].iloc[i] if 'volume' in df.columns else 1
            avg_volume = df['volume'].iloc[max(0, i-20):i].mean() if i >= 20 and 'volume' in df.columns else 1
            volume_ratio = current_volume / (avg_volume + 1e-8)
            
            # ========================================
            # 標籤決策邏輯（優化版）
            # ========================================
            
            # 【標籤 1】強有效反彈
            if (width_change_ratio < -0.10 and  # BB 明顯收縮 >= 10%
                future_price_change > min_rebound and  # 反彈達到最低要求
                max_dd > max_drawdown and  # 沒有大幅回撤
                std_change < 0 and  # 波動率下降
                volume_ratio > 0.8):  # 可能有成交量進場
                df.loc[i, 'label_bounce_valid'] = 1
            
            # 【標籤 1】中等有效反彈
            elif (width_change_ratio < -0.05 and  # BB 收縮 >= 5%
                  future_price_change > min_rebound * 0.7 and  # 反彈達到 70% 的最低要求
                  max_dd > max_drawdown * 0.5 and  # 回撤控制在 50% 以內
                  std_change < 0.05):  # 波動率穩定
                df.loc[i, 'label_bounce_valid'] = 1
            
            # 【標籤 1】BB 寬度在歷史低位
            elif (df['bb_width_percentile'].iloc[i] < 0.25 and  # 寬度在最低 25%
                  is_near_lower and
                  future_price_change > min_rebound * 0.5):  # 只要有反彈跡象
                df.loc[i, 'label_bounce_valid'] = 1
            
            # 【標籤 0】明顯無效反彈
            elif (width_change_ratio > 0.15 and  # BB 明顯擴張 >= 15%
                  future_price_change < -0.002 and  # 沒反彈反而下跌
                  std_change > 0.1):  # 波動率上升
                df.loc[i, 'label_bounce_valid'] = 0
            
            # 【標籤 0】次等無效反彈
            elif (width_change_ratio > 0.05 and  # BB 持續擴張 >= 5%
                  future_price_change < 0.001):  # 幾乎沒反彈
                df.loc[i, 'label_bounce_valid'] = 0
            
            # 其他情況：保持 -1（忽略）
        
        return df


class BBContractionModelTrainerV2:
    """優化版訓練器"""
    
    def __init__(self, output_dir='models'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.models_dir = self.output_dir / 'bb_contraction_v2_models'
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        self.loader = CryptoDataLoader()
        
        # 統計數據
        self.stats = {
            'total': 0,
            'success': 0,
            'failed': 0,
            'total_models': 0
        }
    
    def train_single_symbol(self, symbol: str, timeframe: str):
        """
        訓練單一幣種 + 時框的模型
        """
        separator = '=' * 80
        print(f'\n{separator}')
        print(f'訓練 {symbol} {timeframe} - BB 收縮 V2 優化模型')
        print(f'{separator}')
        
        try:
            self.stats['total_models'] += 1
            
            # 1. 下載數據
            print(f'⬇️  下載 {symbol} {timeframe} 數據...')
            df = self.loader.download_symbol_data(symbol, timeframe)
            if df is None or len(df) < 100:
                print(f'✗ {symbol} {timeframe} 數據不足')
                self.stats['failed'] += 1
                return False
            
            print(f'✅ {symbol} {timeframe}: {len(df)} 根 K 棒')
            
            # 2. 提取特徵
            print(f'🔧 提取特徵...')
            extractor = BBContractionFeatureExtractorV3()
            df = extractor.create_features(df, timeframe=timeframe, lookahead=5)
            
            # 3. 篩選有效標籤
            df_labeled = df[df['label_bounce_valid'] != -1].copy()
            
            label_counts = df_labeled['label_bounce_valid'].value_counts()
            print(f'\n📊 標籤分布：')
            print(f'  有效反彈 (1): {label_counts.get(1, 0):,} 個 ({label_counts.get(1, 0)/len(df_labeled)*100:.1f}%)')
            print(f'  無效反彈 (0): {label_counts.get(0, 0):,} 個 ({label_counts.get(0, 0)/len(df_labeled)*100:.1f}%)')
            
            if len(df_labeled) < 50:
                print(f'✗ 有效樣本過少：{len(df_labeled)} 個')
                self.stats['failed'] += 1
                return False
            
            if label_counts.get(1, 0) < 5 or label_counts.get(0, 0) < 5:
                print(f'✗ 某類別樣本過少')
                self.stats['failed'] += 1
                return False
            
            # 4. 選擇特徵
            feature_cols = [
                'bb_width_change_1bar', 'bb_width_change_2bar', 'bb_width_change_3bar', 'bb_width_change_5bar',
                'bb_width_percentile', 'std_change', 'std_change_3bar',
                'bb_distance_change', 'bb_width_acceleration',
                'bb_width_trend', 'bb_squeeze_score',  # 新特徵
                'rsi_14', 'price_bb_position', 'momentum_5', 'momentum_10', 'momentum_confluence',  # 新特徵
                'volume_ratio', 'volume_strength', 'vol_ratio', 'historical_vol'  # 新特徵
            ]
            
            X = df_labeled[feature_cols]
            y = df_labeled['label_bounce_valid']
            
            # 移除 NaN
            mask = ~X.isna().any(axis=1)
            X = X[mask]
            y = y[mask]
            
            if len(X) < 30:
                print(f'✗ 有效樣本太少：{len(X)} 個')
                self.stats['failed'] += 1
                return False
            
            print(f'\n📈 特徵數：{len(feature_cols)}')
            print(f'📈 有效樣本：{len(X)}')
            
            # 5. 分割訓練/測試集
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )
            except:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )
            
            # 6. 標準化
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # 7. 訓練模型（優化超參）
            print(f'\n🤖 訓練 XGBoost 分類器 (優化超參)...')
            model = XGBClassifier(
                n_estimators=250,
                max_depth=7,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.7,
                random_state=42,
                eval_metric='logloss',
                verbosity=0,
                reg_alpha=1.0,
                reg_lambda=2.0,
                scale_pos_weight=len(y[y==0]) / (len(y[y==1]) + 1e-8) if len(y[y==1]) > 0 else 1
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
            
            print(f'\n📊 測試集性能：')
            print(f'  準確率: {accuracy:.4f} ({accuracy*100:.2f}%)')
            print(f'  精準度: {precision:.4f}')
            print(f'  召回率: {recall:.4f}')
            print(f'  F1 分數: {f1:.4f}')
            print(f'  AUC: {auc:.4f}')
            
            print(f'\n🎯 混淆矩陣：')
            cm = confusion_matrix(y_test, y_pred)
            print(f'  {cm}')
            
            # 特徵重要性
            feature_importance = pd.DataFrame({
                'feature': feature_cols,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print(f'\n⭐ 前 8 重要特徵：')
            for idx, row in feature_importance.head(8).iterrows():
                print(f'  {row["feature"]}: {row["importance"]:.4f}')
            
            # 9. 保存模型
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
            
            self.stats['success'] += 1
            return True
        
        except Exception as e:
            print(f'✗ 訓練失敗: {e}')
            import traceback
            traceback.print_exc()
            self.stats['failed'] += 1
            return False
    
    def run_full_pipeline(self):
        """
        訓練所有幣種 × 時框
        """
        print(f'\n🚀 開始訓練所有幣種的 BB 收縮 V2 優化模型...')
        print(f'🎯 目標：{len(self.loader.symbols)} 個幣種 × {len(self.loader.timeframes)} 個時框 = {len(self.loader.symbols) * len(self.loader.timeframes)} 個模型')
        print(f'\n幣種: {self.loader.symbols}')
        print(f'時框: {self.loader.timeframes}')
        
        start_time = datetime.now()
        
        for idx, symbol in enumerate(self.loader.symbols, 1):
            print(f'\n\n[進度] {idx}/{len(self.loader.symbols)} - {symbol}')
            
            for timeframe in self.loader.timeframes:
                self.train_single_symbol(symbol, timeframe)
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        # 最終統計
        separator = '=' * 80
        print(f'\n{separator}')
        print(f'🎉 訓練完成！')
        print(f'{separator}')
        print(f'成功: {self.stats["success"]}/{self.stats["total_models"]}')
        print(f'失敗: {self.stats["failed"]}/{self.stats["total_models"]}')
        print(f'耗時: {duration}')
        print(f'模型保存位置: {self.models_dir}')
        print(f'{separator}')


if __name__ == '__main__':
    trainer = BBContractionModelTrainerV2()
    trainer.run_full_pipeline()
