#!/usr/bin/env python3
"""
自動超參数調整器

澄洅辛苦的网格搜索，為每个币种找到最优的超參数配置

使用 Optuna 或普通 grid search
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import f1_score, precision_score, recall_score
from xgboost import XGBClassifier
import warnings
import logging
from datetime import datetime
import json

warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    logger.warning('Optuna 未安裝，改用 Grid Search')

try:
    from data_loader import CryptoDataLoader
except ImportError:
    logger.error('找不到 data_loader，請確保在正確的目錄')
    exit(1)

from train_bb_band_contraction_model_v2_optimized import BBContractionFeatureExtractorV3

# ============================================================
# 超參数調整器
# ============================================================

class HyperparameterTuner:
    """為模型找最优的超參数"""
    
    def __init__(self, output_dir='hyperparameter_tuning'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.loader = CryptoDataLoader()
        self.results = {}
    
    def prepare_data(self, symbol: str, timeframe: str):
        """渆敶訓練數據"""
        logger.info(f'渆敶 {symbol} {timeframe} 數據...')
        
        # 下載数据
        df = self.loader.download_symbol_data(symbol, timeframe)
        if df is None or len(df) < 100:
            return None, None, None, None
        
        # 提取特征
        extractor = BBContractionFeatureExtractorV3()
        df = extractor.create_features(df, timeframe=timeframe, lookahead=5)
        
        # 篩選有效標籤
        df_labeled = df[df['label_bounce_valid'] != -1].copy()
        
        if len(df_labeled) < 50:
            return None, None, None, None
        
        # 特征選擇
        feature_cols = [
            'bb_width_change_1bar', 'bb_width_change_2bar', 'bb_width_change_3bar', 'bb_width_change_5bar',
            'bb_width_percentile', 'std_change', 'std_change_3bar',
            'bb_distance_change', 'bb_width_acceleration',
            'bb_width_trend', 'bb_squeeze_score',
            'rsi_14', 'price_bb_position', 'momentum_5', 'momentum_10', 'momentum_confluence',
            'volume_ratio', 'volume_strength', 'vol_ratio', 'historical_vol'
        ]
        
        X = df_labeled[feature_cols]
        y = df_labeled['label_bounce_valid']
        
        # 移除 NaN
        mask = ~X.isna().any(axis=1)
        X = X[mask]
        y = y[mask]
        
        # ========================================
        # 新增：清理無穷大值和 NaN
        # ========================================
        
        # 替換無穷大
        X = X.replace([np.inf, -np.inf], np.nan)
        
        # 替換超大值 (> 1e10)
        X = X.clip(-1e10, 1e10)
        
        # 處理亮照的 NaN
        X = X.fillna(X.median())
        
        # 最侌情況：如果還有 NaN，就用 0 填充
        X = X.fillna(0)
        
        # 會檢查是否還有 NaN 或一日無穷大
        if X.isna().any().any() or np.isinf(X).any().any():
            logger.warning(f'{symbol} {timeframe}: 數據中享有 NaN 或一日無穷大，跳過')
            return None, None, None, None
        
        if len(X) < 30:
            return None, None, None, None
        
        logger.info(f'有效樣本: {len(X)}')
        
        return X, y, feature_cols, df
    
    def objective_optuna(self, trial, X_train, y_train, X_test, y_test):
        """用於 Optuna 的目標函數"""
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 5, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 5.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 5.0),
        }
        
        model = XGBClassifier(
            **params,
            random_state=42,
            eval_metric='logloss',
            verbosity=0,
            scale_pos_weight=len(y_train[y_train==0]) / (len(y_train[y_train==1]) + 1e-8)
        )
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # 优化目標：精准度 + 召回率
        # (交易上最重要的是精准度)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        
        # 权重：70% 精准度 + 30% 召回率
        score = 0.7 * precision + 0.3 * recall
        
        return score
    
    def tune_optuna(self, symbol: str, timeframe: str, n_trials=50):
        """使用 Optuna 進行超參数調整"""
        logger.info(f'使用 Optuna 檢查超參数...')
        
        # 渆整數据
        X, y, feature_cols, df = self.prepare_data(symbol, timeframe)
        if X is None:
            logger.error(f'{symbol} {timeframe} 数据不足')
            return None
        
        # 分割数据
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
        except:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
        
        # 標準化
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # 所有平後（为了 Optuna 使用）
        X_train = X_train_scaled
        X_test = X_test_scaled
        
        # 初始化 Optuna study
        sampler = optuna.samplers.TPESampler(seed=42)
        study = optuna.create_study(sampler=sampler, direction='maximize')
        
        # 定义目標函数
        def objective(trial):
            return self.objective_optuna(trial, X_train, y_train, X_test, y_test)
        
        # 運行發現
        logger.info(f'運行 {n_trials} 次試驗...')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        # 取最优结果
        best_trial = study.best_trial
        best_params = best_trial.params
        best_score = best_trial.value
        
        logger.info(f'最优超參数：{best_params}')
        logger.info(f'最优分数：{best_score:.4f}')
        
        return best_params, best_score, study
    
    def tune_grid_search(self, symbol: str, timeframe: str):
        """使用 Grid Search 進行超參数調整 (当 Optuna 不可用時)"""
        logger.info(f'使用 Grid Search 檢查超參数...')
        
        # 渆整数据
        X, y, feature_cols, df = self.prepare_data(symbol, timeframe)
        if X is None:
            logger.error(f'{symbol} {timeframe} 数据不足')
            return None
        
        # 分割数据
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
        except:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
        
        # 標準化
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # 定义网格
        param_grid = {
            'n_estimators': [100, 150, 200, 250],
            'max_depth': [5, 6, 7, 8, 9],
            'learning_rate': [0.03, 0.05, 0.08, 0.1],
            'subsample': [0.7, 0.8, 0.9],
            'colsample_bytree': [0.6, 0.7, 0.8, 0.9],
            'reg_alpha': [0, 0.5, 1.0, 2.0],
            'reg_lambda': [1.0, 1.5, 2.0, 2.5],
        }
        
        best_score = -1
        best_params = None
        trial_count = 0
        
        # 简化网格（只测试最重要的參数）
        logger.info('简化 Grid Search（只测试关键參数）...')
        
        simplified_grid = {
            'n_estimators': [100, 200, 250],
            'max_depth': [6, 7, 8],
            'learning_rate': [0.05, 0.08],
            'reg_alpha': [0.5, 1.0],
            'reg_lambda': [1.0, 2.0],
        }
        
        from itertools import product
        
        for params_tuple in product(*simplified_grid.values()):
            trial_count += 1
            params = dict(zip(simplified_grid.keys(), params_tuple))
            
            # 設置默认值
            params['subsample'] = 0.8
            params['colsample_bytree'] = 0.7
            
            model = XGBClassifier(
                **params,
                random_state=42,
                eval_metric='logloss',
                verbosity=0,
                scale_pos_weight=len(y_train[y_train==0]) / (len(y_train[y_train==1]) + 1e-8)
            )
            
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            score = 0.7 * precision + 0.3 * recall
            
            logger.info(f'[{trial_count}] n_est={params["n_estimators"]}, depth={params["max_depth"]}, lr={params["learning_rate"]:.2f} => 分数={score:.4f}')
            
            if score > best_score:
                best_score = score
                best_params = params.copy()
        
        logger.info(f'最优超參数：{best_params}')
        logger.info(f'最优分数：{best_score:.4f}')
        
        return best_params, best_score, None
    
    def save_best_params(self, symbol: str, timeframe: str, params: dict, score: float):
        """保存最优超參数"""
        results_file = self.output_dir / f'{symbol}_{timeframe}_best_params.json'
        
        data = {
            'symbol': symbol,
            'timeframe': timeframe,
            'best_params': params,
            'best_score': score,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(results_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f'结果保存至: {results_file}')
        
        return results_file
    
    def run_tuning(self, symbols=None, timeframes=None):
        """
        运行超參数調整
        """
        if symbols is None:
            symbols = self.loader.symbols
        if timeframes is None:
            timeframes = self.loader.timeframes
        
        logger.info(f'🚀 開始超參数調整...')
        logger.info(f'文件下載位置: {self.output_dir}')
        
        for symbol in symbols:
            for timeframe in timeframes:
                logger.info(f'\n⬇️ 調整 {symbol} {timeframe}...')
                
                try:
                    if OPTUNA_AVAILABLE:
                        best_params, best_score, study = self.tune_optuna(symbol, timeframe, n_trials=30)
                    else:
                        best_params, best_score, study = self.tune_grid_search(symbol, timeframe)
                    
                    if best_params:
                        self.save_best_params(symbol, timeframe, best_params, best_score)
                        self.results[f'{symbol}_{timeframe}'] = {
                            'params': best_params,
                            'score': best_score
                        }
                    else:
                        logger.warning(f'{symbol} {timeframe} 調整失敗')
                
                except Exception as e:
                    logger.error(f'{symbol} {timeframe} 错误: {e}')
                    import traceback
                    traceback.print_exc()
        
        # 最终统计
        logger.info(f'\n\u2705 超參数調整完成！')
        logger.info(f'成功: {len(self.results)}/{len(symbols) * len(timeframes)}')
        logger.info(f'结果保存于: {self.output_dir}')


if __name__ == '__main__':
    import sys
    
    tuner = HyperparameterTuner()
    
    # 仅测试 BTC 和 ETH (快速模式)
    symbols = ['BTCUSDT', 'ETHUSDT']
    timeframes = ['15m', '1h']
    
    if len(sys.argv) > 1 and sys.argv[1] == '--full':
        # 全部幣种模式
        logger.info('茅作: 全部幣种模式')
        tuner.run_tuning()  # 使用默认的所有幣种
    else:
        # 快速模式（仅 BTC/ETH）
        logger.info('茅作: 快速模式 (BTC/ETH)')
        tuner.run_tuning(symbols=symbols, timeframes=timeframes)
