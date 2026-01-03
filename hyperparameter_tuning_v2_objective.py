#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
超參數調整器 - 客觀公式版本

為每個币种新找最優的 XGBoost 超參數
使用 Grid Search 不是 Optuna（旨在控制耗時）
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score
from xgboost import XGBClassifier
import warnings
import json
import io
import sys

warnings.filterwarnings("ignore")

# 修複 Windows Unicode 編碼
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

try:
    from data_loader import CryptoDataLoader
except ImportError:
    logger.error("[ERROR] Cannot find data_loader")
    exit(1)

from train_bb_band_v2_objective_formula import BBContractionFeatureExtractorV3_Objective


class HyperparameterTunerV2_Objective:
    """超參數調整器 - 客觀公式版本"""

    def __init__(self, output_dir="hyperparameter_tuning_v2_objective"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.loader = CryptoDataLoader()

    def prepare_data(self, symbol: str, timeframe: str):
        """準備訓練數據"""
        logger.info(f"[PREPARE] Preparing {symbol} {timeframe} data...")

        # 下載數据
        df = self.loader.download_symbol_data(symbol, timeframe)
        if df is None or len(df) < 100:
            logger.error(f"[ERROR] {symbol} {timeframe} - Insufficient data")
            return None, None, None, None

        # 提取特徵（使用客觀公式）
        extractor = BBContractionFeatureExtractorV3_Objective()
        df = extractor.create_features(df, timeframe=timeframe, lookahead=5)

        # 篩選有效標籤
        df_labeled = df[df["label_bounce_valid"] != -1].copy()

        if len(df_labeled) < 50:
            logger.error(f"[ERROR] {symbol} {timeframe} - Insufficient labeled samples")
            return None, None, None, None

        # 特征選擇
        feature_cols = extractor.features
        X = df_labeled[feature_cols]
        y = df_labeled["label_bounce_valid"]

        # 移除 NaN
        mask = ~X.isna().any(axis=1)
        X = X[mask]
        y = y[mask]

        # 替換無窮大
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median())
        X = X.fillna(0)

        if len(X) < 30:
            logger.error(f"[ERROR] {symbol} {timeframe} - Insufficient samples after cleaning")
            return None, None, None, None

        logger.info(f"[OK] Valid samples: {len(X):,}")

        return X, y, feature_cols, df

    def tune_grid_search(self, symbol: str, timeframe: str):
        """使用 Grid Search 進行超參數調整"""
        logger.info(f"[TUNING] Grid Search Hyperparameter Tuning...")

        # 準備数据
        X, y, feature_cols, df = self.prepare_data(symbol, timeframe)
        if X is None:
            return None, None

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

        # 定義網格
        param_grid = {
            "n_estimators": [100, 150, 200, 250],
            "max_depth": [5, 6, 7, 8, 9],
            "learning_rate": [0.03, 0.05, 0.08, 0.1],
            "reg_alpha": [0.5, 1.0, 2.0],
            "reg_lambda": [1.0, 2.0, 3.0],
        }

        best_score = -1
        best_params = None
        trial_count = 0

        # 简化網格（只测試最重要的參數）
        logger.info("[GRID] Simplified Grid Search (testing key parameters)...")

        simplified_grid = {
            "n_estimators": [100, 200, 250],
            "max_depth": [6, 7, 8],
            "learning_rate": [0.05, 0.08],
            "reg_alpha": [0.5, 1.0],
            "reg_lambda": [1.0, 2.0],
        }

        from itertools import product

        for params_tuple in product(*simplified_grid.values()):
            trial_count += 1
            params = dict(zip(simplified_grid.keys(), params_tuple))

            # 設置默认值
            params["subsample"] = 0.8
            params["colsample_bytree"] = 0.7

            model = XGBClassifier(
                **params,
                random_state=42,
                eval_metric="logloss",
                verbosity=0,
                scale_pos_weight=len(y_train[y_train == 0])
                / (len(y_train[y_train == 1]) + 1e-8),
            )

            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)

            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            score = 0.7 * precision + 0.3 * recall

            logger.info(
                f"[TRIAL {trial_count}] n_est={params['n_estimators']}, depth={params['max_depth']}, lr={params['learning_rate']:.2f} => Score={score:.4f}"
            )

            if score > best_score:
                best_score = score
                best_params = params.copy()

        logger.info(f"[BEST] Best hyperparameters: {best_params}")
        logger.info(f"[BEST] Best score: {best_score:.4f}")

        return best_params, best_score

    def save_best_params(self, symbol: str, timeframe: str, params: dict, score: float):
        """保存最優超參數"""
        results_file = (
            self.output_dir / f"{symbol}_{timeframe}_best_params.json"
        )

        data = {
            "symbol": symbol,
            "timeframe": timeframe,
            "best_params": params,
            "best_score": score,
            "timestamp": datetime.now().isoformat(),
        }

        with open(results_file, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"[SAVED] Results saved to: {results_file}")
        return results_file

    def run_tuning(self, symbols=None, timeframes=None):
        """運行超參數調整"""
        if symbols is None:
            symbols = self.loader.symbols
        if timeframes is None:
            timeframes = self.loader.timeframes

        logger.info(f"\n{'='*80}")
        logger.info(f"[START] Starting Hyperparameter Tuning (Objective Formula Version)")
        logger.info(f"{'='*80}")
        logger.info(f"[INFO] Symbols: {len(symbols)}, Timeframes: {len(timeframes)}")
        logger.info(f"[INFO] Total tasks: {len(symbols) * len(timeframes)}")

        for symbol in symbols:
            for timeframe in timeframes:
                logger.info(f"\n[TASK] Tuning {symbol} {timeframe}...")

                try:
                    best_params, best_score = self.tune_grid_search(
                        symbol, timeframe
                    )

                    if best_params:
                        self.save_best_params(symbol, timeframe, best_params, best_score)
                    else:
                        logger.warning(f"[WARNING] {symbol} {timeframe} tuning failed")

                except Exception as e:
                    logger.error(f"[ERROR] {symbol} {timeframe}: {e}")
                    import traceback

                    traceback.print_exc()

        logger.info(f"\n{'='*80}")
        logger.info(f"[COMPLETED] Hyperparameter Tuning Complete")
        logger.info(f"{'='*80}")
        logger.info(f"[RESULTS] Saved to: {self.output_dir}")


if __name__ == "__main__":
    import sys

    tuner = HyperparameterTunerV2_Objective()

    if len(sys.argv) > 1 and sys.argv[1] == "--full":
        # 所有幣种
        tuner.run_tuning()
    else:
        # 快速模式：仅 BTC/ETH
        tuner.run_tuning(symbols=["BTCUSDT", "ETHUSDT"], timeframes=["15m", "1h"])
