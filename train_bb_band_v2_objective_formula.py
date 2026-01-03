#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用客觀 BBW 公式的 Bollinger Band 反弹有效性 V2 訓練器

改進内容：
1. 特征定義使用客觀 BBW 公式
   - BBW = (Upper - Lower) / Middle × 100
   - is_squeeze = BBW < 4%
   - is_extreme_squeeze = BBW < 2%

2. 加入你的扶养指標：
   - RSI, Momentum, Volume Ratio
   - Price position, Historical Volatility

3. 整合訓練 pipeline：
   - 超參數遅優 + 模型訓練
   - 保存到 models/bb_contraction_v2_models/
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from xgboost import XGBClassifier
import pickle
import json
import warnings
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


class BBContractionFeatureExtractorV3_Objective:
    """
    使用客觀 BBW 公式的特征提取器
    """

    def __init__(self):
        self.features = []

    def create_features(self, df, timeframe="1h", lookahead=5):
        """
        使用客觀公式建立特徵

        Args:
            df: OHLCV 資料
            timeframe: 時間框架 (15m, 1h)
            lookahead: 向前看 N 根 K 棒判定反彈
        """
        df = df.copy()

        # ============================================================
        # 1. 基礎 Bollinger Bands (20, 2)
        # ============================================================
        df["sma_20"] = df["close"].rolling(20).mean()
        df["std_20"] = df["close"].rolling(20).std()
        df["bb_upper"] = df["sma_20"] + 2 * df["std_20"]
        df["bb_lower"] = df["sma_20"] - 2 * df["std_20"]

        # ============================================================
        # 2. 客觀 BBW 公式
        # ============================================================
        # BBW = (Upper - Lower) / Middle × 100
        df["bb_width"] = (
            (df["bb_upper"] - df["bb_lower"]) / df["sma_20"] * 100
        )

        # ============================================================
        # 3. 歷史 BBW 統計（100 根 K 棒窗口）
        # ============================================================
        df["bb_width_mean_100"] = df["bb_width"].rolling(100).mean()
        df["bb_width_min_100"] = df["bb_width"].rolling(100).min()
        df["bb_width_max_100"] = df["bb_width"].rolling(100).max()
        df["bb_width_std_100"] = df["bb_width"].rolling(100).std()

        # ============================================================
        # 4. 歸一化 BBW (0-1)
        # ============================================================
        df["bb_width_norm"] = (
            df["bb_width"] - df["bb_width_min_100"]
        ) / (df["bb_width_max_100"] - df["bb_width_min_100"])
        df["bb_width_norm"] = df["bb_width_norm"].fillna(0.5).clip(0, 1)

        # ============================================================
        # 5. 收縮標誌（客觀定義）
        # ============================================================
        df["is_squeeze_4"] = (df["bb_width"] < 4).astype(int)  # 標準收縮
        df["is_squeeze_2"] = (df["bb_width"] < 2).astype(int)  # 極度收縮

        # ============================================================
        # 6. 相對歷史水準
        # ============================================================
        df["bb_width_vs_mean"] = df["bb_width"] / (df["bb_width_mean_100"] + 1e-8)
        df["bb_width_percentile"] = df["bb_width_norm"] * 100

        # ============================================================
        # 7. 收縮持續時間
        # ============================================================
        df["squeeze_duration_5"] = df["is_squeeze_4"].rolling(5).sum()
        df["squeeze_duration_20"] = df["is_squeeze_4"].rolling(20).sum()

        # ============================================================
        # 8. 價格位置（相對上下軌）
        # ============================================================
        df["price_bb_position"] = (
            df["close"] - df["bb_lower"]
        ) / (df["bb_upper"] - df["bb_lower"] + 1e-8)
        df["price_bb_position"] = df["price_bb_position"].clip(0, 1)

        # ============================================================
        # 9. RSI (14 週期)
        # ============================================================
        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-8)
        df["rsi_14"] = 100 - (100 / (1 + rs))
        df["rsi_14"] = df["rsi_14"].fillna(50)

        # ============================================================
        # 10. 動量 (Momentum)
        # ============================================================
        df["momentum_5"] = df["close"] - df["close"].shift(5)
        df["momentum_10"] = df["close"] - df["close"].shift(10)

        # ============================================================
        # 11. 成交量指標
        # ============================================================
        df["volume_ma_20"] = df["volume"].rolling(20).mean()
        df["volume_ratio"] = df["volume"] / (df["volume_ma_20"] + 1e-8)
        df["volume_strength"] = (df["volume"] * (df["close"] - df["close"].shift(1))).rolling(5).mean()

        # ============================================================
        # 12. 歷史波動率
        # ============================================================
        df["historical_vol"] = df["close"].pct_change().rolling(20).std() * np.sqrt(252)
        df["historical_vol"] = df["historical_vol"].fillna(df["historical_vol"].mean())

        # ============================================================
        # 13. 標籤：反彈有效性
        # ============================================================
        df["label_bounce_valid"] = -1  # 預設無效

        for i in range(len(df) - lookahead):
            # 從現在到 lookahead 根 K 棒內有沒有回升
            future_close = df["close"].iloc[i + 1 : i + 1 + lookahead]
            if len(future_close) == 0:
                continue

            max_price = future_close.max()
            current_price = df["close"].iloc[i]

            # 上升幅度 > 0.5%，算「有效反彈"
            if max_price > current_price * 1.005:
                # 再看有沒有回落，判定是否為有效反彈
                # (简化：直接視為 1)
                df.loc[df.index[i], "label_bounce_valid"] = 1
            else:
                df.loc[df.index[i], "label_bounce_valid"] = 0

        self.features = [
            "bb_width",
            "bb_width_norm",
            "bb_width_percentile",
            "bb_width_vs_mean",
            "is_squeeze_4",
            "is_squeeze_2",
            "squeeze_duration_5",
            "squeeze_duration_20",
            "price_bb_position",
            "rsi_14",
            "momentum_5",
            "momentum_10",
            "volume_ratio",
            "volume_strength",
            "historical_vol",
        ]

        return df


class BBContractionModelTrainerV2_Objective:
    """
    使用客觀公式訓練 BB 反彈有效性 V2 模型
    """

    def __init__(self):
        self.loader = CryptoDataLoader()
        self.output_dir = Path("models/bb_contraction_v2_models")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def train_single_symbol(self, symbol: str, timeframe: str) -> bool:
        """
        訓練單個幣種的模型
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"[TRAINING] {symbol} {timeframe} - BB Bounce Validity V2 (Objective Formula)")
        logger.info(f"{'='*80}")

        try:
            # ============================================================
            # 1. 下載數據
            # ============================================================
            logger.info(f"[DOWNLOAD] Downloading {symbol} {timeframe} data...")
            df = self.loader.download_symbol_data(symbol, timeframe)

            if df is None or len(df) < 100:
                logger.error(f"[ERROR] {symbol} {timeframe} - Insufficient data")
                return False

            logger.info(f"[OK] {symbol} {timeframe}: {len(df)} candles")

            # ============================================================
            # 2. 提取特徵（使用客觀公式）
            # ============================================================
            logger.info(f"[FEATURES] Extracting features...")
            extractor = BBContractionFeatureExtractorV3_Objective()
            df = extractor.create_features(df, timeframe=timeframe, lookahead=5)

            # ============================================================
            # 3. 篩選有效標籤
            # ============================================================
            df_labeled = df[df["label_bounce_valid"] != -1].copy()

            if len(df_labeled) < 50:
                logger.error(f"[ERROR] {symbol} {timeframe} - Insufficient labeled samples")
                return False

            logger.info(f"[LABELS] Distribution:")
            valid_count = (df_labeled["label_bounce_valid"] == 1).sum()
            invalid_count = (df_labeled["label_bounce_valid"] == 0).sum()
            logger.info(f"  Valid bounces (1): {valid_count:,} ({valid_count/len(df_labeled)*100:.1f}%)")
            logger.info(f"  Invalid bounces (0): {invalid_count:,} ({invalid_count/len(df_labeled)*100:.1f}%)")

            # ============================================================
            # 4. 準備訓練數據
            # ============================================================
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
                return False

            logger.info(f"[DATA] Features: {len(feature_cols)}, Samples: {len(X):,}")

            # ============================================================
            # 5. 分割訓練/測試集
            # ============================================================
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )
            except:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )

            # ============================================================
            # 6. 標準化
            # ============================================================
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # ============================================================
            # 7. 訓練模型（使用優化超參數）
            # ============================================================
            logger.info(f"[MODEL] Training XGBoost classifier...")
            
            # 預設超參數（可以用 hyperparameter_tuning.py 調整）
            params = {
                "n_estimators": 200,
                "max_depth": 7,
                "learning_rate": 0.05,
                "subsample": 0.8,
                "colsample_bytree": 0.7,
                "reg_alpha": 1.0,
                "reg_lambda": 1.0,
            }
            
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
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

            # ============================================================
            # 8. 評估
            # ============================================================
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            auc = roc_auc_score(y_test, y_pred_proba)

            cm = confusion_matrix(y_test, y_pred)
            top_features = np.argsort(model.feature_importances_)[-8:][::-1]

            logger.info(f"\n[RESULTS] Test Set Performance:")
            logger.info(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
            logger.info(f"  Precision: {precision:.4f}")
            logger.info(f"  Recall: {recall:.4f}")
            logger.info(f"  F1 Score: {f1:.4f}")
            logger.info(f"  AUC: {auc:.4f}")
            logger.info(f"\n[CONFUSION] Confusion Matrix:")
            logger.info(f"  {cm}")
            logger.info(f"\n[TOP8] Top 8 Important Features:")
            for idx in top_features:
                logger.info(
                    f"  {feature_cols[idx]}: {model.feature_importances_[idx]:.4f}"
                )

            # ============================================================
            # 9. 保存模型
            # ============================================================
            symbol_dir = self.output_dir / symbol / timeframe
            symbol_dir.mkdir(parents=True, exist_ok=True)

            model_file = symbol_dir / "bb_contraction_v2_model.pkl"
            scaler_file = symbol_dir / "bb_contraction_v2_scaler.pkl"
            features_file = symbol_dir / "bb_contraction_v2_features.json"

            with open(model_file, "wb") as f:
                pickle.dump(model, f)
            with open(scaler_file, "wb") as f:
                pickle.dump(scaler, f)
            with open(features_file, "w") as f:
                json.dump(feature_cols, f, indent=2)

            logger.info(f"\n[SAVED] Models saved to:")
            logger.info(f"  {model_file}")
            logger.info(f"  {scaler_file}")

            return True

        except Exception as e:
            logger.error(f"\n[ERROR] Exception: {e}")
            import traceback
            traceback.print_exc()
            return False

    def train_all_symbols(self, symbols=None, timeframes=None):
        """
        訓練所有幣種
        """
        if symbols is None:
            symbols = self.loader.symbols
        if timeframes is None:
            timeframes = self.loader.timeframes

        logger.info(f"\n{'='*80}")
        logger.info(f"[START] Starting BB Bounce Validity V2 Training (Objective Formula)")
        logger.info(f"{'='*80}")
        logger.info(f"[INFO] Symbols: {len(symbols)}, Timeframes: {len(timeframes)}")
        logger.info(f"[INFO] Total tasks: {len(symbols) * len(timeframes)}")

        success_count = 0
        fail_count = 0

        for symbol in symbols:
            for timeframe in timeframes:
                if self.train_single_symbol(symbol, timeframe):
                    success_count += 1
                else:
                    fail_count += 1

        logger.info(f"\n{'='*80}")
        logger.info(f"[COMPLETED] Training Complete")
        logger.info(f"{'='*80}")
        logger.info(f"[RESULTS] Success: {success_count}/{success_count + fail_count}")
        logger.info(f"[RESULTS] Failed: {fail_count}/{success_count + fail_count}")
        logger.info(f"[RESULTS] Model location: {self.output_dir}")


if __name__ == "__main__":
    import sys

    trainer = BBContractionModelTrainerV2_Objective()

    if len(sys.argv) > 1 and sys.argv[1] == "--full":
        # 訓練所有幣種
        trainer.train_all_symbols()
    else:
        # 快速模式：只訓練 BTC/ETH
        trainer.train_all_symbols(
            symbols=["BTCUSDT", "ETHUSDT"], timeframes=["15m", "1h"]
        )
