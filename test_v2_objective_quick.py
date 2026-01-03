#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
客觀公式版本的快速測試

這個腳本測試一個 BTCUSDT 15m 的整個訓練流程，看性能是否改善了

真對準確率和召回率，比較舊版本幾何提取
"""

import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any
import sys
import io

# 修複 Windows Unicode 編碼
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from hyperparameter_tuning_v2_objective import HyperparameterTunerV2_Objective
from train_bb_band_v2_objective_formula import (
    BBContractionModelTrainerV2_Objective,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class QuickTestV2Objective:
    """客觀公式版本的快速測試"""

    def __init__(self, symbol="BTCUSDT", timeframe="15m"):
        self.symbol = symbol
        self.timeframe = timeframe
        self.tuner = HyperparameterTunerV2_Objective()
        self.trainer = BBContractionModelTrainerV2_Objective()

    def run(self):
        """運行完整測試流程"""
        separator = "=" * 80
        print(f"\n{separator}")
        print(f"[QUICK TEST] 客觀公式版本快速測試: {self.symbol} {self.timeframe}")
        print(f"{separator}")

        try:
            # ========================================
            # 階段1: 超參數調優
            # ========================================
            print(f"\n[PHASE 1] 階段1: 超參數調優...")
            print("-" * 80)

            logger.info(f"\n[TUNING START] {self.symbol} {self.timeframe}")

            best_params, best_score = self.tuner.tune_grid_search(
                self.symbol, self.timeframe
            )

            if not best_params:
                logger.error(f"[ERROR] {self.symbol} {self.timeframe} 調優失敗")
                return False

            self.tuner.save_best_params(
                self.symbol, self.timeframe, best_params, best_score
            )
            logger.info(
                f"[SUCCESS] {self.symbol} {self.timeframe} 調優完成 - Score: {best_score:.4f}"
            )

            # ========================================
            # 階段2: 模型訓練
            # ========================================
            print(f"\n[PHASE 2] 階段2: 模型訓練...")
            print("-" * 80)

            logger.info(f"\n[TRAINING START] {self.symbol} {self.timeframe}")

            success = self.trainer.train_single_symbol(
                self.symbol, self.timeframe
            )

            if not success:
                logger.error(f"[ERROR] {self.symbol} {self.timeframe} 訓練失敗")
                return False

            logger.info(
                f"[SUCCESS] {self.symbol} {self.timeframe} 訓練完成"
            )

            # ========================================
            # 摘要
            # ========================================
            print(f"\n{separator}")
            print(f"[SUCCESS] 完整測試成功!")
            print(f"{separator}")

            print(f"\n[RESULTS] 結果位置：")
            print(
                f"  超參數: hyperparameter_tuning_v2_objective/{self.symbol}_{self.timeframe}_best_params.json"
            )
            print(
                f"  模型: models/bb_contraction_v2_models/{self.symbol}/{self.timeframe}/bb_contraction_v2_model.pkl"
            )

            print(f"\n[FEATURES] 特徵提取方案：客觀 BBW 公式")
            print(f"  [OK] BBW = (Upper - Lower) / Middle x 100")
            print(f"  [OK] is_squeeze = BBW < 4%")
            print(f"  [OK] is_extreme_squeeze = BBW < 2%")

            print(f"\n[NEXT] 下一步：對比 V1 和 V2 的性能")
            print(f"\n{separator}\n")

            return True

        except Exception as e:
            logger.error(f"[ERROR] 錯誤: {e}")
            import traceback

            traceback.print_exc()
            return False


if __name__ == "__main__":
    tester = QuickTestV2Objective(symbol="BTCUSDT", timeframe="15m")
    success = tester.run()

    sys.exit(0 if success else 1)
