#!/usr/bin/env python3
"""
客觀公式版本的快速測試

這個脚本盤試一个 BTCUSDT 15m 的整個訓練流程，看性能是否改善了

真对準確率和召回率，比較旧版本幾何提取
"""

import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any
import sys

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
        print("\n" + "=" * 80)
        print(f"🧪 客觀公式版本快速測試: {self.symbol} {self.timeframe}")
        print("=" * 80)

        try:
            # ========================================
            # 階段1: 超參数調優
            # ========================================
            print(f"\n\u2b07️  階段1: 超參数調優...")
            print("-" * 80)

            logger.info(f"\n[TUNING START] {self.symbol} {self.timeframe}")

            best_params, best_score = self.tuner.tune_grid_search(
                self.symbol, self.timeframe
            )

            if not best_params:
                logger.error(f"\u274c {self.symbol} {self.timeframe} 調整失敗")
                return False

            self.tuner.save_best_params(
                self.symbol, self.timeframe, best_params, best_score
            )
            logger.info(
                f"\u2705 {self.symbol} {self.timeframe} 調整完成 - Score: {best_score:.4f}"
            )

            # ========================================
            # 階段2: 模型訓練
            # ========================================
            print(f"\n\u2b07️  階段2: 模型訓練...")
            print("-" * 80)

            logger.info(f"\n[TRAINING START] {self.symbol} {self.timeframe}")

            success = self.trainer.train_single_symbol(
                self.symbol, self.timeframe
            )

            if not success:
                logger.error(f"\u274c {self.symbol} {self.timeframe} 訓練失敗")
                return False

            logger.info(
                f"\u2705 {self.symbol} {self.timeframe} 訓練完成"
            )

            # ========================================
            # 汐記
            # ========================================
            print(f"\n" + "=" * 80)
            print(f"✅ 完整測試成功！")
            print(f"=" * 80)

            print(f"\n📈 結果位置：")
            print(
                f"  超參数: hyperparameter_tuning_v2_objective/{self.symbol}_{self.timeframe}_best_params.json"
            )
            print(
                f"  模型: models/bb_contraction_v2_models/{self.symbol}/{self.timeframe}/bb_contraction_v2_model.pkl"
            )

            print(f"\n🤗 特徵提取方案：客觀 BBW 公式")
            print(f"  ✅ BBW = (Upper - Lower) / Middle × 100")
            print(f"  ✅ is_squeeze = BBW < 4%")
            print(f"  ✅ is_extreme_squeeze = BBW < 2%")

            print(f"\n下一步：对比 V1 和 V2 的性能")
            print(f"\n{"=" * 80}\n")

            return True

        except Exception as e:
            logger.error(f"\u26a0️ 錯誤: {e}")
            import traceback

            traceback.print_exc()
            return False


if __name__ == "__main__":
    tester = QuickTestV2Objective(symbol="BTCUSDT", timeframe="15m")
    success = tester.run()

    sys.exit(0 if success else 1)
