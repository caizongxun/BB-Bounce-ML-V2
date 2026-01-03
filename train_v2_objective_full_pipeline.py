#!/usr/bin/env python3
"""
完整訓練管道 - 客觀 BBW 公式版本

流程：
1. 超參数調優 (2-3 小時)
2. 模型訓練 (1 小時)
3. 全程記錄到 JSON + LOG
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any
import sys

from hyperparameter_tuning_v2_objective import HyperparameterTunerV2_Objective
from train_bb_band_v2_objective_formula import (
    BBContractionModelTrainerV2_Objective,
)

# ============================================================
# 日誌記錄
# ============================================================


class TrainingLogger:
    """整合詳标記錄，記錄超參数調優和訓練結果"""

    def __init__(self, log_dir="training_logs_v2_objective"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_id = timestamp

        self.log_file = self.log_dir / f"training_{timestamp}.log"
        self.json_file = self.log_dir / f"training_results_{timestamp}.json"

        # 設置 logger
        self.logger = logging.getLogger("TrainingLogger")
        self.logger.setLevel(logging.DEBUG)

        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # 为了不重複設置，清空正存在的 handler
        self.logger.handlers = []

        # 主控台輸出
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        # 檔案輸出
        file_handler = logging.FileHandler(self.log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        # 統計數據
        self.results = {
            "session_id": self.session_id,
            "start_time": datetime.now().isoformat(),
            "end_time": None,
            "duration_seconds": 0,
            "tuning_results": [],
            "training_results": [],
            "summary": {
                "total_tuning_tasks": 0,
                "successful_tuning": 0,
                "failed_tuning": 0,
                "total_training_tasks": 0,
                "successful_training": 0,
                "failed_training": 0,
                "average_accuracy": 0,
                "average_precision": 0,
                "average_f1": 0,
            },
        }

    def log_tuning_result(self, symbol: str, timeframe: str, params: Dict, score: float):
        """記錄超參数調優結果"""
        result = {
            "symbol": symbol,
            "timeframe": timeframe,
            "params": params,
            "score": float(score),
            "timestamp": datetime.now().isoformat(),
        }

        self.results["tuning_results"].append(result)
        self.results["summary"]["total_tuning_tasks"] += 1
        self.results["summary"]["successful_tuning"] += 1

        self.logger.info(f"[TUNING SUCCESS] {symbol} {timeframe} - Score: {score:.4f}")

    def log_tuning_error(self, symbol: str, timeframe: str, error: str):
        """記錄超參数調優错誤"""
        self.results["summary"]["total_tuning_tasks"] += 1
        self.results["summary"]["failed_tuning"] += 1

        self.logger.error(f"[TUNING FAILED] {symbol} {timeframe} - Error: {error}")

    def log_training_result(
        self, symbol: str, timeframe: str, metrics: Dict[str, Any]
    ):
        """記錄訓練結果"""
        result = {
            "symbol": symbol,
            "timeframe": timeframe,
            "metrics": metrics,
            "timestamp": datetime.now().isoformat(),
        }

        self.results["training_results"].append(result)
        self.results["summary"]["total_training_tasks"] += 1
        self.results["summary"]["successful_training"] += 1

        # 更新平均指標
        accuracies = [
            r["metrics"]["accuracy"] for r in self.results["training_results"]
        ]
        precisions = [
            r["metrics"]["precision"] for r in self.results["training_results"]
        ]
        f1_scores = [
            r["metrics"]["f1_score"] for r in self.results["training_results"]
        ]

        self.results["summary"]["average_accuracy"] = (
            sum(accuracies) / len(accuracies) if accuracies else 0
        )
        self.results["summary"]["average_precision"] = (
            sum(precisions) / len(precisions) if precisions else 0
        )
        self.results["summary"]["average_f1"] = (
            sum(f1_scores) / len(f1_scores) if f1_scores else 0
        )

        self.logger.info(
            f"[TRAINING SUCCESS] {symbol} {timeframe} - Accuracy: {metrics['accuracy']:.4f}, Precision: {metrics['precision']:.4f}, F1: {metrics['f1_score']:.4f}"
        )

    def log_training_error(self, symbol: str, timeframe: str, error: str):
        """記錄訓練错誤"""
        self.results["summary"]["total_training_tasks"] += 1
        self.results["summary"]["failed_training"] += 1

        self.logger.error(f"[TRAINING FAILED] {symbol} {timeframe} - Error: {error}")

    def save_results(self):
        """保存 JSON 統計檔"""
        self.results["end_time"] = datetime.now().isoformat()

        start = datetime.fromisoformat(self.results["start_time"])
        end = datetime.fromisoformat(self.results["end_time"])
        self.results["duration_seconds"] = (end - start).total_seconds()

        with open(self.json_file, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)

        self.logger.info(f"\n📄 詳标記錄已保存: {self.json_file}")

    def print_summary(self):
        """打印摘要"""
        separator = "=" * 80
        print(f"\n{separator}")
        print(f"🎆 訓練統計汐記錄")
        print(f"{separator}")

        summary = self.results["summary"]

        print(f"\n💎 超參数調優結果：")
        print(
            f"  成功: {summary['successful_tuning']}/{summary['total_tuning_tasks']}"
        )
        print(
            f"  失敗: {summary['failed_tuning']}/{summary['total_tuning_tasks']}"
        )

        print(f"\n📊 模型訓練結果：")
        print(
            f"  成功: {summary['successful_training']}/{summary['total_training_tasks']}"
        )
        print(
            f"  失敗: {summary['failed_training']}/{summary['total_training_tasks']}"
        )

        print(f"\n📏 平均性能：")
        print(f"  平均準確率: {summary['average_accuracy']:.4f} ({summary['average_accuracy']*100:.2f}%)")
        print(f"  平均精準度: {summary['average_precision']:.4f}")
        print(f"  平均 F1: {summary['average_f1']:.4f}")

        duration = self.results["duration_seconds"]
        hours = int(duration // 3600)
        minutes = int((duration % 3600) // 60)
        seconds = int(duration % 60)

        print(
            f"\n⏱️  訓練耗時: {hours}h {minutes}m {seconds}s"
        )
        print(f"\n📄 LOG 檔: {self.log_file}")
        print(f"📄 JSON 檔: {self.json_file}")
        print(f"{separator}\n")


class IntegratedTrainingPipelineV2:
    """完整訓練管道 - 客觀 BBW 公式版本"""

    def __init__(self, quick_mode=False):
        self.quick_mode = quick_mode
        self.logger = TrainingLogger()
        self.tuner = HyperparameterTunerV2_Objective()
        self.trainer = BBContractionModelTrainerV2_Objective()

    def run(self):
        """運行整個管道"""
        try:
            separator = "=" * 80
            print(f"\n{separator}")
            print(
                f"🚀 完整訓練管道: 超參数調優 + 模型訓練 (客觀 BBW 公式)"
            )
            print(f"{separator}")
            print(f"📄 LOG 檔: {self.logger.log_file}")
            print(f"📄 JSON 檔: {self.logger.json_file}")

            # ========================================
            # 階段1: 超參数調優
            # ========================================
            print(f"\n\u2b07️  階段1: 超參数調優...")
            print("-" * 80)

            if self.quick_mode:
                symbols = ["BTCUSDT", "ETHUSDT"]
                timeframes = ["15m", "1h"]
            else:
                symbols = self.trainer.loader.symbols
                timeframes = self.trainer.loader.timeframes

            print(
                f"調整目標: {len(symbols)} 幣种 x {len(timeframes)} 時框"
            )

            for symbol in symbols:
                for timeframe in timeframes:
                    try:
                        best_params, best_score = self.tuner.tune_grid_search(
                            symbol, timeframe
                        )

                        if best_params:
                            self.tuner.save_best_params(
                                symbol, timeframe, best_params, best_score
                            )
                            self.logger.log_tuning_result(
                                symbol, timeframe, best_params, best_score
                            )
                        else:
                            self.logger.log_tuning_error(
                                symbol, timeframe, "無法提取特徵"
                            )

                    except Exception as e:
                        self.logger.log_tuning_error(symbol, timeframe, str(e))

            # ========================================
            # 階段2: 模型訓練
            # ========================================
            print(f"\n\u2b07️  階段2: 模型訓練...")
            print("-" * 80)

            self.trainer.train_all_symbols(symbols=symbols, timeframes=timeframes)

            # ========================================
            # 保存詳标記錄
            # ========================================
            self.logger.save_results()
            self.logger.print_summary()

            print(f"\n{separator}")
            print("✅ 訓練完成！")
            print(f"{separator}\n")

            return True

        except KeyboardInterrupt:
            print("\n⚠️ 訓練被中斷 (Ctrl+C)")
            self.logger.save_results()
            self.logger.print_summary()
            sys.exit(1)

        except Exception as e:
            print(f"\n⚠️ 错誤: {e}")
            self.logger.save_results()
            self.logger.print_summary()
            import traceback

            traceback.print_exc()
            sys.exit(1)


if __name__ == "__main__":
    import sys

    quick_mode = True
    if len(sys.argv) > 1 and sys.argv[1] == "--full":
        quick_mode = False

    pipeline = IntegratedTrainingPipelineV2(quick_mode=quick_mode)
    pipeline.run()
