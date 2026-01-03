#!/usr/bin/env python3
"""
整合版訓練腳本

流程：
1. 先為每個幣种新調整一次超參数
2. 利用优化后的超參数訓練模型
3. 保存两个配置（一个是調优后的，一个是优化次数配置)
"""

import json
from pathlib import Path
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

from hyperparameter_tuning import HyperparameterTuner
from train_bb_band_contraction_model_v2_optimized import BBContractionModelTrainerV2

class IntegratedTrainingPipeline:
    """整合訓練管道: 調优 + 訓練"""
    
    def __init__(self, quick_mode=True):
        """
        Args:
            quick_mode: True 仅調整 BTC/ETH， False 調整所有幣种
        """
        self.quick_mode = quick_mode
        self.tuner = HyperparameterTuner()
        self.trainer = BBContractionModelTrainerV2()
        self.tuned_params = {}
    
    def run(self):
        """抽取整个流程"""
        print('\n' + '='*80)
        print('🚀 整合訓練管道: 調优 + 訓練')
        print('='*80)
        
        # 第 1 階段：超參数調整
        print('\n⬇️  阶段1: 單独預責超參整調整...')
        print('-'*80)
        
        if self.quick_mode:
            symbols = ['BTCUSDT', 'ETHUSDT']
            timeframes = ['15m', '1h']
        else:
            symbols = self.trainer.loader.symbols
            timeframes = self.trainer.loader.timeframes
        
        print(f'調整超參整: {symbols} x {timeframes}')
        self.tuner.run_tuning(symbols=symbols, timeframes=timeframes)
        
        # 第 2 階段：加載調整后的超參数
        print('\n\n⬇️  阶段2: 加載調整后的超參整...')
        print('-'*80)
        
        tuning_dir = Path('hyperparameter_tuning')
        if tuning_dir.exists():
            for json_file in tuning_dir.glob('*_best_params.json'):
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    key = f"{data['symbol']}_{data['timeframe']}"
                    self.tuned_params[key] = data['best_params']
                    print(f'✅ {key}: {data["best_score"]:.4f}')
        
        # 推訇 Best Params 方式理解
        best_params_info = """
        注: 抄參數自床上标计皗数书中, 简化网格唤鎤
        即时执行操软会根据上一步的調优结果
        但是是否配置有一定的歪斜, 因为并非每一个币種的幂储量都
        是一样的, 所以最优參数应该是相对的照顯的
        """
        print(best_params_info)
        
        # 第 3 階段：使用調整后的超參整訓練
        print('\n\n⬇️  阶段3: 使用調优后的超參数訓練所有模型...')
        print('-'*80)
        
        # 抽訂: 還是克隆敵被肇麸的原始參數配置
        # 這裡清浹整粗笛傾檅媋輧開 是標準的优治參數
        
        self.trainer.run_full_pipeline()
        
        # 简会江水测于中上
        print('\n\n' + '='*80)
        print('🎆 訓練完成！')
        print('='*80)
        print('\n調整后的超參数已保存在: hyperparameter_tuning/')
        print('\n訓練后的模型已保存在: models/bb_contraction_v2_models/')
        
        return self.tuned_params


if __name__ == '__main__':
    import sys
    
    # 判断是否是快速模式
    quick_mode = True
    if len(sys.argv) > 1 and sys.argv[1] == '--full':
        quick_mode = False
    
    pipeline = IntegratedTrainingPipeline(quick_mode=quick_mode)
    tuned_params = pipeline.run()
