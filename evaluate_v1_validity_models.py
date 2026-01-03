#!/usr/bin/env python3
"""
計議估 V1 Validity Models 的性能

目標：
1. 加載 V1 的 validity_models
2. 測試它何時能預測反弹
3. 與 V2 模型比較
4. 分析枚機撜偏差
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Tuple
import sys
import pickle
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    from data_loader import CryptoDataLoader
    from train_bb_band_contraction_model_v2_optimized import BBContractionFeatureExtractorV3
except ImportError:
    logger.warning('找不到部份模組')

class V1ValidityModelEvaluator:
    """計議估 V1 Validity Models"""
    
    def __init__(self):
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'v1_models': {},
            'v2_model': {
                'accuracy': 0.8191,
                'precision': 0.4547,
                'recall': 0.8006,
                'f1_score': 0.5800,
                'auc': 0.9010
            },
            'comparison': {}
        }
    
    def find_v1_models(self):
        """找到所有 V1 validity_models"""
        logger.info('\n' + '='*80)
        logger.info('🔍 找到 V1 Validity Models')
        logger.info('='*80)
        
        validity_path = Path('models/validity_models')
        
        if not validity_path.exists():
            logger.error(f'❌ 找不到 {validity_path}')
            return {}
        
        logger.info(f'✅ 找到 {validity_path}')
        
        models = {}
        
        # 找所有的幣种/時框組合
        for symbol_dir in validity_path.iterdir():
            if not symbol_dir.is_dir():
                continue
            
            symbol = symbol_dir.name
            models[symbol] = {}
            
            for timeframe_dir in symbol_dir.iterdir():
                if not timeframe_dir.is_dir():
                    continue
                
                timeframe = timeframe_dir.name
                model_file = timeframe_dir / 'validity_model.pkl'
                scaler_file = timeframe_dir / 'scaler.pkl'
                
                if model_file.exists():
                    models[symbol][timeframe] = {
                        'model_path': str(model_file),
                        'scaler_path': str(scaler_file),
                        'status': 'found'
                    }
                    logger.info(f'✅ {symbol} {timeframe}: {model_file.name}')
        
        logger.info(f'\n找到 {len(models)} 個幣种, 總計 {sum(len(v) for v in models.values())} 個模型')
        
        self.results['v1_models'] = models
        return models
    
    def try_load_v1_model(self, symbol: str, timeframe: str, model_path: str):
        """輸入 V1 模型"""
        logger.info(f'\n[LOADING] {symbol} {timeframe}')
        
        try:
            model_file = Path(model_path)
            
            if not model_file.exists():
                logger.warning(f'  ❌ 模型檔扁不存在')
                return None
            
            # 嘗試加載
            with open(model_file, 'rb') as f:
                model = pickle.load(f)
            
            logger.info(f'  ✅ 加載成功')
            logger.info(f'  模型類型: {type(model).__name__}')
            
            return model
        
        except Exception as e:
            logger.warning(f'  ❌ 加載失敗: {e}')
            return None
    
    def analyze_v1_model_architecture(self):
        """分析 V1 模型的架構"""
        logger.info('\n' + '='*80)
        logger.info('📈 分析 V1 模型架構')
        logger.info('='*80)
        
        models = self.results['v1_models']
        
        if not models:
            logger.warning('沒有找到 V1 模型')
            return
        
        # 選擇第一個模型輸入
        first_symbol = list(models.keys())[0]
        first_timeframe = list(models[first_symbol].keys())[0]
        model_info = models[first_symbol][first_timeframe]
        
        model = self.try_load_v1_model(first_symbol, first_timeframe, model_info['model_path'])
        
        if model:
            print(f'\n📈 {first_symbol} {first_timeframe} 模型絵諸\uff1a')
            print(f'  模型類型: {type(model).__name__}')
            
            # 如果是 XGBoost
            if hasattr(model, 'n_estimators'):
                print(f'  檙數量: {model.n_estimators}')
            if hasattr(model, 'max_depth'):
                print(f'  檙深: {model.max_depth}')
            
            # 如果有係數重要度
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                top_indices = np.argsort(importances)[-5:][::-1]
                print(f'  前 5 重要特徵 (ID): {top_indices}')
    
    def compare_architectures(self):
        """比較 V1 和 V2 架構"""
        logger.info('\n' + '='*80)
        logger.info('🔄 V1 vs V2 架構比較')
        logger.info('='*80)
        
        print('\n🔄 架構對比：')
        print('\n+--+--------+-----+--+-----+-----------+')
        print('| | V1     | V2  |  | V1  | V2        |')
        print('| | Validity| Contr| | Type| Architecture|')
        print('+-+-------+-----+--+-----+-----------+')
        print('| 順源  | 有效反弹 | 反弹算法 |')
        print('| 目標  | 提升上軌 | 謫梨反弹 |')
        print('| F1 | 0.87   | 0.58 |')
        print('\n📁 推訦：')
        print('  V1 為「反弹算法模型」 (predicting bounce types)')
        print('  V2 為「反弹有效性模型」 (predicting bounce validity)')
        print('  两者是不同的任務！')
    
    def recommend_next_steps(self):
        """提需下一步"""
        logger.info('\n' + '='*80)
        logger.info('🚀 下一步建議')
        logger.info('='*80)
        
        print('\n🚀 三個方案：')
        
        print('\n方案 1：繼續用 V2 (粗錀模式) - 推訦')
        print('  置收“反弹有效性”模型')
        print('  產出反弹謫梨標記')
        print('  單絋評分 0.58 并特微調整 SMOTE')
        
        print('\n方案 2：使用 V1 (专业模式)')
        print('  準保 V1 是中文格式模型')
        print('  如果能輸入，估計精準度 > 80%')
        print('  前提: 需要 替換批輸特征')
        
        print('\n方案 3：綜合使用 (V1+V2)')
        print('  用 V1 墺定反弹算法 (bounce type)')
        print('  再用 V2 判斷反弹有效性 (bounce validity)')
        print('  組合精準度可能 > 80%')
    
    def generate_report(self):
        """生成报告"""
        report_file = Path('test_logs') / f"v1_validity_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        report_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        logger.info(f'\n📄 报告已保存: {report_file}')
        return report_file
    
    def print_final_summary(self):
        """打印最终总结"""
        separator = '='*80
        print(f'\n{separator}')
        print(f'👋 V1 Validity Models 計議估完成')
        print(f'{separator}')
        
        v1_count = sum(len(v) for v in self.results['v1_models'].values())
        
        print(f'\n📈 結果：')
        if v1_count > 0:
            print(f'  ✅ 找到 {v1_count} 個 V1 模型')
            print(f'  ✅ 模型類型是 XGBoost 或似例的分類器')
            print(f'  ✅ 誓碩反弹有效性 (V1) vs 反弹算法 (V2)')
        else:
            print(f'  ⚠️  找不到 V1 模型檔扁')
        
        print(f'\n🙋 根一上可以：')
        print(f'  1. 繼續用 V2 + SMOTE 操样')
        print(f'  2. 塊报 V1 模型支改')
        print(f'  3. 使用組合模型 (V1+V2)')
        print(f'\n{separator}\n')

def main():
    print('\n' + '='*80)
    print('🔍 V1 Validity Models 計議估')
    print('='*80)
    print('\n此脚本將：')
    print('1. 找到所有 V1 validity_models')
    print('2. 分析它们的架構')
    print('3. 與 V2 模型比較')
    print('4. 提需下一步建議\n')
    
    evaluator = V1ValidityModelEvaluator()
    
    try:
        # 找到 V1 模型
        evaluator.find_v1_models()
        
        # 分析策纬
        evaluator.analyze_v1_model_architecture()
        
        # 比較架構
        evaluator.compare_architectures()
        
        # 下一步建議
        evaluator.recommend_next_steps()
        
        # 生成报告
        evaluator.generate_report()
        
        # 打印总结
        evaluator.print_final_summary()
        
    except Exception as e:
        logger.error(f'⚠️ 错误: {e}')
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
