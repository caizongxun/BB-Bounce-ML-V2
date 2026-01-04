"""
標籤參數調試工具

用於找到最優的標籤參數組合，目標是達到 99% 的準確率
"""

import numpy as np
import pandas as pd
import logging
from pathlib import Path
import json
from label_v3_clean import BBTouchLabelCreator

# 創建日誌目錄
Path('logs').mkdir(exist_ok=True)
Path('outputs/parameter_tuning').mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LabelParameterTuner:
    """標籤參數調試器"""
    
    def __init__(self):
        self.results = []
    
    def test_parameters(self, 
                       touch_thresholds=[0.02, 0.05, 0.1, 0.15, 0.2],
                       lookaheads=[3, 5, 7, 10],
                       min_rebound_pcts=[0.05, 0.1, 0.15, 0.2],
                       symbol='BTCUSDT',
                       timeframe='15m'):
        """
        測試不同的參數組合
        
        記錄：
        - 觸碰點數量
        - 有效反彈比例
        - 標籤準確率
        """
        
        logger.info('開始參數調試...')
        logger.info('=' * 80)
        
        param_count = len(touch_thresholds) * len(lookaheads) * len(min_rebound_pcts)
        tested = 0
        
        for touch_thresh in touch_thresholds:
            for lookahead in lookaheads:
                for min_rebound in min_rebound_pcts:
                    tested += 1
                    logger.info(f'[{tested}/{param_count}] 測試參數：')
                    logger.info(f'  touch_threshold={touch_thresh}')
                    logger.info(f'  lookahead={lookahead}')
                    logger.info(f'  min_rebound_pct={min_rebound}')
                    
                    try:
                        creator = BBTouchLabelCreator(
                            bb_period=20,
                            bb_std=2,
                            touch_threshold=touch_thresh,
                            lookahead=lookahead,
                            min_rebound_pct=min_rebound
                        )
                        
                        creator.load_data(symbol, timeframe)
                        creator.calculate_bb_bands()
                        creator.detect_touches()
                        labels, label_details = creator.create_labels()
                        
                        # 計算指標
                        metrics = self._calculate_metrics(creator, labels, label_details)
                        
                        result = {
                            'touch_threshold': touch_thresh,
                            'lookahead': lookahead,
                            'min_rebound_pct': min_rebound,
                            **metrics
                        }
                        
                        self.results.append(result)
                        
                        logger.info(f'  觸碰數：{metrics["touch_count"]}')
                        logger.info(f'  有效反彈比例：{metrics["valid_ratio"]:.2f}%')
                        logger.info(f'  標籤準確率：{metrics["accuracy"]:.2f}%')
                        logger.info('')
                        
                    except Exception as e:
                        logger.error(f'  錯誤：{e}')
                        continue
        
        logger.info('=' * 80)
        logger.info('參數調試完成！')
        
        return self.results
    
    def _calculate_metrics(self, creator, labels, label_details):
        """計算評估指標"""
        
        valid_lower = np.sum(labels == 1)
        valid_upper = np.sum(labels == 2)
        invalid = np.sum(labels == 0)
        no_touch = np.sum(labels == -1)
        
        total_touches = valid_lower + valid_upper + invalid
        total_valid = valid_lower + valid_upper
        
        # 計算有效反彈比例
        valid_ratio = (total_valid / total_touches * 100) if total_touches > 0 else 0
        
        # 計算準確率（回測）
        accuracy = self._calculate_accuracy(creator, labels)
        
        return {
            'touch_count': total_touches,
            'valid_lower_count': valid_lower,
            'valid_upper_count': valid_upper,
            'invalid_count': invalid,
            'no_touch_count': no_touch,
            'valid_ratio': valid_ratio,
            'accuracy': accuracy
        }
    
    def _calculate_accuracy(self, creator, labels):
        """計算標籤準確率（通過回測）"""
        
        valid_lower = creator.df[creator.df['label'] == 1].index.tolist()
        valid_upper = creator.df[creator.df['label'] == 2].index.tolist()
        
        if len(valid_lower) == 0 and len(valid_upper) == 0:
            return 0
        
        correct = 0
        total = 0
        
        # 下軌有效反彈（做多）
        for idx in valid_lower:
            if idx + creator.lookahead >= len(creator.df):
                continue
            
            entry = creator.df.iloc[idx]['close']
            close = creator.df.iloc[idx + creator.lookahead]['close']
            
            if close > entry:
                correct += 1
            
            total += 1
        
        # 上軌有效反彈（做空）
        for idx in valid_upper:
            if idx + creator.lookahead >= len(creator.df):
                continue
            
            entry = creator.df.iloc[idx]['close']
            close = creator.df.iloc[idx + creator.lookahead]['close']
            
            if close < entry:
                correct += 1
            
            total += 1
        
        return (correct / total * 100) if total > 0 else 0
    
    def show_best_parameters(self, top_n=10):
        """顯示最優參數"""
        
        if not self.results:
            logger.warning('沒有測試結果')
            return
        
        # 按準確率排序
        sorted_results = sorted(
            self.results,
            key=lambda x: x['accuracy'],
            reverse=True
        )
        
        logger.info('\n' + '=' * 80)
        logger.info(f'前 {top_n} 個最優參數組合：')
        logger.info('=' * 80)
        
        for rank, result in enumerate(sorted_results[:top_n], 1):
            logger.info(f'\n排名 {rank}：')
            logger.info(f'  touch_threshold = {result["touch_threshold"]}')
            logger.info(f'  lookahead = {result["lookahead"]}')
            logger.info(f'  min_rebound_pct = {result["min_rebound_pct"]}')
            logger.info(f'  觸碰數 = {result["touch_count"]}')
            logger.info(f'  有效反彈比例 = {result["valid_ratio"]:.2f}%')
            logger.info(f'  準確率 = {result["accuracy"]:.2f}%')
        
        logger.info('\n' + '=' * 80)
        
        return sorted_results[:top_n]
    
    def save_results(self, output_file='parameter_tuning_results.json'):
        """保存結果"""
        
        output_dir = Path('outputs/parameter_tuning')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = output_dir / output_file
        
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        logger.info(f'結果已保存到：{output_path}')
        
        return output_path


def main():
    """主函數"""
    Path('logs').mkdir(exist_ok=True)
    
    tuner = LabelParameterTuner()
    
    # 測試參數組合
    # 逐步擴大搜索範圍
    results = tuner.test_parameters(
        touch_thresholds=[0.02, 0.05, 0.1, 0.15, 0.2],
        lookaheads=[3, 5, 7, 10],
        min_rebound_pcts=[0.05, 0.1, 0.15, 0.2],
        symbol='BTCUSDT',
        timeframe='15m'
    )
    
    # 顯示最佳結果
    best = tuner.show_best_parameters(top_n=10)
    
    # 保存結果
    tuner.save_results()
    
    logger.info('\n調試完成！')
    logger.info('建議使用準確率最高的參數組合')


if __name__ == '__main__':
    main()
