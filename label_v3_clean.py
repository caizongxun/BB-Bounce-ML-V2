"""
BB 反彈標籤系統 - 重新設計版本

邏輯：
1. 標記所有觸碰/接近 BB 上軌或下軌的 K 棒
2. 在這些 K 棒上判斷後續是否有有效反彈
3. 驗證標籤準確率

標籤定義：
- 1: 觸碰下軌，且後續有上漲反彈 (有效)
- 0: 觸碰下軌，但後續無上漲反彈 (無效)
- 2: 觸碰上軌，且後續有下跌反彈 (有效)
- -1: 無觸碰 (忽略)
"""

import numpy as np
import pandas as pd
import logging
from datetime import datetime
from pathlib import Path
import json

# 創建日誌目錄
Path('logs').mkdir(exist_ok=True)
Path('outputs/labels').mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/label_creation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class BBTouchLabelCreator:
    """BB 觸碰標籤創建器"""
    
    def __init__(self, 
                 bb_period=20,
                 bb_std=2,
                 touch_threshold=0.05,  # 0.05% 的閾值
                 lookahead=5,           # 後續 5 根 K 棒驗證反彈
                 min_rebound_pct=0.1):  # 最小反彈 0.1%
        """
        初始化參數
        
        Args:
            bb_period: BB 計算周期
            bb_std: BB 標準差倍數
            touch_threshold: 觸碰距離閾值（百分比）
            lookahead: 後續驗證的 K 棒數
            min_rebound_pct: 最小反彈百分比
        """
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.touch_threshold = touch_threshold
        self.lookahead = lookahead
        self.min_rebound_pct = min_rebound_pct
        
        self.df = None
        self.touch_points = []
        
    def load_data(self, symbol='BTCUSDT', timeframe='15m'):
        """加載數據"""
        try:
            self.df = pd.read_csv(f'data/{symbol}_{timeframe}.csv')
            logger.info(f'成功加載 {symbol}_{timeframe}: {len(self.df)} 行數據')
            return self.df
        except FileNotFoundError:
            logger.error(f'找不到數據文件：data/{symbol}_{timeframe}.csv')
            raise
    
    def calculate_bb_bands(self):
        """計算 Bollinger Bands"""
        logger.info('計算 Bollinger Bands...')
        
        self.df['sma'] = self.df['close'].rolling(self.bb_period).mean()
        self.df['std'] = self.df['close'].rolling(self.bb_period).std()
        self.df['upper_band'] = self.df['sma'] + (self.df['std'] * self.bb_std)
        self.df['lower_band'] = self.df['sma'] - (self.df['std'] * self.bb_std)
        self.df['mid_band'] = self.df['sma']
        self.df['bb_width'] = self.df['upper_band'] - self.df['lower_band']
        
        logger.info('BB 計算完成')
    
    def detect_touches(self):
        """
        檢測所有觸碰/接近 BB 軌道的 K 棒
        
        觸碰定義：
        - 觸碰下軌：close 或 low 接近 lower_band（距離 < threshold% 的 BB 寬度）
        - 觸碰上軌：close 或 high 接近 upper_band（距離 < threshold% 的 BB 寬度）
        """
        logger.info('檢測觸碰點...')
        
        touches = []
        
        for i in range(self.bb_period, len(self.df) - self.lookahead):
            row = self.df.iloc[i]
            
            if pd.isna(row['bb_width']) or row['bb_width'] == 0:
                continue
            
            # 計算距離軌道的距離（相對於 BB 寬度的百分比）
            dist_to_lower = (row['close'] - row['lower_band']) / row['bb_width']
            dist_to_upper = (row['upper_band'] - row['close']) / row['bb_width']
            
            # 使用 low/high 也檢查一下
            low_to_lower = (row['low'] - row['lower_band']) / row['bb_width']
            high_to_upper = (row['upper_band'] - row['high']) / row['bb_width']
            
            # 檢測下軌觸碰
            if low_to_lower < self.touch_threshold or dist_to_lower < self.touch_threshold:
                touch_type = 'lower'
                touches.append({
                    'index': i,
                    'time': row.get('time', i),
                    'type': touch_type,
                    'price': row['close'],
                    'lower_band': row['lower_band'],
                    'upper_band': row['upper_band'],
                    'bb_width': row['bb_width']
                })
            
            # 檢測上軌觸碰
            elif high_to_upper < self.touch_threshold or dist_to_upper < self.touch_threshold:
                touch_type = 'upper'
                touches.append({
                    'index': i,
                    'time': row.get('time', i),
                    'type': touch_type,
                    'price': row['close'],
                    'lower_band': row['lower_band'],
                    'upper_band': row['upper_band'],
                    'bb_width': row['bb_width']
                })
        
        self.touch_points = touches
        logger.info(f'檢測到 {len(touches)} 個觸碰點')
        
        return touches
    
    def validate_rebound(self, touch_idx, touch_type):
        """
        驗證後續是否有有效反彈
        
        邏輯：
        - 下軌觸碰：後續價格是否上漲 >= min_rebound_pct
        - 上軌觸碰：後續價格是否下跌 >= min_rebound_pct
        
        Returns:
            (is_valid, max_move, future_prices)
        """
        if touch_idx + self.lookahead >= len(self.df):
            return False, 0, []
        
        touch_price = self.df.iloc[touch_idx]['close']
        future_data = self.df.iloc[touch_idx + 1:touch_idx + self.lookahead + 1]
        future_prices = future_data['close'].values
        
        if len(future_prices) < self.lookahead:
            return False, 0, future_prices
        
        if touch_type == 'lower':
            # 下軌反彈：檢查最高點是否上漲足夠
            max_price = future_prices.max()
            max_move = (max_price - touch_price) / touch_price * 100
            
            is_valid = max_move >= self.min_rebound_pct
            
        else:  # upper
            # 上軌反彈：檢查最低點是否下跌足夠
            min_price = future_prices.min()
            max_move = (touch_price - min_price) / touch_price * 100
            
            is_valid = max_move >= self.min_rebound_pct
        
        return is_valid, max_move, future_prices
    
    def create_labels(self):
        """
        為所有 K 棒創建標籤
        
        標籤：
        - 1: 觸碰下軌 + 有反彈
        - 0: 觸碰下軌 + 無反彈
        - 2: 觸碰上軌 + 有反彈
        - -1: 無觸碰
        """
        logger.info('為所有 K 棒創建標籤...')
        
        labels = np.full(len(self.df), -1, dtype=int)
        label_details = {}  # 存儲標籤詳細信息
        
        valid_count = 0
        invalid_count = 0
        
        for touch in self.touch_points:
            idx = touch['index']
            touch_type = touch['type']
            
            # 驗證反彈
            is_valid, max_move, future_prices = self.validate_rebound(idx, touch_type)
            
            if touch_type == 'lower':
                label = 1 if is_valid else 0
            else:  # upper
                label = 2 if is_valid else 0
            
            labels[idx] = label
            
            label_details[idx] = {
                'type': touch_type,
                'is_valid': is_valid,
                'max_move_pct': max_move,
                'price': touch['price'],
                'lower_band': touch['lower_band'],
                'upper_band': touch['upper_band'],
                'future_prices': future_prices.tolist()
            }
            
            if is_valid:
                valid_count += 1
            else:
                invalid_count += 1
        
        self.df['label'] = labels
        
        logger.info('=' * 60)
        logger.info('標籤統計：')
        logger.info(f'  下軌有效反彈 (label=1)：{sum(labels == 1)}')
        logger.info(f'  下軌無效反彈 (label=0)：{sum(labels == 0)}')
        logger.info(f'  上軌有效反彈 (label=2)：{sum(labels == 2)}')
        logger.info(f'  無觸碰 (label=-1)：{sum(labels == -1)}')
        logger.info(f'  總有效反彈：{sum((labels == 1) | (labels == 2))}')
        logger.info(f'  總無效反彈：{sum(labels == 0)}')
        if (valid_count + invalid_count) > 0:
            logger.info(f'  有效率：{valid_count / (valid_count + invalid_count) * 100:.2f}%')
        logger.info('=' * 60)
        
        return labels, label_details
    
    def analyze_rebound_characteristics(self, label_details):
        """分析有效反彈的特徵"""
        logger.info('\n分析有效反彈特徵...')
        
        valid_rebounds = {k: v for k, v in label_details.items() if v['is_valid']}
        invalid_rebounds = {k: v for k, v in label_details.items() if not v['is_valid']}
        
        if len(valid_rebounds) == 0:
            logger.warning('沒有有效反彈樣本')
            return
        
        valid_moves = [v['max_move_pct'] for v in valid_rebounds.values()]
        invalid_moves = [v['max_move_pct'] for v in invalid_rebounds.values()] if invalid_rebounds else []
        
        logger.info(f'\n有效反彈統計 (n={len(valid_rebounds)})：')
        logger.info(f'  平均幅度：{np.mean(valid_moves):.4f}%')
        logger.info(f'  中位幅度：{np.median(valid_moves):.4f}%')
        logger.info(f'  最大幅度：{np.max(valid_moves):.4f}%')
        logger.info(f'  最小幅度：{np.min(valid_moves):.4f}%')
        logger.info(f'  標準差：{np.std(valid_moves):.4f}%')
        
        if invalid_rebounds:
            logger.info(f'\n無效反彈統計 (n={len(invalid_rebounds)})：')
            logger.info(f'  平均幅度：{np.mean(invalid_moves):.4f}%')
            logger.info(f'  中位幅度：{np.median(invalid_moves):.4f}%')
            logger.info(f'  最大幅度：{np.max(invalid_moves):.4f}%')
            logger.info(f'  最小幅度：{np.min(invalid_moves):.4f}%')
            logger.info(f'  標準差：{np.std(invalid_moves):.4f}%')
    
    def save_labels(self, symbol='BTCUSDT', timeframe='15m'):
        """保存標籤"""
        output_dir = Path('outputs/labels')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存標籤到 CSV
        output_path = output_dir / f'{symbol}_{timeframe}_labels.csv'
        
        # 只保存標籤和相關列
        save_columns = ['time', 'open', 'high', 'low', 'close', 'volume',
                       'upper_band', 'lower_band', 'mid_band', 'bb_width', 'label']
        
        save_df = self.df[[col for col in save_columns if col in self.df.columns]].copy()
        save_df.to_csv(output_path, index=False)
        
        logger.info(f'標籤已保存到：{output_path}')
        
        return output_path
    
    def validate_labels_with_backtest(self):
        """
        驗證標籤的準確率
        
        邏輯：在有效反彈點做多（下軌）或做空（上軌），
        檢查後續 5 根 K 棒是否盈利
        """
        logger.info('\n進行回測驗證標籤準確率...')
        logger.info('=' * 60)
        
        valid_lower = self.df[self.df['label'] == 1].index.tolist()
        valid_upper = self.df[self.df['label'] == 2].index.tolist()
        invalid = self.df[self.df['label'] == 0].index.tolist()
        
        if len(valid_lower) == 0 and len(valid_upper) == 0:
            logger.warning('沒有有效反彈樣本進行回測')
            return
        
        # 回測有效反彈（下軌做多）
        profitable_valid_long = 0
        if valid_lower:
            for idx in valid_lower:
                if idx + self.lookahead >= len(self.df):
                    continue
                
                entry_price = self.df.iloc[idx]['close']
                close_price = self.df.iloc[idx + self.lookahead]['close']
                
                pnl = (close_price - entry_price) / entry_price * 100
                
                if pnl > 0:
                    profitable_valid_long += 1
            
            win_rate_long = profitable_valid_long / len(valid_lower) * 100 if valid_lower else 0
            logger.info(f'\n下軌反彈做多（應該上漲）：')
            logger.info(f'  信號數：{len(valid_lower)}')
            logger.info(f'  盈利信號數：{profitable_valid_long}')
            logger.info(f'  勝率：{win_rate_long:.2f}%')
        
        # 回測有效反彈（上軌做空）
        profitable_valid_short = 0
        if valid_upper:
            for idx in valid_upper:
                if idx + self.lookahead >= len(self.df):
                    continue
                
                entry_price = self.df.iloc[idx]['close']
                close_price = self.df.iloc[idx + self.lookahead]['close']
                
                pnl = (entry_price - close_price) / entry_price * 100
                
                if pnl > 0:
                    profitable_valid_short += 1
            
            win_rate_short = profitable_valid_short / len(valid_upper) * 100 if valid_upper else 0
            logger.info(f'\n上軌反彈做空（應該下跌）：')
            logger.info(f'  信號數：{len(valid_upper)}')
            logger.info(f'  盈利信號數：{profitable_valid_short}')
            logger.info(f'  勝率：{win_rate_short:.2f}%')
        
        # 回測無效反彈（應該虧損）
        if invalid:
            losing_invalid = 0
            for idx in invalid:
                if idx + self.lookahead >= len(self.df):
                    continue
                
                entry_price = self.df.iloc[idx]['close']
                close_price = self.df.iloc[idx + self.lookahead]['close']
                
                # 無效反彈應該繼續下跌或上漲（取決於類型）
                touch_type = self._get_touch_type_at_idx(idx)
                
                if touch_type == 'lower':
                    # 下軌無效：應該繼續下跌
                    pnl = (close_price - entry_price) / entry_price * 100
                    if pnl < 0:
                        losing_invalid += 1
                else:
                    # 上軌無效：應該繼續上漲
                    pnl = (entry_price - close_price) / entry_price * 100
                    if pnl < 0:
                        losing_invalid += 1
            
            accuracy_invalid = losing_invalid / len(invalid) * 100 if invalid else 0
            logger.info(f'\n無效反彈判斷準確率：')
            logger.info(f'  信號數：{len(invalid)}')
            logger.info(f'  正確判斷數：{losing_invalid}')
            logger.info(f'  準確率：{accuracy_invalid:.2f}%')
        
        logger.info('=' * 60)
        
        # 整體準確率
        total_signals = len(valid_lower) + len(valid_upper) + len(invalid)
        if total_signals > 0:
            total_correct = profitable_valid_long + profitable_valid_short + losing_invalid
            overall_accuracy = total_correct / total_signals * 100
            logger.info(f'\n整體標籤準確率：{overall_accuracy:.2f}%')
            logger.info(f'  目標：99%')
    
    def _get_touch_type_at_idx(self, idx):
        """獲取指定索引的觸碰類型"""
        for touch in self.touch_points:
            if touch['index'] == idx:
                return touch['type']
        return None
    
    def run_full_pipeline(self, symbol='BTCUSDT', timeframe='15m'):
        """完整流程"""
        logger.info(f'開始為 {symbol}_{timeframe} 創建標籤')
        logger.info('=' * 60)
        
        # 加載數據
        self.load_data(symbol, timeframe)
        
        # 計算 BB
        self.calculate_bb_bands()
        
        # 檢測觸碰
        self.detect_touches()
        
        # 創建標籤
        labels, label_details = self.create_labels()
        
        # 分析特徵
        self.analyze_rebound_characteristics(label_details)
        
        # 保存標籤
        self.save_labels(symbol, timeframe)
        
        # 驗證標籤
        self.validate_labels_with_backtest()
        
        logger.info('流程完成！')
        
        return labels


def main():
    """主函數"""
    Path('logs').mkdir(exist_ok=True)
    Path('outputs/labels').mkdir(parents=True, exist_ok=True)
    
    # 測試單個幣種
    creator = BBTouchLabelCreator(
        bb_period=20,
        bb_std=2,
        touch_threshold=0.05,      # 0.05% 閾值
        lookahead=5,               # 後續 5 根 K 棒
        min_rebound_pct=0.1        # 最小反彈 0.1%
    )
    
    labels = creator.run_full_pipeline('BTCUSDT', '15m')


if __name__ == '__main__':
    main()
