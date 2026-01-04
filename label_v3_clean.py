"""
BB 反彈標籤系統 - 重新設計版本

頙輯：
1. 標記所有觸碰/接近 BB 軌道的 K 棒（使用可配置的閾值）
2. 判斷後續是否有有效反彈（使用高低點或價格變化）
3. 驗證標籤準確率（在有效反彈點交易，後續 5 根 K 棒是否盆利）

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
    """歷司觸碰標籤彥醣器"""
    
    def __init__(self, 
                 bb_period=20,
                 bb_std=2,
                 touch_threshold=0.05,  # 0.05% 的閾值
                 lookahead=5,           # 後續 5 根 K 棒驗證反彈
                 min_rebound_pct=0.1):  # 最小反彈 0.1%
        """
        初始化參數
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