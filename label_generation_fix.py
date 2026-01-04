"""
完整的標籤生成流程修復腳本
適用於 BB Bounce ML V3 專案
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ProfitabilityLabelGenerator:
    """生成 BB Bounce 可交易性標籤"""
    
    def __init__(self, data_dir='data', output_dir='outputs/labels'):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 標籤生成參數
        self.min_bounce_pct = 0.5  # 最小反彈百分比
        self.min_candles_recovery = 5  # 最少恢復蠟燭數
        self.lookback_bars = 50  # 向後看的蠟燭數
        
    def find_csv_files(self):
        """尋找所有 CSV 檔案"""
        csv_files = list(self.data_dir.glob('*_15m.csv'))
        logger.info(f"找到 {len(csv_files)} 個 15m CSV 檔案")
        return csv_files
    
    def generate_labels(self, csv_file):
        """為單一交易對生成標籤"""
        try:
            symbol = csv_file.stem.split('_')[0]  # 例如 BTCUSDT_15m.csv -> BTCUSDT
            
            logger.info(f"正在處理 {symbol}...")
            
            # 讀取數據
            df = pd.read_csv(csv_file)
            
            if df.empty:
                logger.warning(f"{symbol} 資料為空")
                return None
            
            # 確保有必要的列
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_cols):
                logger.warning(f"{symbol} 缺少必要列: {required_cols}")
                return None
            
            # 計算 Bollinger Bands
            df = self._calculate_bollinger_bands(df)
            
            # 生成標籤
            df = self._generate_trade_labels(df, symbol)
            
            # 保存結果
            output_file = self.output_dir / f"{symbol}_15m_profitability_v2.csv"
            df.to_csv(output_file, index=False)
            
            valid_trades = (df['is_profitable'] == 1).sum()
            total_trades = (df['is_profitable'].notna()).sum()
            
            logger.info(f"{symbol} 完成 - 有效交易: {valid_trades}/{total_trades}")
            
            return {
                'symbol': symbol,
                'total_rows': len(df),
                'valid_trades': valid_trades,
                'output_file': str(output_file)
            }
            
        except Exception as e:
            logger.error(f"處理 {csv_file.name} 時出錯: {str(e)}")
            return None
    
    def _calculate_bollinger_bands(self, df, period=20, std_dev=2):
        """計算 Bollinger Bands"""
        df['sma'] = df['close'].rolling(window=period).mean()
        df['std'] = df['close'].rolling(window=period).std()
        df['bb_upper'] = df['sma'] + (df['std'] * std_dev)
        df['bb_lower'] = df['sma'] - (df['std'] * std_dev)
        df['bb_width'] = df['bb_upper'] - df['bb_lower']
        return df
    
    def _generate_trade_labels(self, df, symbol):
        """根據 BB 和反彈邏輯生成交易標籤"""
        df['is_profitable'] = np.nan
        df['bounce_type'] = 'none'
        df['bounce_pct'] = 0.0
        
        for i in range(self.lookback_bars, len(df) - self.min_candles_recovery):
            current_row = df.iloc[i]
            
            # 檢查是否觸碰到下軌
            if current_row['low'] <= current_row['bb_lower'] * 1.01:  # 允許 1% 偏差
                # 檢查之後的反彈
                future_bars = df.iloc[i+1:i+self.min_candles_recovery+1]
                
                if len(future_bars) >= self.min_candles_recovery:
                    min_price = future_bars['low'].min()
                    max_price = future_bars['high'].max()
                    
                    # 計算反彈百分比
                    bounce_pct = ((max_price - min_price) / min_price) * 100
                    
                    # 判斷是否為有效反彈
                    if bounce_pct >= self.min_bounce_pct:
                        df.at[i, 'is_profitable'] = 1
                        df.at[i, 'bounce_type'] = 'lower_bounce'
                        df.at[i, 'bounce_pct'] = bounce_pct
                    else:
                        df.at[i, 'is_profitable'] = 0
                        df.at[i, 'bounce_type'] = 'failed_bounce'
            
            # 檢查是否觸碰到上軌
            elif current_row['high'] >= current_row['bb_upper'] * 0.99:  # 允許 1% 偏差
                future_bars = df.iloc[i+1:i+self.min_candles_recovery+1]
                
                if len(future_bars) >= self.min_candles_recovery:
                    min_price = future_bars['low'].min()
                    max_price = future_bars['high'].max()
                    
                    bounce_pct = ((max_price - min_price) / max_price) * 100
                    
                    if bounce_pct >= self.min_bounce_pct:
                        df.at[i, 'is_profitable'] = 1
                        df.at[i, 'bounce_type'] = 'upper_bounce'
                        df.at[i, 'bounce_pct'] = bounce_pct
                    else:
                        df.at[i, 'is_profitable'] = 0
                        df.at[i, 'bounce_type'] = 'failed_bounce'
        
        return df
    
    def process_all(self):
        """處理所有 CSV 檔案"""
        csv_files = self.find_csv_files()
        
        if not csv_files:
            logger.error(f"在 {self.data_dir} 找不到 CSV 檔案")
            return []
        
        results = []
        for csv_file in csv_files:
            result = self.generate_labels(csv_file)
            if result:
                results.append(result)
        
        # 生成統計報告
        self._print_summary(results)
        
        return results
    
    def _print_summary(self, results):
        """打印摘要"""
        logger.info("\n" + "="*70)
        logger.info("標籤生成完成摘要")
        logger.info("="*70)
        
        total_symbols = len(results)
        total_valid_trades = sum(r['valid_trades'] for r in results)
        total_rows = sum(r['total_rows'] for r in results)
        
        logger.info(f"已處理交易對: {total_symbols}")
        logger.info(f"總行數: {total_rows:,}")
        logger.info(f"有效交易標籤: {total_valid_trades:,}")
        logger.info(f"輸出目錄: {self.output_dir}")
        logger.info("="*70)
        
        # 詳細表
        logger.info("\n詳細統計:")
        logger.info(f"{'交易對':<12} {'行數':>10} {'有效交易':>12} {'成功率':>10}")
        logger.info("-" * 50)
        
        for r in sorted(results, key=lambda x: x['valid_trades'], reverse=True):
            success_rate = (r['valid_trades'] / r['total_rows'] * 100) if r['total_rows'] > 0 else 0
            logger.info(f"{r['symbol']:<12} {r['total_rows']:>10,} {r['valid_trades']:>12} {success_rate:>9.1f}%")


def main():
    logger.info("開始 BB Bounce 標籤生成流程")
    logger.info("="*70)
    
    # 確定資料目錄
    # 如果您的 CSV 檔案在其他位置，請修改這裡
    data_dir = Path('data')
    
    if not data_dir.exists():
        # 如果 data 目錄不存在，嘗試其他常見位置
        alt_locations = [
            Path('.'),  # 當前目錄
            Path('datasets--zongowo111--v2-crypto-ohlcv-data'),
            Path('datasets'),
        ]
        
        for alt_dir in alt_locations:
            if list(alt_dir.glob('*_15m.csv')):
                data_dir = alt_dir
                break
    
    logger.info(f"使用資料目錄: {data_dir}")
    
    # 創建生成器並處理
    generator = ProfitabilityLabelGenerator(
        data_dir=str(data_dir),
        output_dir='outputs/labels'
    )
    
    results = generator.process_all()
    
    if results:
        logger.info("\n標籤生成成功！")
        logger.info("現在您可以執行訓練腳本:")
        logger.info("  python trainbbmodel.py")
    else:
        logger.error("標籤生成失敗。請檢查資料檔案位置。")


if __name__ == '__main__':
    main()
