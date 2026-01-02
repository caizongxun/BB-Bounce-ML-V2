import os
import pandas as pd
import numpy as np
from pathlib import Path
from huggingface_hub import hf_hub_download, list_repo_files

class CryptoDataLoader:
    def __init__(self, repo_id='zongowo111/v2-crypto-ohlcv-data', cache_dir='./data'):
        self.repo_id = repo_id
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.symbols = [
            'AAVEUSDT', 'ADAUSDT', 'ALGOUSDT', 'ARBUSDT', 'ATOMUSDT',
            'AVAXUSDT', 'BCHUSDT', 'BNBUSDT', 'BTCUSDT', 'DOGEUSDT',
            'DOTUSDT', 'ETCUSDT', 'ETHUSDT', 'FILUSDT', 'LINKUSDT',
            'LTCUSDT', 'MATICUSDT', 'NEARUSDT', 'OPUSDT', 'SOLUSDT',
            'UNIUSDT', 'XRPUSDT'
        ]
        self.timeframes = ['15m', '1h']
    
    def download_symbol_data(self, symbol, timeframe='15m'):
        """
        從 HuggingFace 下載特定幣種的 K 線數據
        """
        try:
            # 檔案路徑
            file_path = f'klines/{symbol}/{symbol.replace("USDT", "").upper()}_{timeframe}.parquet'
            
            print(f'⬇️  下載 {symbol} {timeframe} 數據...')
            
            # 下載
            path = hf_hub_download(
                repo_id=self.repo_id,
                filename=file_path,
                cache_dir=str(self.cache_dir),
                repo_type='dataset'
            )
            
            # 讀取
            df = pd.read_parquet(path)
            
            # 標準化列名
            df.columns = df.columns.str.lower()
            if 'timestamp' in df.columns:
                df['time'] = pd.to_datetime(df['timestamp'])
            elif 'open_time' in df.columns:
                df['time'] = pd.to_datetime(df['open_time'])
            
            print(f'✅ {symbol} {timeframe}: {len(df)} 根 K 棒')
            return df
            
        except Exception as e:
            print(f'❌ {symbol} {timeframe} 下載失敗: {e}')
            return None
    
    def download_all_data(self):
        """
        下載所有幣種的所有時間框數據
        """
        all_data = {}
        
        for symbol in self.symbols:
            all_data[symbol] = {}
            for timeframe in self.timeframes:
                df = self.download_symbol_data(symbol, timeframe)
                if df is not None:
                    all_data[symbol][timeframe] = df
        
        return all_data
    
    def load_cached_data(self, symbol, timeframe='15m'):
        """
        從快取讀取數據（如果已下載過）
        """
        files = list(self.cache_dir.rglob('*.parquet'))
        for file in files:
            if symbol in file.name and timeframe in file.name:
                return pd.read_parquet(file)
        return None


if __name__ == '__main__':
    loader = CryptoDataLoader()
    
    # 下載所有數據
    print('🚀 開始下載所有數據...')
    all_data = loader.download_all_data()
    
    print(f'\n✅ 完成！共下載 {len(all_data)} 種幣種')
    for symbol in all_data:
        print(f'  {symbol}: {list(all_data[symbol].keys())}')
