"""
BB Bounce ML - 多源數據獲取器

支持:
1. Binance US (推薦 - 低延遲, 高頻率)
2. yfinance (備用 - 無需 API KEY)
3. 本機數據庫 (高級 - 自定義)
"""

import os
import time
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class BinanceUSDataFetcher:
    """
    Binance US 數據獲取器 - 推薦選項
    
    優點:
    - 實時數據, 最小延遲
    - 支持 WebSocket 流 (高頻)
    - 免費, 無需 API KEY (公開端點)
    - 高度可靠
    
    安裝: pip install python-binance
    """
    
    def __init__(self):
        self.client = None
        self.initialized = False
        self._init_client()
    
    def _init_client(self):
        """初始化 Binance US 客戶端"""
        try:
            from binance.client import Client
            from binance.exceptions import BinanceAPIException
            
            # Binance US 公開端點 (無需 API KEY)
            self.Client = Client
            self.BinanceAPIException = BinanceAPIException
            self.client = Client(api_key='', api_secret='')
            self.initialized = True
            logger.info("[BinanceUS] Client initialized successfully")
        except ImportError:
            logger.warning("[BinanceUS] python-binance not installed")
            logger.warning("Install with: pip install python-binance")
            self.initialized = False
        except Exception as e:
            logger.error(f"[BinanceUS] Initialization failed: {e}")
            self.initialized = False
    
    def get_klines(
        self,
        symbols: List[str],
        timeframe: str = "15m",
        limit: int = 100
    ) -> Dict[str, List[Dict]]:
        """
        獲取 K 線數據
        
        Args:
            symbols: ["BTCUSDT", "ETHUSDT", ...]
            timeframe: "15m", "1h", "4h", "1d"
            limit: 返回多少根 K 線 (推薦 100+ 用來計算指標)
        
        Returns:
            {
                "BTCUSDT": [
                    {
                        "timestamp": 1735880000000,
                        "open": 42500.0,
                        "high": 42800.0,
                        "low": 42200.0,
                        "close": 42600.0,
                        "volume": 120.5,
                    },
                    ...
                ],
                "ETHUSDT": [...],
            }
        """
        if not self.initialized:
            return {}
        
        result = {}
        
        for symbol in symbols:
            try:
                # 從 Binance US 獲取最新 K 線
                klines = self.client.get_klines(
                    symbol=symbol,
                    interval=self._format_interval(timeframe),
                    limit=limit
                )
                
                # 解析 K 線數據
                candles = []
                for kline in klines:
                    candle = {
                        "timestamp": int(kline[0]),
                        "open": float(kline[1]),
                        "high": float(kline[2]),
                        "low": float(kline[3]),
                        "close": float(kline[4]),
                        "volume": float(kline[7]),  # Quote asset volume
                    }
                    candles.append(candle)
                
                result[symbol] = candles
                logger.debug(f"[BinanceUS] Fetched {len(candles)} candles for {symbol}")
            
            except Exception as e:
                logger.error(f"[BinanceUS] Failed to fetch {symbol}: {e}")
                result[symbol] = []
        
        return result
    
    @staticmethod
    def _format_interval(timeframe: str) -> str:
        """轉換時間框架格式"""
        mapping = {
            "15m": "15m",
            "1h": "1h",
            "4h": "4h",
            "1d": "1d",
            "1w": "1w",
        }
        return mapping.get(timeframe, "15m")


class YFinanceDataFetcher:
    """
    yfinance 數據獲取器 - 備用選項
    
    優點:
    - 無需 API KEY
    - 簡單易用
    - 支持多種資產
    
    缺點:
    - 延遲較高 (2-5秒)
    - 速率限制
    - 不支持實時數據流
    
    安裝: pip install yfinance
    """
    
    def __init__(self):
        self.yf = None
        self.initialized = False
        self._init_yfinance()
    
    def _init_yfinance(self):
        """初始化 yfinance"""
        try:
            import yfinance as yf
            self.yf = yf
            self.initialized = True
            logger.info("[yfinance] Initialized successfully")
        except ImportError:
            logger.warning("[yfinance] Not installed")
            logger.warning("Install with: pip install yfinance")
            self.initialized = False
    
    def get_klines(
        self,
        symbols: List[str],
        timeframe: str = "15m",
        limit: int = 100
    ) -> Dict[str, List[Dict]]:
        """
        獲取 K 線數據 (yfinance 版本)
        
        注意: yfinance 自動轉換 symbol 格式 (BTCUSDT -> BTC-USD)
        """
        if not self.initialized:
            return {}
        
        result = {}
        
        for symbol in symbols:
            try:
                # yfinance 需要轉換符號格式
                yf_symbol = symbol.replace("USDT", "-USD")
                
                # 計算所需數據點數
                end_date = datetime.now()
                start_date = end_date - timedelta(days=30)  # 取 30 天數據
                
                # 下載數據
                df = self.yf.download(
                    yf_symbol,
                    start=start_date,
                    end=end_date,
                    interval=self._format_interval(timeframe),
                    progress=False,
                    prepost=False
                )
                
                if df.empty:
                    logger.warning(f"[yfinance] No data for {symbol}")
                    result[symbol] = []
                    continue
                
                # 轉換為標準格式
                candles = []
                for idx, row in df.iterrows():
                    candle = {
                        "timestamp": int(idx.timestamp() * 1000),
                        "open": float(row["Open"]),
                        "high": float(row["High"]),
                        "low": float(row["Low"]),
                        "close": float(row["Close"]),
                        "volume": float(row["Volume"]),
                    }
                    candles.append(candle)
                
                # 只保留最新 limit 根
                result[symbol] = candles[-limit:]
                logger.debug(f"[yfinance] Fetched {len(result[symbol])} candles for {symbol}")
            
            except Exception as e:
                logger.error(f"[yfinance] Failed to fetch {symbol}: {e}")
                result[symbol] = []
        
        return result
    
    @staticmethod
    def _format_interval(timeframe: str) -> str:
        """轉換時間框架格式"""
        mapping = {
            "15m": "15m",
            "1h": "1h",
            "4h": "4h",
            "1d": "1d",
            "1w": "1wk",
        }
        return mapping.get(timeframe, "15m")


class DataFetcher:
    """
    統一數據獲取器 - 自動選擇最佳源
    
    優先級:
    1. Binance US (如果可用)
    2. yfinance (備用)
    3. 本機數據庫 (如果配置)
    """
    
    def __init__(
        self,
        preferred_source: str = "binance",
        fallback_to_yfinance: bool = True
    ):
        """
        Args:
            preferred_source: "binance" 或 "yfinance"
            fallback_to_yfinance: 如果首選源失敗，是否回退到 yfinance
        """
        self.preferred_source = preferred_source
        self.fallback_to_yfinance = fallback_to_yfinance
        self.binance_fetcher = None
        self.yfinance_fetcher = None
        
        # 初始化數據源
        if preferred_source == "binance":
            self.binance_fetcher = BinanceUSDataFetcher()
            if fallback_to_yfinance and not self.binance_fetcher.initialized:
                logger.info("[DataFetcher] Binance not available, initializing yfinance as fallback")
                self.yfinance_fetcher = YFinanceDataFetcher()
        else:
            self.yfinance_fetcher = YFinanceDataFetcher()
    
    def get_klines(
        self,
        symbols: List[str],
        timeframe: str = "15m",
        limit: int = 100
    ) -> Dict[str, List[Dict]]:
        """
        從首選源獲取 K 線數據
        
        自動回退到備用源如果首選源失敗
        """
        # 嘗試首選源
        if self.binance_fetcher and self.binance_fetcher.initialized:
            try:
                result = self.binance_fetcher.get_klines(symbols, timeframe, limit)
                if any(result.values()):  # 至少有一個幣種成功
                    return result
            except Exception as e:
                logger.warning(f"[DataFetcher] Binance fetch failed: {e}")
        
        # 回退到 yfinance
        if self.yfinance_fetcher and self.yfinance_fetcher.initialized:
            try:
                return self.yfinance_fetcher.get_klines(symbols, timeframe, limit)
            except Exception as e:
                logger.error(f"[DataFetcher] yfinance fetch failed: {e}")
        
        # 都失敗
        logger.error("[DataFetcher] All data sources failed")
        return {sym: [] for sym in symbols}
    
    def is_available(self) -> bool:
        """檢查是否有可用的數據源"""
        if self.binance_fetcher and self.binance_fetcher.initialized:
            return True
        if self.yfinance_fetcher and self.yfinance_fetcher.initialized:
            return True
        return False


if __name__ == "__main__":
    # 測試數據獲取器
    print("\n=== BB Bounce ML - Data Fetcher Test ===")
    
    # 測試符號
    test_symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]
    
    # 初始化獲取器
    fetcher = DataFetcher(preferred_source="binance", fallback_to_yfinance=True)
    
    if not fetcher.is_available():
        print("\nError: No data sources available")
        print("\nInstall required packages:")
        print("  pip install python-binance")
        print("  pip install yfinance")
        exit(1)
    
    # 獲取數據
    print("\nFetching data...")
    klines = fetcher.get_klines(
        symbols=test_symbols,
        timeframe="15m",
        limit=10
    )
    
    # 顯示結果
    for symbol, candles in klines.items():
        if candles:
            latest = candles[-1]
            print(f"\n{symbol}:")
            print(f"  Close: ${latest['close']:.2f}")
            print(f"  Volume: {latest['volume']:.2f}")
            print(f"  Latest: {datetime.fromtimestamp(latest['timestamp']/1000)}")
        else:
            print(f"\n{symbol}: No data")
    
    print("\n" + "="*50)
