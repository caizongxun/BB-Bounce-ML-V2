"""
BB反彈ML系統 V3 - 完整測試套件
頑毒検查系統的所有功能
"""

import requests
import json
from datetime import datetime
from typing import Dict, List, Any


class Colors:
    """ANSI風格紅緑樹輸出"""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class TestRunner:
    """測試運行器"""
    
    def __init__(self, base_url: str = 'http://localhost:5000'):
        self.base_url = base_url
        self.results = {
            'passed': 0,
            'failed': 0,
            'errors': []
        }
        self.test_symbol = 'BTCUSDT'
        self.test_timeframe = '15m'
        self.test_ohlcv = {
            'open': 45000,
            'high': 45500,
            'low': 44900,
            'close': 45200,
            'volume': 1000000
        }
    
    def print_header(self, title: str):
        """列印測試段落掌"""
        print(f"\n{Colors.HEADER}{Colors.BOLD}─ {title} ─{Colors.ENDC}")
    
    def print_success(self, message: str):
        """列印成功訊息"""
        print(f"{Colors.GREEN}OK{Colors.ENDC} {message}")
    
    def print_error(self, message: str):
        """列印錯誤訊息"""
        print(f"{Colors.RED}FAIL{Colors.ENDC} {message}")
    
    def print_info(self, message: str):
        """列印資訊訊息"""
        print(f"{Colors.CYAN}INFO{Colors.ENDC} {message}")
    
    def print_value(self, key: str, value: Any):
        """列印偏值對"""
        print(f"{Colors.YELLOW}{key}{Colors.ENDC}: {value}")
    
    def test_health_check(self):
        """測試：健康検查"""
        self.print_header("測試 1: 健康検查")
        
        try:
            response = requests.get(f'{self.base_url}/health')
            response.raise_for_status()
            
            data = response.json()
            
            # 棲棄測試
            assert response.status_code == 200, f"接收代碼需為200，實際為{response.status_code}"
            assert 'status' in data, "回應应當包含'status'"
            assert data['status'] == 'ok', f"狀態需為'ok'，實際為{data['status']}"
            assert 'models_loaded' in data, "回應应當包含'models_loaded'"
            
            # 顯示細節
            self.print_success("健康検查通過")
            self.print_value("狀態", data['status'])
            self.print_value("BB模型數", data['models_loaded']['bb_models'])
            self.print_value("Validity模型數", data['models_loaded']['validity_models'])
            self.print_value("Vol模型數", data['models_loaded']['vol_models'])
            
            self.results['passed'] += 1
            return True
        
        except Exception as e:
            self.print_error(f健康検查失敗: {str(e)}".strip())
            self.results['failed'] += 1
            self.results['errors'].append(str(e))
            return False
    
    def test_single_prediction(self):
        """測試：單個預測"""
        self.print_header("測試 2: 單個預測")
        
        try:
            response = requests.post(
                f'{self.base_url}/predict',
                json={
                    'symbol': self.test_symbol,
                    'timeframe': self.test_timeframe,
                    'ohlcv': self.test_ohlcv
                }
            )
            response.raise_for_status()
            
            data = response.json()
            
            # 棲棄測試
            assert response.status_code == 200, f"接收代碼需為200"
            assert 'symbol' in data, "缺少 'symbol' 似去"
            assert 'signal' in data, "缺少 'signal' 似去"
            assert 'bb_touch' in data, "缺少 'bb_touch' 似去"
            
            # 顯示細節
            self.print_success(f隨接 {self.test_symbol} 預測成功".strip())
            self.print_value("幣種", data['symbol'])
            self.print_value("時框", data['timeframe'])
            self.print_value("交易信號", data['signal'])
            self.print_value("信心度", f"{data.get('confidence', 0)*100:.1f}%")
            
            if data.get('bb_touch'):
                self.print_value("BB觸厬", data['bb_touch']['touched'])
            
            self.results['passed'] += 1
            return True
        
        except Exception as e:
            self.print_error(f單個預測失敗: {str(e)}".strip())
            self.results['failed'] += 1
            self.results['errors'].append(str(e))
            return False
    
    def test_batch_prediction(self):
        """測試：批量預測"""
        self.print_header("測試 3: 批量預測")
        
        try:
            symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']
            ohlcv_data = {symbol: self.test_ohlcv for symbol in symbols}
            
            response = requests.post(
                f'{self.base_url}/predict_batch',
                json={
                    'symbols': symbols,
                    'timeframe': self.test_timeframe,
                    'ohlcv_data': ohlcv_data
                }
            )
            response.raise_for_status()
            
            data = response.json()
            
            # 棲棄測試
            assert response.status_code == 200, f"接收代碼需為200"
            assert 'results' in data, "缺少 'results' 項目"
            assert isinstance(data['results'], list), "'results' 應略是一個清底"
            
            # 顯示細節
            self.print_success(f批量預測成功 - {len(data['results'])} 結果".strip())
            self.print_value("批次求範囲", f"{len(symbols)}個幣種")
            self.print_value("次起ヨ了成果數", len(data['results']))
            
            if data['results']:
                self.print_value("第一結果信號", data['results'][0].get('signal', 'N/A'))
            
            self.results['passed'] += 1
            return True
        
        except Exception as e:
            self.print_error(f批量預測失敗: {str(e)}".strip())
            self.results['failed'] += 1
            self.results['errors'].append(str(e))
            return False
    
    def test_response_structure(self):
        """測試：回應結構"""
        self.print_header("測試 4: 回應結構驗證")
        
        try:
            response = requests.post(
                f'{self.base_url}/predict',
                json={
                    'symbol': self.test_symbol,
                    'timeframe': self.test_timeframe,
                    'ohlcv': self.test_ohlcv
                }
            )
            response.raise_for_status()
            
            data = response.json()
            
            # 驗證必需似去
            required_fields = ['symbol', 'timeframe', 'signal', 'confidence']
            for field in required_fields:
                assert field in data, f"缺少必需似去: {field}"
            
            # 驗證BB觸厬的三層結構
            if data.get('bb_touch'):
                bb_fields = ['touched', 'touch_type', 'confidence']
                for field in bb_fields:
                    assert field in data['bb_touch'], f"bb_touch缺少: {field}"
            
            # 驗證有效性的三層結構
            if data.get('validity'):
                validity_fields = ['valid', 'probability', 'quality']
                for field in validity_fields:
                    assert field in data['validity'], f"validity缺少: {field}"
            
            # 驗證波動性的三層結構
            if data.get('volatility'):
                vol_fields = ['predicted_vol', 'will_expand', 'expansion_strength']
                for field in vol_fields:
                    assert field in data['volatility'], f"volatility缺少: {field}"
            
            self.print_success("回應結構稽合校驗退出")
            self.print_value("必需似去", ', '.join(required_fields))
            self.print_value("次起層數", 3)
            
            self.results['passed'] += 1
            return True
        
        except AssertionError as e:
            self.print_error(f結構驗證失敗: {str(e)}".strip())
            self.results['failed'] += 1
            self.results['errors'].append(str(e))
            return False
    
    def test_signal_distribution(self):
        """測試：信號分布分析"""
        self.print_header("測試 5: 信號統計分析")
        
        try:
            # 騎機卻詰關锧16種不同修改數詡
            signal_counts = {}
            confidence_sum = 0
            test_count = 5
            
            for i in range(test_count):
                test_ohlcv = {
                    'open': 45000 + i*100,
                    'high': 45500 + i*100,
                    'low': 44900 + i*100,
                    'close': 45200 + i*100,
                    'volume': 1000000 + i*10000
                }
                
                response = requests.post(
                    f'{self.base_url}/predict',
                    json={
                        'symbol': self.test_symbol,
                        'timeframe': self.test_timeframe,
                        'ohlcv': test_ohlcv
                    }
                )
                
                if response.status_code == 200:
                    data = response.json()
                    signal = data.get('signal', 'UNKNOWN')
                    signal_counts[signal] = signal_counts.get(signal, 0) + 1
                    confidence_sum += data.get('confidence', 0)
            
            # 顯示統計情報
            self.print_success(f准尝您三次試驗求胨 - 統計 {sum(signal_counts.values())} 次次達")
            
            for signal, count in sorted(signal_counts.items()):
                percentage = (count / sum(signal_counts.values())) * 100
                self.print_value(f信號 {signal}", f"{count} 次 ({percentage:.1f}%)")
            
            if sum(signal_counts.values()) > 0:
                avg_confidence = confidence_sum / sum(signal_counts.values())
                self.print_value("平均信心度", f"{avg_confidence*100:.1f}%")
            
            self.results['passed'] += 1
            return True
        
        except Exception as e:
            self.print_error(f信號統計分析失敗: {str(e)}".strip())
            self.results['failed'] += 1
            self.results['errors'].append(str(e))
            return False
    
    def run_all_tests(self):
        """運行所有測試"""
        print(f"{Colors.BOLD}{Colors.HEADER}"
              f"\n{'='*60}\n"
              f"  BB反彈ML系統 V3 - 完整測試套件\n"
              f"  {'='*60}\n{Colors.ENDC}")
        
        # 運行測試
        self.test_health_check()
        self.test_single_prediction()
        self.test_batch_prediction()
        self.test_response_structure()
        self.test_signal_distribution()
        
        # 顯示求和統計
        self.print_summary()
    
    def print_summary(self):
        """顯示測試統計"""
        print(f"\n{Colors.BOLD}{Colors.HEADER}"
              f"{'='*60}\n"
              f"  測試統計\n"
              f"  {'='*60}\n{Colors.ENDC}")
        
        total = self.results['passed'] + self.results['failed']
        
        if self.results['failed'] == 0:
            status_color = Colors.GREEN
            status_text = "ALL TESTS PASSED"
        else:
            status_color = Colors.RED
            status_text = f"SOME TESTS FAILED"
        
        print(f"{Colors.BOLD}測試統計{Colors.ENDC}")
        print(f"{Colors.GREEN}PASS{Colors.ENDC}: {self.results['passed']}/{total}")
        print(f"{Colors.RED}FAIL{Colors.ENDC}: {self.results['failed']}/{total}")
        print(f"\n{status_color}{status_text}{Colors.ENDC}")
        
        if self.results['errors']:
            print(f"\n{Colors.BOLD}錯誤詳述{Colors.ENDC}")
            for i, error in enumerate(self.results['errors'], 1):
                print(f"{Colors.RED}{i}. {error}{Colors.ENDC}")
        
        print(f"\n{Colors.BOLD}{'='*60}{Colors.ENDC}\n")


if __name__ == '__main__':
    runner = TestRunner()
    runner.run_all_tests()
