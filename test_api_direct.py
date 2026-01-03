#!/usr/bin/env python3
"""
直接測試 API 端點
"""

import requests
import json
from datetime import datetime

API_URL = 'http://localhost:5000/predict'

test_cases = [
    {'symbol': 'BTCUSDT', 'timeframe': '15m'},
    {'symbol': 'ETHUSDT', 'timeframe': '15m'},
    {'symbol': 'BTCUSDT', 'timeframe': '1h'},
]

print('\n' + '='*60)
print('BB 反彈 ML 系統 - API 直接測試')
print('='*60)
print(f'時間: {datetime.now().isoformat()}')
print(f'API: {API_URL}')

for test_case in test_cases:
    symbol = test_case['symbol']
    timeframe = test_case['timeframe']
    
    print(f'\n{"-"*60}')
    print(f'測試: {symbol} {timeframe}')
    print(f'{"-"*60}')
    
    try:
        response = requests.post(
            API_URL,
            json=test_case,
            timeout=10
        )
        
        print(f'狀態碼: {response.status_code}')
        
        if response.status_code == 200:
            data = response.json()
            print(f'\n✅ 成功')
            print(f'\nBB 軌道狀態:')
            bb = data['bb_touch']
            print(f'  狀態: {bb["status"]}')
            print(f'  方向: {bb["direction"]}')
            print(f'  距離: {bb["distance_percent"]:.8f}%')
            print(f'  警告等級: {bb["warning_level"]}')
            
            print(f'\n模型預測結果:')
            validity = data['validity']
            volatility = data['volatility']
            
            if validity is None:
                print(f'  有效性: ❌ None (未預測)')
            else:
                print(f'  有效性: ✅ {validity["probability"]:.1f}% ({validity["quality"]})')
            
            if volatility is None:
                print(f'  波動性: ❌ None (未預測)')
            else:
                print(f'  波動性: ✅ {volatility["predicted_vol"]:.2f}x ({volatility["volatility_level"]})')
            
            print(f'\n完整 JSON:')
            print(json.dumps(data, indent=2, ensure_ascii=False))
        else:
            print(f'\n❌ 錯誤')
            print(f'回應: {response.text}')
    
    except Exception as e:
        print(f'\n❌ 連接失敗: {e}')
        print(f'請確認服務正在運行: python realtime_service_v5_simplified.py')

print(f'\n' + '='*60)
print('測試完成')
print('='*60)
