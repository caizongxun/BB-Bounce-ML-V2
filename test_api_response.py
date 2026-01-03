#!/usr/bin/env python3
"""
測試 API 返回結果
直接調用 /predict 端點，查看是否返回有效性/波動性預測
"""

import requests
import json
from datetime import datetime

API_URL = 'http://localhost:5000/predict'

test_cases = [
    {'symbol': 'BTCUSDT', 'timeframe': '15m'},
    {'symbol': 'BTCUSDT', 'timeframe': '1h'},
    {'symbol': 'ETHUSDT', 'timeframe': '15m'},
]

print('='*70)
print('BB 反彈 ML 系統 - API 返回測試')
print('='*70)
print(f'\n測試時間: {datetime.now().isoformat()}')
print(f'API 地址: {API_URL}')

for i, test_case in enumerate(test_cases, 1):
    print(f'\n\n{'='*70}')
    print(f'[測試 {i}] {test_case["symbol"]} {test_case["timeframe"]}')
    print('='*70)
    
    try:
        response = requests.post(API_URL, json=test_case, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            
            print(f'✅ 返回成功 (HTTP {response.status_code})')
            print(f'\nBB 接觸狀態:')
            print(f'  狀態: {data["bb_touch"]["status"]}')
            print(f'  方向: {data["bb_touch"]["direction"]}')
            print(f'  距離: {data["bb_touch"]["distance_percent"]:.8f}%')
            print(f'  警告: {data["bb_touch"]["warning_level"]}')
            print(f'  上軌: {data["bb_touch"]["bb_upper"]}')
            print(f'  中線: {data["bb_touch"]["bb_middle"]}')
            print(f'  下軌: {data["bb_touch"]["bb_lower"]}')
            
            # 檢查有效性預測
            print(f'\n有效性預測:')
            if data['validity'] is None:
                print(f'  ❌ null (沒有調用模型)')
            else:
                print(f'  ✅ {data["validity"]}')
            
            # 檢查波動性預測
            print(f'\n波動性預測:')
            if data['volatility'] is None:
                print(f'  ❌ null (沒有調用模型)')
            else:
                print(f'  ✅ {data["volatility"]}')
            
            # 整個返回體
            print(f'\n完整 JSON 返回:')
            print(json.dumps(data, indent=2, ensure_ascii=False))
        else:
            print(f'❌ 返回失敗 (HTTP {response.status_code})')
            print(f'返回內容: {response.text}')
    
    except requests.exceptions.ConnectionError:
        print(f'❌ 連接失敗: 無法連接到 {API_URL}')
        print(f'   請確認後端服務已啟動')
    except Exception as e:
        print(f'❌ 錯誤: {e}')

print(f'\n\n{'='*70}')
print('測試完成')
print('='*70)
print('\n如果有效性/波動性 = null，說明模型沒有被調用')
print('檢查事項：')
print('  1. 後端日誌中是否有 [17特徵] 或 [有效性] 的日誌')
print('  2. warning_level 是否真的是 danger/warning/caution')
print('  3. 後端是否有錯誤拋出')
