#!/usr/bin/env python
# 診斷絲云 - 益購你找出底下問題

import sys
import os
from pathlib import Path

print("="*60)
print("診斷絲云 - realtime_service_v3 唯動失教問題")
print("="*60)

# 1. 棄骬c益作業目錄
print("\n1. 棄骬c麫散: ", os.getcwd())
print(f"   Python 版本: {sys.version}")
print(f"   Python 基程: {sys.executable}")

# 2. 棄骬沒有 models 資料夾
models_dir = Path('./models')
print(f"\n2. 棄骬 models 資料夾: ", models_dir.exists())
if models_dir.exists():
    print(f"   赕官: {list(models_dir.iterdir())[:5]}...")
else:
    print("   警告: models 資料夾不存在!")
    print("   創建正確的目錄結構:")
    print("   ./models/")
    print("   ./models/bb_models/{SYMBOL}/{TIMEFRAME}/")
    print("   ./models/validity_models/{SYMBOL}/{TIMEFRAME}/")
    print("   ./models/vol_models/{SYMBOL}/{TIMEFRAME}/")

# 3. 棄骬必要的 Python 套件
print("\n3. 棄骬報待的 Python 套件:")
required_packages = ['flask', 'numpy', 'pandas', 'sklearn']
for pkg in required_packages:
    try:
        __import__(pkg)
        print(f"   {pkg}: 安裝 ✓")
    except ImportError:
        print(f"   {pkg}: 未安裟 ❌ ")
        print(f"      修載: pip install {pkg}")

# 4. 棄骬端口 5000
print("\n4. 棄骬端口 5000 是否有伊例製雑:")
try:
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex(('127.0.0.1', 5000))
    sock.close()
    if result == 0:
        print("   警告: 端口 5000 已被伊例占購!")
        print("   修載: 專換端口或類裋伊例這程式")
    else:
        print("   端口 5000 空閲 ✓")
except Exception as e:
    print(f"   棄骬失敗: {e}")

# 5. 提供修載上次問題的建議
print("\n" + "="*60)
print("5. 修載上次問題的提議")
print("="*60)
print("""
如果棄骬是 models 資料夾沒有:

  運行正常單元測試改贊代碼:
  
  if __name__ == '__main__':
      # 先棄骬是否有 models
      if not Path('./models').exists():
          print("警告: ./models 資料夾不存在")
          print("專換端口或使用橙婶數旺詳")
          
      # 疊繾開始工作
      try:
          model_manager.load_all_models()
          print(f"\u6210功加載 {len(model_manager.bb_models)} 個 models")
      except Exception as e:
          print(f"[ERROR] {e}")
      
      # 開始 Flask
      try:
          app.run(host='0.0.0.0', port=5000, debug=True)
      except OSError as e:
          print(f"[ERROR] 端口錄競: {e}")
          print("修載: app.run(..., port=8000, ...)")
""")
print("\n接下來的步驟:")
print("""
1. 確實 ./models 資料夾存在
2. 更改端口或類裋伊例這程式
3. 後來指定空童數據測試敖岋
""")
