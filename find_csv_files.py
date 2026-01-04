"""
診斷腳本：尋找所有 CSV 檔案並列出位置
"""

from pathlib import Path
import os

def find_all_csv_files():
    """遞迴尋找所有 CSV 檔案"""
    root = Path('.')
    csv_files = list(root.rglob('*_15m.csv'))
    csv_files.extend(root.rglob('*_1h.csv'))
    
    return sorted(set(csv_files))

def main():
    print("="*70)
    print("斐尋 CSV 檔案...")
    print("="*70)
    
    csv_files = find_all_csv_files()
    
    if not csv_files:
        print("未找到任何 CSV 檔案！")
        print("\n嘗試以下步驟:")
        print("1. 確認下載完成（根據您的日誌，應該成功了）")
        print("2. 検查這些位置:")
        print("   - ./data/")
        print("   - ./datasets--zongowo111--v2-crypto-ohlcv-data/")
        print("   - 當前目錄")
        return
    
    print(f"\n找到 {len(csv_files)} 個 CSV 檔案\n")
    
    # 按時間框組織
    tf_15m = [f for f in csv_files if '_15m' in f.name]
    tf_1h = [f for f in csv_files if '_1h' in f.name]
    
    print(f"15m 時間框: {len(tf_15m)} 個")
    print(f"1h 時間框: {len(tf_1h)} 個\n")
    
    print("詳細位置:")
    print("-" * 70)
    
    for csv_file in sorted(csv_files)[:20]:  # 顯示前 20 個
        size_mb = csv_file.stat().st_size / (1024*1024)
        print(f"{csv_file.name:<30} {size_mb:>8.2f}MB  {csv_file.parent}")
    
    if len(csv_files) > 20:
        print(f"... 還有 {len(csv_files) - 20} 個檔案")
    
    # 找出基礎目錄
    print("\n" + "="*70)
    print("建議的資料目錄位置:")
    print("="*70)
    
    # 找出最常見的父目錄
    parent_dirs = {}
    for csv_file in csv_files:
        parent = str(csv_file.parent)
        parent_dirs[parent] = parent_dirs.get(parent, 0) + 1
    
    for parent, count in sorted(parent_dirs.items(), key=lambda x: x[1], reverse=True):
        print(f"{parent:<50} ({count} 個檔案)")
    
    # 建議命令
    print("\n" + "="*70)
    print("建議的執行步驟:")
    print("="*70)
    
    if tf_15m:
        base_dir = tf_15m[0].parent
        print(f"\n1. 設定資料目錄為: {base_dir}")
        print(f"\n2. 修改 label_generation_fix.py:")
        print(f"   data_dir = Path('{base_dir}')")
        print(f"\n3. 執行標籤生成:")
        print(f"   python label_generation_fix.py")

if __name__ == '__main__':
    main()
