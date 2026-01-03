#!/usr/bin/env python3
"""
BB Bounce ML V2 - 依賴自動安裝腳本

使用方法:
    python install_dependencies.py
"""

import subprocess
import sys
import platform


def print_header(title):
    print("\n" + "="*60)
    print(title.center(60))
    print("="*60)


def print_section(title):
    print(f"\n{title}")
    print("-" * 60)


def run_command(cmd, description):
    print(f"\n► {description}...")
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if result.returncode == 0:
            print(f"✓ {description} 成功")
            return True
        else:
            print(f"✗ {description} 失敗")
            if result.stderr:
                print(f"  錯誤: {result.stderr[:200]}")
            return False
    except subprocess.TimeoutExpired:
        print(f"✗ {description} 超時")
        return False
    except Exception as e:
        print(f"✗ {description} 出錯: {e}")
        return False


def check_python_version():
    """檢查 Python 版本"""
    print_section("檢查 Python 版本")
    
    version = sys.version_info
    print(f"當前 Python: {version.major}.{version.minor}.{version.micro}")
    
    if version.major >= 3 and version.minor >= 8:
        print("✓ Python 版本符合要求 (>=3.8)")
        return True
    else:
        print("✗ Python 版本過低，需要 3.8 或更高")
        return False


def install_basic_dependencies():
    """安裝基礎依賴"""
    print_section("安裝基礎依賴")
    
    basic_packages = [
        ("flask", "Flask (Web 框架)"),
        ("flask-socketio", "Flask-SocketIO (實時通信)"),
        ("flask-cors", "Flask-CORS (跨域支持)"),
        ("numpy", "NumPy (數值計算)"),
        ("scikit-learn", "scikit-learn (機器學習)"),
    ]
    
    success = True
    for package, description in basic_packages:
        cmd = f"{sys.executable} -m pip install {package}"
        if not run_command(cmd, f"安裝 {description}"):
            success = False
    
    return success


def install_data_fetcher_dependencies():
    """安裝數據源依賴"""
    print_section("安裝數據源依賴")
    
    print("\n選擇要安裝的數據源:")
    print("  1. 完整安裝 (Binance + yfinance) - 推薦")
    print("  2. 僅安裝 Binance US")
    print("  3. 僅安裝 yfinance")
    print("  4. 跳過")
    
    choice = input("\n請選擇 (1-4): ").strip()
    
    success = True
    
    if choice in ["1", "2"]:
        cmd = f"{sys.executable} -m pip install python-binance"
        if not run_command(cmd, "安裝 python-binance"):
            success = False
    
    if choice in ["1", "3"]:
        cmd = f"{sys.executable} -m pip install yfinance"
        if not run_command(cmd, "安裝 yfinance"):
            success = False
    
    if choice == "4":
        print("\n⚠ 警告: 未安裝數據源，系統將無法取得實時數據")
        success = False
    
    return success


def verify_installation():
    """驗證安裝"""
    print_section("驗證安裝")
    
    packages_to_check = [
        ("flask", "Flask"),
        ("flask_socketio", "Flask-SocketIO"),
        ("flask_cors", "Flask-CORS"),
        ("numpy", "NumPy"),
        ("sklearn", "scikit-learn"),
    ]
    
    all_ok = True
    
    for package, description in packages_to_check:
        try:
            __import__(package)
            print(f"✓ {description} 已安裝")
        except ImportError:
            print(f"✗ {description} 未安裝")
            all_ok = False
    
    # 檢查數據源
    print("\n數據源檢查:")
    try:
        import binance
        print(f"✓ Binance US (python-binance) 已安裝")
    except ImportError:
        print(f"✗ Binance US (python-binance) 未安裝")
    
    try:
        import yfinance
        print(f"✓ yfinance 已安裝")
    except ImportError:
        print(f"✗ yfinance 未安裝")
    
    return all_ok


def test_data_fetcher():
    """測試數據獲取器"""
    print_section("測試數據獲取器")
    
    try:
        from data_fetcher import DataFetcher
        
        print("初始化數據獲取器...")
        fetcher = DataFetcher(
            preferred_source="binance",
            fallback_to_yfinance=True
        )
        
        if not fetcher.is_available():
            print("\n✗ 警告: 沒有可用的數據源")
            print("  請確保至少安裝了以下之一:")
            print("    - python-binance")
            print("    - yfinance")
            return False
        
        print("✓ 數據獲取器初始化成功")
        
        print("\n測試獲取數據...")
        test_symbols = ["BTCUSDT", "ETHUSDT"]
        data = fetcher.get_klines(test_symbols, "15m", 5)
        
        success_count = 0
        for symbol, candles in data.items():
            if candles:
                print(f"✓ {symbol}: 成功獲取 {len(candles)} 根 K 線")
                success_count += 1
            else:
                print(f"✗ {symbol}: 無數據")
        
        if success_count > 0:
            print("\n✓ 數據獲取測試成功")
            return True
        else:
            print("\n✗ 數據獲取測試失敗")
            return False
    
    except Exception as e:
        print(f"✗ 數據獲取器測試失敗: {e}")
        return False


def main():
    print_header("BB Bounce ML V2 - 依賴安裝向導")
    
    # 檢查 Python 版本
    if not check_python_version():
        print("\n✗ 安裝失敗: Python 版本不符合")
        sys.exit(1)
    
    # 安裝基礎依賴
    if not install_basic_dependencies():
        print("\n⚠ 部分基礎依賴安裝失敗")
    
    # 安裝數據源依賴
    if not install_data_fetcher_dependencies():
        print("\n⚠ 數據源依賴安裝不完整")
    
    # 驗證安裝
    verify_installation()
    
    # 測試數據獲取器
    test_data_fetcher()
    
    print_header("安裝完成")
    print("""
    
    ✓ 安裝完成！
    
    下一步:
    
    1. 啟動服務:
       python realtime_service.py
    
    2. 打開儀表板:
       http://localhost:5000/detector
    
    3. 查看文檔:
       - 數據源集成指南.md
       - 完整操作指南.md
    
    祝你使用愉快！ 🚀
    """)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n已取消安裝")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n✗ 錯誤: {e}")
        sys.exit(1)
