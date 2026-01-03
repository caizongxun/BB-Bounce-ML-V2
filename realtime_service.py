from flask import Flask, render_template, send_from_directory
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import threading
import time
import json
from datetime import datetime
from realtime_detector_v2 import RealtimeBBDetectorV2

app = Flask(__name__, template_folder="templates")
app.config["SECRET_KEY"] = "your_secret_key_here"
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# 初始化二層檢測器 V2
bb_detector_v2 = RealtimeBBDetectorV2(
    symbols=[
        "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
        "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "TONUSDT", "LINKUSDT",
        "OPUSDT", "ARBUSDT", "SUIUSDT", "APEUSDT", "MATICUSDT",
        "LTCUSDT", "TRXUSDT", "FTMUSDT", "INJUSDT", "SEIUSDT",
        "TIAUSDT", "ORDIUSDT",
    ],
    model_dir="models",
    device="cpu"
)

# 全域狀態
scan_active = True
last_signals = {}  # symbol -> last signal
symbol_states = {}  # symbol -> state (for UI list)


def fetch_latest_candles(symbols, timeframe="15m"):
    """
    從交易所 API 取得最新 K 線
    
    注意: 這是假實現,請根據你的實際數據源 (Binance, 本機DB等) 修改
    返回格式: {symbol: [candles]}
    
    candle 格式:
    {
        "timestamp": 1735880000000,
        "open": 42500.0,
        "high": 42800.0,
        "low": 42200.0,
        "close": 42600.0,
        "volume": 120.5,
        "bb_upper": 43000.0,
        "bb_middle": 42500.0,
        "bb_lower": 42000.0,
        "rsi": 68.5,
        "adx": 24.1,
        "atr": 400.0
    }
    """
    # TODO: 替換為實際數據源
    # 例如:
    # from binance_connector import spot_client
    # 或從本機資料庫讀取
    
    candles_dict = {}
    for symbol in symbols:
        candles_dict[symbol] = []
    
    return candles_dict


def realtime_scan_loop():
    """
    實時掃描 loop: 每 5 秒執行一次
    
    流程:
    1. 取得最新 K 線
    2. 加入 detector 緩衝區
    3. 執行二層掃描 (所有 22 個幣種)
    4. 推送信號到前端 (WebSocket)
    5. 更新所有幣種狀態
    """
    print("[realtime_scan_loop] Started (5-second interval)...")
    
    while scan_active:
        try:
            # 1. 取得最新 K 線 (實現自己的數據源)
            latest_candles = fetch_latest_candles(
                symbols=bb_detector_v2.symbols,
                timeframe="15m"
            )
            
            # 2. 加入檢測器緩衝區
            for symbol, candles in latest_candles.items():
                if len(candles) > 0:
                    # 加入最新 K 線
                    for candle in candles:
                        bb_detector_v2.add_candle(symbol, candle)
            
            # 3. 執行二層掃描 (所有 22 個幣種)
            signals = bb_detector_v2.scan_all(timeframe="15m")
            
            # 4. 推送信號到前端
            for signal in signals:
                symbol = signal["symbol"]
                last_signals[symbol] = signal
                
                # 廣播信號給所有連接的客戶端
                socketio.emit(
                    "realtime_signal",
                    signal,
                    broadcast=True
                )
                
                print(f"[signal] {symbol} {signal['side']} @ {signal['validity_prob']:.2%}")
            
            # 5. 更新所有幣種狀態 (用於左側幣種列表)
            symbol_states = bb_detector_v2.get_all_symbols_state()
            
            # 廣播所有幣種狀態更新給前端
            socketio.emit(
                "symbols_state_update",
                symbol_states,
                broadcast=True
            )
            
            time.sleep(5.0)  # 每 5 秒掃描一次
        
        except Exception as e:
            print(f"[realtime_scan_loop] Error: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(5.0)


# ========== Flask Routes ==========

@app.route("/")
def index():
    return "BB Bounce ML Realtime Detector V2 - Ready"


@app.route("/detector")
def detector_dashboard():
    """提供實時儀表板"""
    return render_template("realtime_dashboard_v2.html")


@app.route("/api/symbols")
def api_symbols():
    """取得所有監控幣種列表"""
    return {
        "symbols": bb_detector_v2.symbols,
        "count": len(bb_detector_v2.symbols)
    }


@app.route("/api/symbols/state")
def api_symbols_state():
    """取得所有幣種狀態 (用於初始化左側列表)"""
    return bb_detector_v2.get_all_symbols_state()


@app.route("/api/signals/latest")
def api_signals_latest():
    """取得最新信號"""
    return {"signals": list(last_signals.values())}


@app.route("/api/symbol/<symbol>")
def api_symbol_state(symbol):
    """取得指定幣種狀態"""
    state = bb_detector_v2.get_symbol_state(symbol)
    return state


# ========== WebSocket Events ==========

@socketio.on("connect")
def handle_connect():
    print(f"[socket] Client connected")
    
    # 連接時推送所有幣種列表
    emit("connection_response", {
        "status": "connected",
        "symbols": bb_detector_v2.symbols,
        "count": len(bb_detector_v2.symbols)
    })


@socketio.on("disconnect")
def handle_disconnect():
    print(f"[socket] Client disconnected")


@socketio.on("request_symbol_list")
def handle_request_symbol_list():
    """
    前端請求完整幣種列表 (包括所有 22 個幣種)
    用於初始化左側監控幣種列表
    """
    all_states = bb_detector_v2.get_all_symbols_state()
    emit("symbol_list_response", {
        "symbols": bb_detector_v2.symbols,
        "states": all_states,
        "count": len(bb_detector_v2.symbols)
    })


@socketio.on("force_refresh")
def handle_force_refresh(data):
    """
    前端點擊『強制刷新所有幣種』時觸發
    
    強制重新掃描所有幣種一次
    """
    print("[force_refresh] Triggering full scan...")
    try:
        # 執行完整掃描
        signals = bb_detector_v2.scan_all(timeframe="15m")
        
        # 推送所有信號
        for signal in signals:
            last_signals[signal["symbol"]] = signal
            socketio.emit("realtime_signal", signal, broadcast=True)
        
        # 推送所有幣種狀態
        all_states = bb_detector_v2.get_all_symbols_state()
        socketio.emit("symbols_state_update", all_states, broadcast=True)
        
        print(f"[force_refresh] Found {len(signals)} signals")
        
        # 回傳確認
        emit("force_refresh_response", {
            "status": "success",
            "signals_found": len(signals),
            "timestamp": int(time.time() * 1000)
        })
    
    except Exception as e:
        print(f"[force_refresh] Error: {e}")
        emit("force_refresh_response", {
            "status": "error",
            "error": str(e)
        })


@socketio.on("get_symbol_state")
def handle_get_symbol_state(data):
    """
    前端點擊左側幣種時觸發
    取得該幣種的詳細狀態
    """
    symbol = data.get("symbol")
    if symbol:
        state = bb_detector_v2.get_symbol_state(symbol)
        emit("symbol_state", {"symbol": symbol, "state": state})


# ========== 應用啟動 ==========

if __name__ == "__main__":
    print("\n" + "="*60)
    print("BB Bounce ML Realtime Detector V2 - Service")
    print("="*60)
    print(f"Monitoring {len(bb_detector_v2.symbols)} symbols")
    print(f"Scan interval: 5 seconds")
    print(f"Two-layer detection: Classifier (Layer 1) + Validity (Layer 2)")
    print("="*60 + "\n")
    
    # 啟動背景掃描 loop
    scan_thread = threading.Thread(target=realtime_scan_loop, daemon=True)
    scan_thread.start()
    print("[main] Background scan thread started (5s interval)\n")
    
    # 啟動 Flask + SocketIO 服務
    try:
        socketio.run(
            app,
            host="127.0.0.1",
            port=5000,
            debug=False,
            use_reloader=False
        )
    except KeyboardInterrupt:
        print("\n[main] Shutting down...")
        scan_active = False
