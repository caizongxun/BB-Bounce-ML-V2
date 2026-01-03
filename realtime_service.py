from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import threading
import time
from datetime import datetime
from realtime_detector_v2 import RealtimeBBDetectorV2
import logging

app = Flask(__name__, template_folder="templates")
app.config["SECRET_KEY"] = "your_secret_key_here"
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# 配置日誌
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s')
logger = logging.getLogger(__name__)

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
last_signals = {}
user_selected_symbols = {}  # sid -> set of selected symbols
user_timeframes = {}  # sid -> current timeframe
user_detection_modes = {}  # sid -> detection mode


def fetch_latest_candles(symbols, timeframe="15m"):
    """
    從交易所 API 取得最新 K 線
    注意: 這是假實現，請根據你的實際數據源 (Binance, 本機DB等) 修改
    """
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
    logger.info("[realtime_scan_loop] Started (5-second interval)...")
    
    while scan_active:
        try:
            # 1. 取得最新 K 線
            latest_candles = fetch_latest_candles(
                symbols=bb_detector_v2.symbols,
                timeframe="15m"
            )
            
            # 2. 加入檢測器緩衝區
            for symbol, candles in latest_candles.items():
                if len(candles) > 0:
                    for candle in candles:
                        bb_detector_v2.add_candle(symbol, candle)
            
            # 3. 執行二層掃描 (所有 22 個幣種)
            signals = bb_detector_v2.scan_all(timeframe="15m")
            
            # 4. 推送信號到前端
            for signal in signals:
                symbol = signal["symbol"]
                last_signals[symbol] = signal
                
                # 廣播信號給所有客戶端
                socketio.emit(
                    "realtime_signal",
                    signal,
                    to=None
                )
                
                logger.info(f"[signal] {symbol} {signal['side'].upper()} @ {signal['validity_prob']:.1%} confidence")
            
            # 5. 更新所有幣種狀態
            symbol_states = bb_detector_v2.get_all_symbols_state()
            socketio.emit(
                "symbols_state_update",
                symbol_states,
                to=None
            )
            
            time.sleep(5.0)
        
        except Exception as e:
            logger.error(f"[realtime_scan_loop] Error: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(5.0)


# ========== Flask Routes ==========

@app.route("/")
def index():
    return "BB Bounce ML Realtime Detector V2 - Ready"


@app.route("/detector")
def detector_dashboard():
    return render_template("realtime_dashboard_v2.html")


@app.route("/api/symbols")
def api_symbols():
    return {
        "symbols": bb_detector_v2.symbols,
        "count": len(bb_detector_v2.symbols)
    }


@app.route("/api/symbols/state")
def api_symbols_state():
    return bb_detector_v2.get_all_symbols_state()


@app.route("/api/signals/latest")
def api_signals_latest():
    return {"signals": list(last_signals.values())}


@app.route("/api/symbol/<symbol>")
def api_symbol_state(symbol):
    state = bb_detector_v2.get_symbol_state(symbol)
    return state


# ========== WebSocket Events ==========

@socketio.on("connect")
def handle_connect():
    from flask import request
    sid = request.sid
    logger.info(f"[socket] Client connected: {sid}")
    
    # 初始化該用戶的狀態
    user_selected_symbols[sid] = set()
    user_timeframes[sid] = "15m"
    user_detection_modes[sid] = "all"
    
    emit("connection_response", {
        "status": "connected",
        "symbols": bb_detector_v2.symbols,
        "count": len(bb_detector_v2.symbols),
        "available_models": [
            "volatility_classifier",
            "bb_band_classifier", 
            "validity_validator"
        ]
    })


@socketio.on("disconnect")
def handle_disconnect():
    from flask import request
    sid = request.sid
    logger.info(f"[socket] Client disconnected: {sid}")
    
    # 清理該用戶的狀態
    if sid in user_selected_symbols:
        del user_selected_symbols[sid]
    if sid in user_timeframes:
        del user_timeframes[sid]
    if sid in user_detection_modes:
        del user_detection_modes[sid]


@socketio.on("request_symbol_list")
def handle_request_symbol_list():
    logger.info("[socket] Request symbol list")
    all_states = bb_detector_v2.get_all_symbols_state()
    emit("symbol_list_response", {
        "symbols": bb_detector_v2.symbols,
        "states": all_states,
        "count": len(bb_detector_v2.symbols)
    })


@socketio.on("select_symbol")
def handle_select_symbol(data):
    from flask import request
    sid = request.sid
    symbol = data.get("symbol")
    selected = data.get("selected", False)
    
    if symbol not in bb_detector_v2.symbols:
        emit("error", {"message": f"Unknown symbol: {symbol}"})
        return
    
    if selected:
        user_selected_symbols[sid].add(symbol)
        logger.info(f"[select_symbol] {sid} selected {symbol}")
    else:
        user_selected_symbols[sid].discard(symbol)
        logger.info(f"[select_symbol] {sid} deselected {symbol}")
    
    emit("selection_updated", {
        "symbol": symbol,
        "selected": selected,
        "selected_symbols": list(user_selected_symbols[sid])
    })


@socketio.on("set_timeframe")
def handle_set_timeframe(data):
    from flask import request
    sid = request.sid
    timeframe = data.get("timeframe", "15m")
    
    if timeframe not in ["15m", "1h"]:
        emit("error", {"message": f"Invalid timeframe: {timeframe}"})
        return
    
    user_timeframes[sid] = timeframe
    logger.info(f"[set_timeframe] {sid} set to {timeframe}")
    
    emit("timeframe_updated", {
        "timeframe": timeframe,
        "message": f"時間框架已更新為 {timeframe}"
    })


@socketio.on("set_detection_mode")
def handle_set_detection_mode(data):
    from flask import request
    sid = request.sid
    mode = data.get("mode", "all")
    
    valid_modes = ["all", "bb-only", "high-confidence"]
    if mode not in valid_modes:
        emit("error", {"message": f"Invalid detection mode: {mode}"})
        return
    
    user_detection_modes[sid] = mode
    logger.info(f"[set_detection_mode] {sid} set to {mode}")
    
    emit("detection_mode_updated", {
        "mode": mode,
        "message": f"檢測模式已更新為 {mode}"
    })


@socketio.on("force_refresh")
def handle_force_refresh(data):
    logger.info("[force_refresh] Triggering full scan...")
    try:
        signals = bb_detector_v2.scan_all(timeframe="15m")
        
        for signal in signals:
            last_signals[signal["symbol"]] = signal
            socketio.emit("realtime_signal", signal, to=None)
        
        all_states = bb_detector_v2.get_all_symbols_state()
        socketio.emit("symbols_state_update", all_states, to=None)
        
        logger.info(f"[force_refresh] Found {len(signals)} signals")
        
        emit("force_refresh_response", {
            "status": "success",
            "signals_found": len(signals),
            "timestamp": int(time.time() * 1000)
        })
    
    except Exception as e:
        logger.error(f"[force_refresh] Error: {e}")
        emit("force_refresh_response", {
            "status": "error",
            "error": str(e)
        })


# ========== 應用啟動 ==========

if __name__ == "__main__":
    print("\n" + "="*70)
    print("BB Bounce ML Realtime Detector V2 - Service")
    print("="*70)
    print(f"Monitoring {len(bb_detector_v2.symbols)} symbols")
    print(f"Scan interval: 5 seconds")
    print(f"Available detection models:")
    print(f"  • Volatility Classifier (波動大小分類)")
    print(f"  • BB Band Classifier (上下軌分類)")
    print(f"  • Validity Validator (有效性驗證)")
    print("="*70 + "\n")
    
    # 啟動背景掃描 loop
    scan_thread = threading.Thread(target=realtime_scan_loop, daemon=True)
    scan_thread.start()
    logger.info("[main] Background scan thread started (5s interval)")
    
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
        logger.info("[main] Shutting down...")
        scan_active = False
