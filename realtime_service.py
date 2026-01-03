from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import threading
import time
from datetime import datetime
from realtime_detector_v2 import RealtimeBBDetectorV2
from data_fetcher import DataFetcher
import logging
import os

app = Flask(__name__, template_folder="templates")
app.config["SECRET_KEY"] = "your_secret_key_here"
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# 配置日誌
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s')
logger = logging.getLogger(__name__)

# 初始化數據獲取器
data_fetcher = DataFetcher(
    preferred_source="binance",
    fallback_to_yfinance=True
)

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
last_candles_fetch_time = 0
fetch_interval = 5  # 每 5 秒從交易所取得 K 線


def calculate_technical_indicators(candles):
    """
    計算技術指標 (RSI, ADX, ATR, BB)
    
    添加到每個 candle: rsi, adx, atr, bb_upper, bb_middle, bb_lower, bb_width
    """
    if len(candles) < 20:
        return candles
    
    try:
        import numpy as np
        
        closes = np.array([c.get("close", 0) for c in candles])
        highs = np.array([c.get("high", 0) for c in candles])
        lows = np.array([c.get("low", 0) for c in candles])
        
        # ===== RSI (Relative Strength Index) =====
        deltas = np.diff(closes)
        seed = deltas[:1]
        up = seed[seed >= 0].sum() / 1 if len(seed[seed >= 0]) > 0 else 0
        down = -seed[seed < 0].sum() / 1 if len(seed[seed < 0]) > 0 else 0
        
        rs = np.zeros_like(closes)
        rs[:1] = 100. if down != 0 else 0.
        
        for i in range(1, len(closes)):
            delta = deltas[i - 1]
            if delta > 0:
                upval = delta
                downval = 0.
            else:
                upval = 0.
                downval = -delta
            
            up = (up * 13. + upval) / 14.
            down = (down * 13. + downval) / 14.
            
            rs[i] = 100. if down != 0 else (100. if up != 0 else 0.)
        
        rsi = np.where(rs != 0, 100. - 100. / (1. + up / down), 0.)
        
        # ===== Bollinger Bands =====
        sma = np.convolve(closes, np.ones(20) / 20, mode='valid')
        sma = np.pad(sma, (len(closes) - len(sma), 0), 'edge')
        
        std = np.array([np.std(closes[max(0, i-19):i+1]) for i in range(len(closes))])
        bb_upper = sma + 2 * std
        bb_lower = sma - 2 * std
        bb_middle = sma
        bb_width = bb_upper - bb_lower
        
        # ===== ATR (Average True Range) =====
        tr = np.zeros_like(closes)
        for i in range(len(closes)):
            high_low = highs[i] - lows[i]
            high_close = abs(highs[i] - closes[i-1]) if i > 0 else high_low
            low_close = abs(lows[i] - closes[i-1]) if i > 0 else high_low
            tr[i] = max(high_low, high_close, low_close)
        
        atr = np.convolve(tr, np.ones(14) / 14, mode='same')
        
        # ===== ADX (Average Directional Index) - 簡化版 =====
        plus_dm = np.zeros_like(closes)
        minus_dm = np.zeros_like(closes)
        
        for i in range(1, len(closes)):
            up_move = highs[i] - highs[i-1]
            down_move = lows[i-1] - lows[i]
            
            if up_move > down_move and up_move > 0:
                plus_dm[i] = up_move
            if down_move > up_move and down_move > 0:
                minus_dm[i] = down_move
        
        plus_di = np.convolve(plus_dm / (atr + 0.001), np.ones(14) / 14, mode='same') * 100
        minus_di = np.convolve(minus_dm / (atr + 0.001), np.ones(14) / 14, mode='same') * 100
        
        dx = abs(plus_di - minus_di) / (plus_di + minus_di + 0.001) * 100
        adx = np.convolve(dx, np.ones(14) / 14, mode='same')
        
        # 應用指標到每個 candle
        for i in range(len(candles)):
            candles[i]["rsi"] = float(rsi[i]) if not np.isnan(rsi[i]) else 50.0
            candles[i]["bb_upper"] = float(bb_upper[i]) if not np.isnan(bb_upper[i]) else 0.0
            candles[i]["bb_middle"] = float(bb_middle[i]) if not np.isnan(bb_middle[i]) else 0.0
            candles[i]["bb_lower"] = float(bb_lower[i]) if not np.isnan(bb_lower[i]) else 0.0
            candles[i]["bb_width"] = float(bb_width[i]) if not np.isnan(bb_width[i]) else 0.0
            candles[i]["atr"] = float(atr[i]) if not np.isnan(atr[i]) else 0.0
            candles[i]["adx"] = float(adx[i]) if not np.isnan(adx[i]) else 20.0
    
    except Exception as e:
        logger.warning(f"[calculate_technical_indicators] Error: {e}")
    
    return candles


def fetch_latest_candles(symbols, timeframe="15m"):
    """
    從交易所 API 取得最新 K 線並計算技術指標
    
    使用 DataFetcher 自動選擇最佳數據源 (Binance 或 yfinance)
    """
    global last_candles_fetch_time
    
    current_time = time.time()
    if current_time - last_candles_fetch_time < fetch_interval:
        # 迴避頻繁請求
        return {}
    
    last_candles_fetch_time = current_time
    
    try:
        # 從數據源取得原始 K 線
        raw_candles = data_fetcher.get_klines(
            symbols=symbols,
            timeframe=timeframe,
            limit=100  # 取 100 根 K 線 (用來計算指標)
        )
        
        # 計算每個幣種的技術指標
        candles_dict = {}
        for symbol, candles in raw_candles.items():
            if len(candles) > 0:
                # 計算技術指標
                candles_with_indicators = calculate_technical_indicators(candles)
                candles_dict[symbol] = candles_with_indicators
            else:
                candles_dict[symbol] = []
        
        logger.debug(f"[fetch_latest_candles] Fetched data for {len(candles_dict)} symbols")
        return candles_dict
    
    except Exception as e:
        logger.error(f"[fetch_latest_candles] Error: {e}")
        return {sym: [] for sym in symbols}


def generate_diagnostics():
    """
    生成完整的診斷報告
    
    返回:
    {
        "total_symbols": 22,
        "loaded_classifiers": 15,
        "loaded_validity_models": 15,
        "classifiers": [
            {
                "symbol": "BTCUSDT",
                "status": "已加載" or "預設值",
                "file": "BTCUSDT_bb_classifier.pkl",
                "features": 8,
                "diagnosis": "OK" or "缺失"
            },
            ...
        ],
        "validity_models": [...],
        "data_source": "Binance US" or "yfinance",
        "diagnostic_report": "..."
    }
    """
    diagnostics = {
        "total_symbols": len(bb_detector_v2.symbols),
        "loaded_classifiers": len(bb_detector_v2.classifiers),
        "loaded_validity_models": len(bb_detector_v2.validity_models),
        "classifiers": [],
        "validity_models": [],
        "data_source": "Binance US" if (data_fetcher.binance_fetcher and data_fetcher.binance_fetcher.initialized) else "yfinance",
    }
    
    # 分類器診斷
    for symbol in bb_detector_v2.symbols:
        classifier_path = f"models/{symbol}_bb_classifier.pkl"
        is_loaded = symbol in bb_detector_v2.classifiers
        
        diagnostics["classifiers"].append({
            "symbol": symbol,
            "status": "已加載" if is_loaded else "預設值",
            "file": classifier_path,
            "features": 8 if is_loaded else "-",
            "diagnosis": "✓ 正常運作" if is_loaded else "⚠ 使用啟發式判斷"
        })
    
    # 有效性模型診斷
    for symbol in bb_detector_v2.symbols:
        validity_path = f"models/{symbol}_validity_model.pkl"
        is_loaded = symbol in bb_detector_v2.validity_models
        
        diagnostics["validity_models"].append({
            "symbol": symbol,
            "status": "已加載" if is_loaded else "預設值",
            "file": validity_path,
            "features": 6 if is_loaded else "-",
            "diagnosis": "✓ 正常運作" if is_loaded else "⚠ 使用預設 65% 信心度"
        })
    
    # 生成診斷報告
    report_lines = [
        "="*70,
        "BB Bounce ML V2 - 模型診斷報告",
        "="*70,
        f"\n系統時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"監控幣種: {diagnostics['total_symbols']} 個",
        f"數據源: {diagnostics['data_source']}",
        f"\n[分類器 Layer 1 - BB 位置檢測]",
        f"已加載: {diagnostics['loaded_classifiers']}/{diagnostics['total_symbols']}",
        f"加載率: {(diagnostics['loaded_classifiers'] / diagnostics['total_symbols'] * 100):.0f}%",
        f"\n[有效性模型 Layer 2 - 信號驗證]",
        f"已加載: {diagnostics['loaded_validity_models']}/{diagnostics['total_symbols']}",
        f"加載率: {(diagnostics['loaded_validity_models'] / diagnostics['total_symbols'] * 100):.0f}%",
    ]
    
    # 檢查模型文件
    missing_classifiers = []
    missing_validity = []
    
    for symbol in bb_detector_v2.symbols:
        if symbol not in bb_detector_v2.classifiers:
            missing_classifiers.append(symbol)
        if symbol not in bb_detector_v2.validity_models:
            missing_validity.append(symbol)
    
    if missing_classifiers:
        report_lines.append(f"\n⚠ 缺失分類器 ({len(missing_classifiers)}): {', '.join(missing_classifiers[:5])}...")
    else:
        report_lines.append(f"\n✓ 所有分類器已加載")
    
    if missing_validity:
        report_lines.append(f"⚠ 缺失有效性模型 ({len(missing_validity)}): {', '.join(missing_validity[:5])}...")
        report_lines.append(f"\n💡 後果: 所有信心度將固定為 65% (預設值)")
    else:
        report_lines.append(f"\n✓ 所有有效性模型已加載")
    
    report_lines.append(f"\n[建議]")
    if missing_classifiers or missing_validity:
        report_lines.append(f"1. 運行訓練腳本: python train_models.py")
        report_lines.append(f"2. 檢查 models/ 目錄是否有所有 .pkl 文件")
        report_lines.append(f"3. 重啟服務: python realtime_service.py")
    else:
        report_lines.append(f"✓ 系統正常運作")
        report_lines.append(f"信心度應該在 30%-95% 範圍內變化")
    
    report_lines.append(f"\n" + "="*70)
    
    diagnostics["diagnostic_report"] = "\n".join(report_lines)
    
    return diagnostics


def realtime_scan_loop():
    """
    實時掃描 loop: 每 5 秒執行一次
    
    流程:
    1. 取得最新 K 線 + 計算技術指標
    2. 加入 detector 緩衝區
    3. 執行二層掃描 (所有 22 個幣種)
    4. 推送信號到前端 (WebSocket)
    5. 更新所有幣種狀態
    """
    logger.info("[realtime_scan_loop] Started (5-second interval)...")
    logger.info(f"[realtime_scan_loop] Data source status: {'Available' if data_fetcher.is_available() else 'Not available'}")
    
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
                    # 保留最新 5 根 K 線
                    for candle in candles[-5:]:
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


@app.route("/debug")
def debug_dashboard():
    return render_template("debug_dashboard.html")


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


@app.route("/api/status")
def api_status():
    return {
        "data_source": "Binance US" if (data_fetcher.binance_fetcher and data_fetcher.binance_fetcher.initialized) else "yfinance",
        "data_source_available": data_fetcher.is_available(),
        "symbols_monitored": len(bb_detector_v2.symbols),
        "signals_count": len(last_signals),
        "scanner_active": scan_active,
    }


@app.route("/api/diagnostics")
def api_diagnostics():
    return generate_diagnostics()


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
        "data_source": "Binance US" if (data_fetcher.binance_fetcher and data_fetcher.binance_fetcher.initialized) else "yfinance",
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
def handle_request_symbol_list(data=None):
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
def handle_force_refresh(data=None):
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


@socketio.on("request_diagnostics")
def handle_request_diagnostics(data=None):
    logger.info("[socket] Request diagnostics")
    diagnostics = generate_diagnostics()
    emit("diagnostics_response", diagnostics)


# ========== 應用啟動 ==========

if __name__ == "__main__":
    # 檢查數據源
    data_source_status = "Available" if data_fetcher.is_available() else "❌ Not available"
    data_source_name = "Binance US" if (data_fetcher.binance_fetcher and data_fetcher.binance_fetcher.initialized) else "yfinance"
    
    print("\n" + "="*70)
    print("BB Bounce ML Realtime Detector V2 - Service")
    print("="*70)
    print(f"Monitoring {len(bb_detector_v2.symbols)} symbols")
    print(f"Scan interval: 5 seconds")
    print(f"Data Source: {data_source_name} - {data_source_status}")
    print(f"Available detection models:")
    print(f"  • Volatility Classifier (波動大小分類)")
    print(f"  • BB Band Classifier (上下軌分類)")
    print(f"  • Validity Validator (有效性驗證)")
    print("="*70 + "\n")
    print(f"📊 實時儀表板: http://127.0.0.1:5000/detector")
    print(f"🔧 調試儀表板: http://127.0.0.1:5000/debug")
    print()
    
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
