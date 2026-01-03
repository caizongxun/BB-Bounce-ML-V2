#!/usr/bin/env python3
# 最小化測試伺務器 - 用於連接測試

from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import json

app = Flask(__name__)
app.config["SECRET_KEY"] = "test_secret_key"
CORS(app)
socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode='threading'
)

# 模擬的幣種列表
TEST_SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
    "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "TONUSDT", "LINKUSDT",
    "OPUSDT", "ARBUSDT", "SUIUSDT", "APEUSDT", "MATICUSDT",
    "LTCUSDT", "TRXUSDT", "FTMUSDT", "INJUSDT", "SEIUSDT",
    "TIAUSDT", "ORDIUSDT",
]

@app.route("/")
def index():
    return "BB Bounce ML Test Server - Ready"

@app.route("/detector")
def detector():
    return render_template("realtime_dashboard_v2.html")

@socketio.on("connect")
def handle_connect():
    print(f"[客戶端連接] 一個用戶端已連接")
    emit("connection_response", {
        "status": "connected",
        "message": "WebSocket 連接成功"
    })

@socketio.on("disconnect")
def handle_disconnect():
    print(f"[客戶端断開] 一個用戶端已断開")

@socketio.on("request_symbol_list")
def handle_request_symbol_list():
    print("[請求] 當客戶端請求幣種列表")
    response = {
        "symbols": TEST_SYMBOLS,
        "count": len(TEST_SYMBOLS)
    }
    print(f"[回應] 發送 {len(TEST_SYMBOLS)} 個幣種")
    emit("symbol_list_response", response)

@socketio.on("select_symbol")
def handle_select_symbol(data):
    symbol = data.get("symbol")
    selected = data.get("selected", False)
    print(f"[選擇] {symbol} - {'選擇' if selected else '取消'}")
    emit("selection_updated", {
        "symbol": symbol,
        "selected": selected
    })

@socketio.on("set_timeframe")
def handle_set_timeframe(data):
    timeframe = data.get("timeframe")
    print(f"[時間框架] 變更為 {timeframe}")
    emit("timeframe_updated", {
        "timeframe": timeframe
    })

@socketio.on("force_refresh")
def handle_force_refresh(data):
    print("[強制刷新] 客戶端要求強制刷新")
    # 模擬一些信號
    test_signals = [
        {
            "symbol": "BTCUSDT",
            "side": "long",
            "validity_prob": 0.85,
            "bb_position_label": "Upper Band",
            "rsi": 65.5,
            "adx": 28.3,
            "timestamp": 1735880000000
        },
        {
            "symbol": "ETHUSDT",
            "side": "short",
            "validity_prob": 0.72,
            "bb_position_label": "Lower Band",
            "rsi": 35.2,
            "adx": 22.1,
            "timestamp": 1735879900000
        }
    ]
    for signal in test_signals:
        emit("realtime_signal", signal)
    emit("force_refresh_response", {
        "status": "success",
        "signals_found": len(test_signals)
    })

if __name__ == "__main__":
    print("\n" + "="*60)
    print("BB Bounce ML - 測試伺務器")
    print("="*60)
    print(f"伺務器推茶位址: http://localhost:5000")
    print(f處戠器: http://localhost:5000/detector")
    print(f監控幣種: {len(TEST_SYMBOLS)} 個")
    print("="*60 + "\n")
    
    socketio.run(
        app,
        host="127.0.0.1",
        port=5000,
        debug=True,
        use_reloader=False
    )
