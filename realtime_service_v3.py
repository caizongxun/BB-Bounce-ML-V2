# 修載提示：層級1: BB觸厬阈測作遲位置提上 (430-460 行)

# 修載的 predict_bb_touch() 函数詳段

def predict_bb_touch(self, symbol, timeframe, ohlcv_data):
    """
    修載版本：解決「觸厬是但位置未知」的矛牶
    """
    key = (symbol, timeframe)
    if key not in self.bb_models:
        return None
    
    models = self.bb_models[key]
    if not models['model'] or not models['scaler']:
        return None
    
    try:
        features = self.bb_feature_extractor.extract_features(symbol, timeframe, ohlcv_data)
        features_scaled = models['scaler'].transform([features])
        prediction = models['model'].predict(features_scaled)[0]
        probabilities = models['model'].predict_proba(features_scaled)[0]  # 取得全部概率
        confidence = float(np.max(probabilities))  # 最高概率
        
        # 修載的位置似車邨道（重要）
        label_map = models['label_map'] or {0: 'lower', 1: 'none', 2: 'upper'}
        
        # 採用最高概率的一個作為預測位置
        best_class = np.argmax(probabilities)  # 最高概率的類別
        touch_type = label_map.get(best_class, 'unknown')
        
        # 轉換適輯：
        # - 如果 best_class != 1 (不是 'none')
        # - 且信心度 > 0.3
        # 則認为有觸厬
        touched = (best_class != 1) and (confidence > 0.3)
        
        # 如果沒有觸厬，位置設為 'none'
        if not touched:
            touch_type = 'none'
        
        return {
            'touched': touched,
            'touch_type': touch_type,
            'confidence': float(confidence),
            'prediction': int(best_class),
            'probabilities': {label_map.get(i, f'class_{i}'): float(p) for i, p in enumerate(probabilities)}
        }
    except Exception as e:
        logger.error(f'BB觸厬預測失變 {symbol} {timeframe}: {e}')
        return None
