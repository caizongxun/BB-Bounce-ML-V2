import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from typing import Dict, Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
import ccxt
from datetime import datetime, timedelta
import logging

from data_loader import CryptoDataLoader
from validity_label_generator import ValidityLabelGenerator
from validity_features import ValidityFeatures

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ValidityModelTrainer:
    """
    軌道有效性模型訓練器
    """
    
    def __init__(self, models_dir='models'):
        self.models_dir = Path(models_dir)
        self.validity_models_dir = self.models_dir / 'validity_models'
        self.validity_models_dir.mkdir(parents=True, exist_ok=True)
        
        self.loader = CryptoDataLoader()
        self.label_gen = ValidityLabelGenerator(
            lookahead=10,
            min_bounce_pct=0.5,
            momentum_decay_thresh=0.3
        )
        self.feature_extractor = ValidityFeatures(lookahead=10)
    
    def train_symbol_validity_model(self, 
                                   symbol: str, 
                                   timeframe: str = '15m',
                                   test_size: float = 0.2) -> Dict:
        """
        訓練單一幣種的有效性模型
        """
        print(f'\n{"="*60}')
        print(f'訓練有效性模型: {symbol} {timeframe}')
        print(f'{"="*60}')
        
        try:
            # 1. 下載數據
            print(f'\n✅ 正在下載 {symbol} {timeframe} 數據...')
            df = self.loader.download_symbol_data(symbol, timeframe)
            if df is None or len(df) < 200:
                print(f'❌ 數據不足')
                return None
            
            print(f'   已下載 {len(df)} 根 K 棒')
            
            # 2. 生成有效性標籤
            print(f'\n✅ 生成有效性標籤...')
            df = self.label_gen.generate_validity_labels(df, touch_range=0.02)
            
            # 統計有效性
            stats = self.label_gen.get_validity_statistics(df)
            print(f'   下軌有效率: {stats["support_validity_rate"]*100:.1f}%')
            print(f'   上軌有效率: {stats["resistance_validity_rate"]*100:.1f}%')
            print(f'   整體有效率: {stats["overall_validity_rate"]*100:.1f}%')
            
            # 3. 基于有效標籤粗每個觸碰點的訓練數據
            print(f'\n✅ 提取特徵...')
            df = self.feature_extractor.extract_all_features(df)
            
            # 口变穗變量
            # validity_label: 1 = 有效, 0 = 無效
            df['validity_label_binary'] = (df['touch'] != 0).astype(int)  # 是否觸碰
            df['is_valid'] = ((df['is_valid_support'] == 1) | (df['is_valid_resistance'] == 1)).astype(int)  # 是否有效
            
            # 4. 粗選特徵和標籤
            feature_names = self.feature_extractor.get_feature_names()
            X = df[feature_names]
            y = df['is_valid']  # 二分粗：有效 vs 無效
            
            # 5. 只粗選觸碰點的數據
            # 因為我們句觸碰點決定是有效或無效
            touch_mask = df['touch'] != 0
            X_touch = X[touch_mask]
            y_touch = y[touch_mask]
            
            if len(X_touch) < 50:
                print(f'❌ 觸碰数据不足 ({len(X_touch)} 個)')
                return None
            
            print(f'   有效性訓練數据: {len(X_touch)} 筆')
            print(f'   有效控節: {y_touch.sum()} 筆')
            print(f'   無效控節: {(1-y_touch).sum()} 筆')
            
            # 6. 進行訓練／測試分割
            print(f'\n✅ 分割訓練/測試集...')
            X_train, X_test, y_train, y_test = train_test_split(
                X_touch, y_touch, test_size=test_size, random_state=42, stratify=y_touch
            )
            
            print(f'   訓練集: {len(X_train)} 筆')
            print(f'   測試集: {len(X_test)} 筆')
            
            # 7. 正證化特徵
            print(f'\n✅ 正證化特徵...')
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # 8. 訓練模型
            print(f'\n✅ 訓練 XGBoost 有效性模型...')
            
            # 計算類別權重 (處理不正茨)
            n_valid = y_train.sum()
            n_invalid = len(y_train) - n_valid
            class_weight = {0: n_valid / len(y_train), 1: n_invalid / len(y_train)}
            
            model = XGBClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                scale_pos_weight=(n_invalid / n_valid),  # XGBoost 特有模式
                verbosity=0
            )
            
            model.fit(X_train_scaled, y_train, verbose=0)
            
            # 9. 計算性能指標
            print(f'\n📊 模型性能:')
            
            y_train_pred = model.predict(X_train_scaled)
            y_test_pred = model.predict(X_test_scaled)
            
            train_acc = accuracy_score(y_train, y_train_pred)
            test_acc = accuracy_score(y_test, y_test_pred)
            
            train_f1 = f1_score(y_train, y_train_pred)
            test_f1 = f1_score(y_test, y_test_pred)
            
            train_prec = precision_score(y_train, y_train_pred)
            test_prec = precision_score(y_test, y_test_pred)
            
            train_recall = recall_score(y_train, y_train_pred)
            test_recall = recall_score(y_test, y_test_pred)
            
            print(f'  訓練集精準度: {train_acc:.4f} ({train_acc*100:.2f}%)')
            print(f'  測試集精準度: {test_acc:.4f} ({test_acc*100:.2f}%)')
            print(f'  訓練集 F1: {train_f1:.4f}')
            print(f'  測試集 F1: {test_f1:.4f}')
            print(f'  訓練集精稆度: {train_prec:.4f}')
            print(f'  測試集精稆度: {test_prec:.4f}')
            print(f'  訓練集召回率: {train_recall:.4f}')
            print(f'  測試集召回率: {test_recall:.4f}')
            
            # 檢查過似合
            overfit_acc = train_acc - test_acc
            print(f'\n⚠️  過似合棄查:')
            if overfit_acc < 0.05:
                print(f'  ✅ 沒有過似合溋象 (不準問寶: {overfit_acc:.4f})')
            elif overfit_acc < 0.1:
                print(f'  ⚠️  輕微過似合 (不準問寶: {overfit_acc:.4f})')
            else:
                print(f'  ❌ 中度過似合 (不準問寶: {overfit_acc:.4f})')
            
            # 檢杧矩陣
            print(f'\n檢杧矩陣 (測試集):')
            cm = confusion_matrix(y_test, y_test_pred)
            print(f'  TN: {cm[0, 0]}, FP: {cm[0, 1]}')
            print(f'  FN: {cm[1, 0]}, TP: {cm[1, 1]}')
            
            # 10. 串推特徒重要性
            print(f'\n📄 特徵重要性排序 (前 10 個):')
            feature_importance = model.feature_importances_
            feature_imp_df = pd.DataFrame({
                'feature': feature_names,
                'importance': feature_importance
            }).sort_values('importance', ascending=False)
            
            for idx, row in feature_imp_df.head(10).iterrows():
                print(f'  {row["feature"]:30s}: {row["importance"]:.4f}')
            
            # 11. 上存模型
            print(f'\n✅ 棄上存模型...')
            symbol_model_dir = self.validity_models_dir / symbol / timeframe
            symbol_model_dir.mkdir(parents=True, exist_ok=True)
            
            model_path = symbol_model_dir / 'validity_model.pkl'
            scaler_path = symbol_model_dir / 'scaler.pkl'
            feature_names_path = symbol_model_dir / 'feature_names.pkl'
            
            joblib.dump(model, model_path)
            joblib.dump(scaler, scaler_path)
            joblib.dump(feature_names, feature_names_path)
            
            print(f'   模型已上存到: {model_path}')
            
            # 12. 回傳結果
            return {
                'symbol': symbol,
                'timeframe': timeframe,
                'model': model,
                'scaler': scaler,
                'feature_names': feature_names,
                'train_acc': train_acc,
                'test_acc': test_acc,
                'train_f1': train_f1,
                'test_f1': test_f1,
                'test_precision': test_prec,
                'test_recall': test_recall,
                'overfit_gap': overfit_acc,
                'feature_importance': feature_imp_df,
                'stats': stats
            }
        
        except Exception as e:
            print(f'\n❌ 訓練失敗: {e}')
            import traceback
            traceback.print_exc()
            return None
    
    def train_all_symbols(self, timeframe: str = '15m') -> Dict:
        """
        訓練所有幣種的有效性模型
        """
        print(f'\n✨ 開始訓練有效性模型...')
        
        results = {}
        successful_count = 0
        
        for symbol in self.loader.symbols:
            result = self.train_symbol_validity_model(symbol, timeframe)
            
            if result is not None:
                results[symbol] = result
                successful_count += 1
        
        # 綜合統計
        print(f'\n\n{"="*60}')
        print(f'訓練完成！')
        print(f'{"="*60}')
        print(f'成功訓練: {successful_count}/{len(self.loader.symbols)} 個幣種')
        
        # 顯示詳適性能
        print(f'\n📊 綜合性能統計:')
        if results:
            avg_test_acc = np.mean([r['test_acc'] for r in results.values()])
            avg_test_f1 = np.mean([r['test_f1'] for r in results.values()])
            avg_test_prec = np.mean([r['test_precision'] for r in results.values()])
            avg_test_recall = np.mean([r['test_recall'] for r in results.values()])
            
            print(f'  平均測試集精準度: {avg_test_acc:.4f} ({avg_test_acc*100:.2f}%)')
            print(f'  平均 F1 分數: {avg_test_f1:.4f}')
            print(f'  平均精稆度: {avg_test_prec:.4f}')
            print(f'  平均召回率: {avg_test_recall:.4f}')
        
        return results


if __name__ == '__main__':
    trainer = ValidityModelTrainer()
    
    # 訓練單一幣種
    print('\n🚀 正在訓練 BTCUSDT 1h 有效性模型...')
    result = trainer.train_symbol_validity_model('BTCUSDT', '1h')
    
    # 或訓練所有幣種
    # results = trainer.train_all_symbols('15m')
