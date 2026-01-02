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
import time

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
            min_bounce_pct=0.3,
            momentum_decay_thresh=0.15
        )
        self.feature_extractor = ValidityFeatures(lookahead=10)
    
    def train_symbol_validity_model(self, 
                                   symbol: str, 
                                   timeframe: str = '15m',
                                   test_size: float = 0.2) -> Dict:
        """
        訓練單一幣種的有效性模型
        """
        print(f'\n{"-"*60}')
        print(f'Training validity model: {symbol} {timeframe}')
        print(f'{"-"*60}')
        
        try:
            # 1. Download data
            print(f'\n[1/8] Downloading {symbol} {timeframe} data...')
            df = self.loader.download_symbol_data(symbol, timeframe)
            if df is None or len(df) < 200:
                print(f'  ERROR: Insufficient data')
                return None
            
            print(f'  OK: Downloaded {len(df)} candles')
            
            # 2. Generate validity labels
            print(f'\n[2/8] Generating validity labels...')
            df = self.label_gen.generate_validity_labels(df, touch_range=0.02)
            
            # Statistics
            stats = self.label_gen.get_validity_statistics(df)
            print(f'  Support validity rate: {stats["support_validity_rate"]*100:.1f}%')
            print(f'  Resistance validity rate: {stats["resistance_validity_rate"]*100:.1f}%')
            print(f'  Overall validity rate: {stats["overall_validity_rate"]*100:.1f}%')
            
            # 3. Extract features
            print(f'\n[3/8] Extracting features...')
            df = self.feature_extractor.extract_all_features(df)
            
            # Create target variable
            df['is_valid'] = ((df['is_valid_support'] == 1) | (df['is_valid_resistance'] == 1)).astype(int)
            
            # 4. Select features and labels
            feature_names = self.feature_extractor.get_feature_names()
            X = df[feature_names]
            y = df['is_valid']
            
            # 5. Select only touch points
            touch_mask = df['touch'] != 0
            X_touch = X[touch_mask]
            y_touch = y[touch_mask]
            
            if len(X_touch) < 30:
                print(f'  ERROR: Insufficient touch data ({len(X_touch)} points)')
                return None
            
            print(f'  OK: Training data: {len(X_touch)} samples')
            print(f'    Valid: {y_touch.sum()} ({y_touch.sum()/len(y_touch)*100:.1f}%)')
            print(f'    Invalid: {(1-y_touch).sum()} ({(1-y_touch).sum()/len(y_touch)*100:.1f}%)')
            
            # 6. Train/test split
            print(f'\n[4/8] Splitting train/test sets...')
            X_train, X_test, y_train, y_test = train_test_split(
                X_touch, y_touch, test_size=test_size, random_state=42, stratify=y_touch
            )
            
            print(f'  Train set: {len(X_train)} samples')
            print(f'  Test set: {len(X_test)} samples')
            
            # 7. Normalize features
            print(f'\n[5/8] Normalizing features...')
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # 8. Train model
            print(f'\n[6/8] Training XGBoost model...')
            
            # Calculate class weight
            n_valid = y_train.sum()
            n_invalid = len(y_train) - n_valid
            
            model = XGBClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                scale_pos_weight=(n_invalid / max(n_valid, 1)),
                verbosity=0
            )
            
            model.fit(X_train_scaled, y_train, verbose=0)
            
            # 9. Evaluate model
            print(f'\n[7/8] Evaluating model...')
            
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
            
            print(f'  Train accuracy: {train_acc*100:.2f}%')
            print(f'  Test accuracy: {test_acc*100:.2f}%')
            print(f'  Train F1: {train_f1:.4f}')
            print(f'  Test F1: {test_f1:.4f}')
            print(f'  Test precision: {test_prec:.4f}')
            print(f'  Test recall: {test_recall:.4f}')
            
            # Check overfitting
            overfit_gap = train_acc - test_acc
            print(f'\n  Overfitting check:')
            if overfit_gap < 0.05:
                print(f'    OK: No overfitting (gap: {overfit_gap:.4f})')
            elif overfit_gap < 0.15:
                print(f'    WARNING: Slight overfitting (gap: {overfit_gap:.4f})')
            else:
                print(f'    ERROR: Moderate overfitting (gap: {overfit_gap:.4f})')
            
            # 10. Feature importance
            print(f'\n  Top 10 important features:')
            feature_importance = model.feature_importances_
            feature_imp_df = pd.DataFrame({
                'feature': feature_names,
                'importance': feature_importance
            }).sort_values('importance', ascending=False)
            
            for idx, row in feature_imp_df.head(10).iterrows():
                print(f'    {row["feature"]:30s}: {row["importance"]:.4f}')
            
            # 11. Save model
            print(f'\n[8/8] Saving model...')
            symbol_model_dir = self.validity_models_dir / symbol / timeframe
            symbol_model_dir.mkdir(parents=True, exist_ok=True)
            
            model_path = symbol_model_dir / 'validity_model.pkl'
            scaler_path = symbol_model_dir / 'scaler.pkl'
            feature_names_path = symbol_model_dir / 'feature_names.pkl'
            
            joblib.dump(model, model_path)
            joblib.dump(scaler, scaler_path)
            joblib.dump(feature_names, feature_names_path)
            
            print(f'  Model saved to: {symbol_model_dir}')
            
            # 12. Return results
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
                'overfit_gap': overfit_gap,
                'feature_importance': feature_imp_df,
                'stats': stats
            }
        
        except Exception as e:
            print(f'\nERROR: Training failed: {e}')
            import traceback
            traceback.print_exc()
            return None
    
    def train_all_symbols(self, timeframe: str = '15m') -> Dict:
        """
        Train validity models for all symbols
        """
        print(f'\n\n{"*"*60}')
        print(f'Training validity models for all symbols')
        print(f'Timeframe: {timeframe}')
        print(f'Total symbols: {len(self.loader.symbols)}')
        print(f'{"*"*60}')
        
        results = {}
        successful_count = 0
        failed_symbols = []
        
        start_time = time.time()
        
        for idx, symbol in enumerate(self.loader.symbols, 1):
            print(f'\n[{idx}/{len(self.loader.symbols)}] ', end='')
            result = self.train_symbol_validity_model(symbol, timeframe)
            
            if result is not None:
                results[symbol] = result
                successful_count += 1
                print(f'  SUCCESS! Accuracy: {result["test_acc"]*100:.2f}%')
            else:
                failed_symbols.append(symbol)
                print(f'  FAILED')
        
        elapsed_time = time.time() - start_time
        
        # Summary
        print(f'\n\n{"="*60}')
        print(f'Training completed!')
        print(f'{"="*60}')
        print(f'\nSuccessful: {successful_count}/{len(self.loader.symbols)} symbols')
        print(f'Training time: {elapsed_time/60:.1f} minutes')
        
        if failed_symbols:
            print(f'\nFailed symbols ({len(failed_symbols)}):' )
            for symbol in failed_symbols:
                print(f'  - {symbol}')
        
        # Performance statistics
        print(f'\n\nPerformance statistics:')
        if results:
            test_accs = [r['test_acc'] for r in results.values()]
            test_f1s = [r['test_f1'] for r in results.values()]
            
            print(f'  Average test accuracy: {np.mean(test_accs)*100:.2f}%')
            print(f'  Best accuracy: {np.max(test_accs)*100:.2f}%')
            print(f'  Worst accuracy: {np.min(test_accs)*100:.2f}%')
            print(f'  Average F1 score: {np.mean(test_f1s):.4f}')
        
        # Performance ranking
        print(f'\n\nPerformance ranking (Top 10):' )
        if results:
            sorted_results = sorted(
                [(s, r['test_acc']) for s, r in results.items()],
                key=lambda x: x[1],
                reverse=True
            )
            
            for i, (symbol, acc) in enumerate(sorted_results[:10], 1):
                print(f'  {i:2d}. {symbol:10s}: {acc*100:.2f}%')
        
        return results


if __name__ == '__main__':
    trainer = ValidityModelTrainer()
    
    # Option 1: Train single symbol
    # result = trainer.train_symbol_validity_model('BTCUSDT', '1h')
    
    # Option 2: Train all symbols (recommended)
    results = trainer.train_all_symbols('1h')
