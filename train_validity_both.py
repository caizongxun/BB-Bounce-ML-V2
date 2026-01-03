"""
訓練所有幣種的 15m 和 1h validity 模型
"""
from train_validity_model import ValidityModelTrainer
import time

if __name__ == '__main__':
    trainer = ValidityModelTrainer()
    
    print("\n" + "="*70)
    print("DUAL-TIMEFRAME VALIDITY MODEL TRAINING")
    print("="*70)
    
    # Phase 1: Train 15m models
    print("\n" + "-"*70)
    print("PHASE 1: Training 15m Validity Models")
    print("-"*70)
    
    start_time_15m = time.time()
    results_15m = trainer.train_all_symbols(timeframe='15m')
    elapsed_15m = time.time() - start_time_15m
    
    successful_15m = len([r for r in results_15m.values() if r is not None])
    print(f"\n15m Training Summary:")
    print(f"  Successful: {successful_15m}/22 symbols")
    print(f"  Time: {elapsed_15m/60:.1f} minutes")
    
    # Phase 2: Train 1h models
    print("\n" + "-"*70)
    print("PHASE 2: Training 1h Validity Models")
    print("-"*70)
    
    start_time_1h = time.time()
    results_1h = trainer.train_all_symbols(timeframe='1h')
    elapsed_1h = time.time() - start_time_1h
    
    successful_1h = len([r for r in results_1h.values() if r is not None])
    print(f"\n1h Training Summary:")
    print(f"  Successful: {successful_1h}/22 symbols")
    print(f"  Time: {elapsed_1h/60:.1f} minutes")
    
    # Final summary
    print("\n" + "="*70)
    print("DUAL-TIMEFRAME TRAINING COMPLETED!")
    print("="*70)
    
    total_time = elapsed_15m + elapsed_1h
    print(f"\nTotal Time: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")
    print(f"\n15m Models: {successful_15m}/22 symbols trained")
    print(f"  Location: models/validity_models/*/15m/")
    print(f"\n1h Models: {successful_1h}/22 symbols trained")
    print(f"  Location: models/validity_models/*/1h/")
    
    # Performance comparison
    if results_15m and results_1h:
        print(f"\nPerformance Comparison:")
        
        acc_15m = [r['test_acc'] for r in results_15m.values() if r]
        acc_1h = [r['test_acc'] for r in results_1h.values() if r]
        
        if acc_15m and acc_1h:
            import numpy as np
            print(f"  15m Average Accuracy: {np.mean(acc_15m)*100:.2f}%")
            print(f"  1h Average Accuracy:  {np.mean(acc_1h)*100:.2f}%")
            print(f"\n  15m Models Directory:")
            print(f"    models/validity_models/[SYMBOL]/15m/")
            print(f"      - validity_model.pkl")
            print(f"      - scaler.pkl")
            print(f"      - feature_names.pkl")
            print(f"\n  1h Models Directory:")
            print(f"    models/validity_models/[SYMBOL]/1h/")
            print(f"      - validity_model.pkl")
            print(f"      - scaler.pkl")
            print(f"      - feature_names.pkl")
    
    print(f"\nNext step: Run test_enhanced_detector.py to verify all models")
