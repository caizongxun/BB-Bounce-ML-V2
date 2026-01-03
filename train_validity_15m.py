"""
訓練 15 分鐘時間框架的 validity 模型
"""
from train_validity_model import ValidityModelTrainer
import sys

if __name__ == '__main__':
    trainer = ValidityModelTrainer()
    
    # 訓練所有幣種的 15m validity 模型
    print("\n" + "="*70)
    print("TRAINING 15m VALIDITY MODELS FOR ALL SYMBOLS")
    print("="*70)
    
    results = trainer.train_all_symbols(timeframe='15m')
    
    print("\n" + "="*70)
    print("15m TRAINING COMPLETED!")
    print("="*70)
    
    # Summary
    successful = len([r for r in results.values() if r is not None])
    print(f"\nTotal trained: {successful} symbols")
    print(f"Location: models/validity_models/*/15m/")
