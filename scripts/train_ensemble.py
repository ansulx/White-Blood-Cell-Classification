"""
Script to train multiple models for ensembling
Trains different architectures to create a diverse ensemble
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import Config

def train_model(model_name):
    """Train a single model by modifying config and running train.py"""
    print(f"\n{'='*60}")
    print(f"Training {model_name}")
    print(f"{'='*60}\n")
    
    # Modify config temporarily
    config = Config()
    original_model = config.MODEL_NAME
    
    # Update model name in config
    import src.config as config_module
    original_model = config_module.Config.MODEL_NAME
    config_module.Config.MODEL_NAME = model_name
    
    # Import and run training
    from scripts.train import main
    main()
    
    # Restore original
    config_module.Config.MODEL_NAME = original_model

def main():
    """Train multiple models for ensemble - BEST AVAILABLE models strategy"""
    # Strategy: 3 BEST diverse architectures (maximum performance, no redundancy)
    # 1. V2 Large: Strongest ConvNeXt (best CNN accuracy)
    # 2. Swin V2 Large: Best transformer (largest window, best accuracy)
    # 3. MaxViT XLarge: Best multi-axis (maximum performance)
    models_to_train = [
        'convnextv2_large',                    # BEST ConvNeXt V2 - strongest CNN
        'swinv2_large_window12to16_192to256',   # FIXED: Optimized window size for 512 images
        'maxvit_xlarge_tf_384',                # BEST MaxViT - XLarge variant
    ]
    
    print("Training ensemble models...")
    print(f"Models to train: {models_to_train}")
    
    for model_name in models_to_train:
        try:
            train_model(model_name)
        except Exception as e:
            print(f"Error training {model_name}: {e}")
            continue
    
    print("\n" + "="*60)
    print("Ensemble training complete!")
    print("="*60)

if __name__ == '__main__':
    main()

