"""
Script to train multiple models for ensembling
Trains different architectures to create a diverse ensemble
"""

import subprocess
import sys
from config import Config

def train_model(model_name):
    """Train a single model by modifying config and running train.py"""
    print(f"\n{'='*60}")
    print(f"Training {model_name}")
    print(f"{'='*60}\n")
    
    # Modify config temporarily
    config = Config()
    original_model = config.MODEL_NAME
    
    # Update model name in config
    import config as config_module
    config_module.Config.MODEL_NAME = model_name
    
    # Import and run training
    from train import main
    main()
    
    # Restore original
    config_module.Config.MODEL_NAME = original_model

def main():
    """Train multiple models for ensemble"""
    models_to_train = [
        'efficientnet_b4',
        'efficientnet_b5',
        'convnext_base',
        # Add more models as needed
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

