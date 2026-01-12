"""
Quick script to check if model files are corrupted
Run this to identify which file is causing issues
"""

import torch
from pathlib import Path
from src.config import Config

config = Config()

print("="*70)
print("Checking ConvNeXt V2 Large Model Files")
print("="*70)

for fold in range(5):
    model_path = config.MODEL_DIR / f'convnextv2_large_fold{fold}_best.pth'
    
    if not model_path.exists():
        print(f"\n❌ Fold {fold}: FILE NOT FOUND")
        print(f"   {model_path}")
        continue
    
    file_size_mb = model_path.stat().st_size / (1024 * 1024)
    print(f"\nFold {fold}: {model_path.name}")
    print(f"  Size: {file_size_mb:.1f}MB ({file_size_mb/1024:.2f}GB)")
    
    try:
        # Try to load the checkpoint
        checkpoint = torch.load(model_path, map_location='cpu')
        print(f"  ✅ File loads successfully")
        print(f"  Keys in checkpoint: {list(checkpoint.keys())[:5]}...")
    except RuntimeError as e:
        if "central directory" in str(e) or "zip archive" in str(e):
            print(f"  ❌ CORRUPTED - File is incomplete or corrupted")
            print(f"  Error: {str(e)[:100]}")
        else:
            print(f"  ❌ ERROR: {str(e)[:100]}")
    except Exception as e:
        print(f"  ❌ ERROR: {type(e).__name__}: {str(e)[:100]}")

print("\n" + "="*70)
print("If any file shows ❌, you need to re-copy that file")
print("="*70)
