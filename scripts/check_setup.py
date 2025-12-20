"""
Quick setup verification script
Checks if all dependencies are installed and data is accessible
"""

import sys
import os
from pathlib import Path

def check_dependencies():
    """Check if required packages are installed"""
    print("Checking dependencies...")
    required_packages = [
        'torch',
        'torchvision',
        'timm',
        'albumentations',
        'pandas',
        'numpy',
        'PIL',
        'sklearn',
        'tqdm'
    ]
    
    missing = []
    for package in required_packages:
        try:
            if package == 'PIL':
                __import__('PIL')
            elif package == 'sklearn':
                __import__('sklearn')
            else:
                __import__(package)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package} - MISSING")
            missing.append(package)
    
    if missing:
        print(f"\nMissing packages: {', '.join(missing)}")
        print("Install them with: pip install -r requirements.txt")
        return False
    return True

def check_data():
    """Check if data files exist"""
    print("\nChecking data files...")
    base_dir = Path(__file__).parent
    
    required_files = [
        'phase1_label.csv',
        'phase2_train.csv',
        'phase2_test.csv',
        'phase2_eval.csv',
        'phase1',
        'phase2/train',
        'phase2/test',
        'phase2/eval'
    ]
    
    all_exist = True
    for file_path in required_files:
        full_path = base_dir / file_path
        if full_path.exists():
            if full_path.is_dir():
                count = len(list(full_path.glob('*.jpg')))
                print(f"  ✓ {file_path} ({count} images)")
            else:
                print(f"  ✓ {file_path}")
        else:
            print(f"  ✗ {file_path} - MISSING")
            all_exist = False
    
    return all_exist

def check_gpu():
    """Check GPU availability"""
    print("\nChecking GPU...")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"  ✓ CUDA available")
            print(f"    Device: {torch.cuda.get_device_name(0)}")
            print(f"    Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print("  ⚠ CUDA not available - will use CPU (training will be slower)")
    except:
        print("  ✗ Could not check GPU status")

def main():
    print("=" * 60)
    print("WBC-Bench-2026 Setup Verification")
    print("=" * 60)
    
    deps_ok = check_dependencies()
    data_ok = check_data()
    check_gpu()
    
    print("\n" + "=" * 60)
    if deps_ok and data_ok:
        print("✓ Setup looks good! You can start training.")
        print("\nNext steps:")
        print("  1. Run: python explore_data.py")
        print("  2. Run: python train.py")
        print("  3. Run: python inference.py")
    else:
        print("✗ Setup incomplete. Please fix the issues above.")
    print("=" * 60)

if __name__ == '__main__':
    main()

