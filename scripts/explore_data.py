"""
Data exploration script for WBC-Bench-2026 competition
Analyzes dataset distribution, class imbalance, and image statistics
"""

import pandas as pd
import numpy as np
from collections import Counter
from PIL import Image
import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_dataset():
    """Analyze the dataset structure and statistics"""
    
    print("=" * 60)
    print("WBC-Bench-2026 Dataset Analysis")
    print("=" * 60)
    
    # Phase 1 analysis
    print("\n--- Phase 1 Dataset ---")
    phase1_df = pd.read_csv('phase1_label.csv')
    print(f"Total images: {len(phase1_df)}")
    print(f"Columns: {phase1_df.columns.tolist()}")
    
    phase1_labels = phase1_df['labels'].value_counts()
    print("\nClass distribution (Phase 1):")
    print(phase1_labels)
    print(f"\nNumber of classes: {len(phase1_labels)}")
    
    # Phase 2 analysis
    print("\n--- Phase 2 Dataset ---")
    phase2_train = pd.read_csv('phase2_train.csv')
    phase2_test = pd.read_csv('phase2_test.csv')
    phase2_eval = pd.read_csv('phase2_eval.csv')
    
    print(f"Train images: {len(phase2_train)}")
    print(f"Test images: {len(phase2_test)}")
    print(f"Eval images: {len(phase2_eval)}")
    
    phase2_labels = phase2_train['labels'].value_counts()
    print("\nClass distribution (Phase 2 Train):")
    print(phase2_labels)
    print(f"\nNumber of classes: {len(phase2_labels)}")
    
    # Check for class overlap
    phase1_classes = set(phase1_labels.index)
    phase2_classes = set(phase2_labels.index)
    print(f"\nClasses in Phase 1: {sorted(phase1_classes)}")
    print(f"Classes in Phase 2: {sorted(phase2_classes)}")
    print(f"Common classes: {sorted(phase1_classes & phase2_classes)}")
    print(f"Phase 1 only: {sorted(phase1_classes - phase2_classes)}")
    print(f"Phase 2 only: {sorted(phase2_classes - phase1_classes)}")
    
    # Image statistics
    print("\n--- Image Statistics ---")
    sample_images = []
    for img_path in ['phase1/00185772.jpg', 'phase2/train/01416766.jpg']:
        if os.path.exists(img_path):
            img = Image.open(img_path)
            sample_images.append({
                'path': img_path,
                'size': img.size,
                'mode': img.mode
            })
            print(f"{img_path}: {img.size}, {img.mode}")
    
    # Check image sizes distribution
    print("\nChecking image size distribution...")
    sizes = []
    for img_file in list(Path('phase1').glob('*.jpg'))[:100]:  # Sample 100 images
        try:
            img = Image.open(img_file)
            sizes.append(img.size)
        except:
            pass
    
    if sizes:
        sizes = np.array(sizes)
        print(f"Width range: {sizes[:, 0].min()} - {sizes[:, 0].max()}, Mean: {sizes[:, 0].mean():.1f}")
        print(f"Height range: {sizes[:, 1].min()} - {sizes[:, 1].max()}, Mean: {sizes[:, 1].mean():.1f}")
    
    # Class imbalance analysis
    print("\n--- Class Imbalance Analysis ---")
    all_labels = pd.concat([phase1_df['labels'], phase2_train['labels']])
    label_counts = all_labels.value_counts()
    imbalance_ratio = label_counts.max() / label_counts.min()
    print(f"Most common class: {label_counts.idxmax()} ({label_counts.max()} samples)")
    print(f"Least common class: {label_counts.idxmin()} ({label_counts.min()} samples)")
    print(f"Imbalance ratio: {imbalance_ratio:.2f}x")
    
    return {
        'phase1': phase1_df,
        'phase2_train': phase2_train,
        'phase2_test': phase2_test,
        'phase2_eval': phase2_eval,
        'all_classes': sorted(set(all_labels))
    }

if __name__ == '__main__':
    data = analyze_dataset()
    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)

