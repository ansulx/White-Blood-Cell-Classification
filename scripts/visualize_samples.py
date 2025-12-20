"""
Visualize sample images from each class
Helps understand the dataset visually
"""

import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import numpy as np

def visualize_classes(num_samples=5):
    """Visualize samples from each class"""
    
    # Load data
    phase1 = pd.read_csv('phase1_label.csv')
    phase2_train = pd.read_csv('phase2_train.csv')
    all_train = pd.concat([phase1[['ID', 'labels']], phase2_train[['ID', 'labels']]])
    
    # Get unique classes
    classes = sorted(all_train['labels'].unique())
    num_classes = len(classes)
    
    # Create figure
    fig, axes = plt.subplots(num_classes, num_samples, figsize=(20, 4*num_classes))
    if num_classes == 1:
        axes = axes.reshape(1, -1)
    
    for class_idx, class_name in enumerate(classes):
        # Get samples for this class
        class_samples = all_train[all_train['labels'] == class_name]['ID'].head(num_samples)
        
        for sample_idx, img_name in enumerate(class_samples):
            # Try to find image
            img_path = None
            for base_dir in ['phase1', 'phase2/train']:
                test_path = Path(base_dir) / img_name
                if test_path.exists():
                    img_path = test_path
                    break
            
            if img_path and img_path.exists():
                img = Image.open(img_path)
                axes[class_idx, sample_idx].imshow(img)
                axes[class_idx, sample_idx].axis('off')
                if sample_idx == 0:
                    axes[class_idx, sample_idx].set_ylabel(
                        f'{class_name}\n({len(all_train[all_train["labels"]==class_name])} samples)',
                        fontsize=12, fontweight='bold'
                    )
            else:
                axes[class_idx, sample_idx].text(0.5, 0.5, 'Not found', 
                                                 ha='center', va='center')
                axes[class_idx, sample_idx].axis('off')
    
    plt.tight_layout()
    plt.savefig('class_samples_visualization.png', dpi=150, bbox_inches='tight')
    print("Saved visualization to: class_samples_visualization.png")
    plt.show()

def analyze_image_quality():
    """Analyze image quality metrics"""
    phase1 = pd.read_csv('phase1_label.csv')
    phase2_train = pd.read_csv('phase2_train.csv')
    all_train = pd.concat([phase1[['ID', 'labels']], phase2_train[['ID', 'labels']]])
    
    sizes = []
    for img_name in all_train['ID'].head(1000):  # Sample 1000
        for base_dir in ['phase1', 'phase2/train']:
            img_path = Path(base_dir) / img_name
            if img_path.exists():
                try:
                    img = Image.open(img_path)
                    sizes.append(img.size)
                    break
                except:
                    pass
    
    if sizes:
        sizes = np.array(sizes)
        print("\nImage Size Statistics:")
        print(f"  Width:  {sizes[:, 0].min()} - {sizes[:, 0].max()} (mean: {sizes[:, 0].mean():.1f})")
        print(f"  Height: {sizes[:, 1].min()} - {sizes[:, 1].max()} (mean: {sizes[:, 1].mean():.1f})")
        print(f"  Aspect Ratio: {sizes[:, 0].mean() / sizes[:, 1].mean():.3f}")

if __name__ == '__main__':
    print("Visualizing class samples...")
    visualize_classes(num_samples=5)
    
    print("\nAnalyzing image quality...")
    analyze_image_quality()

