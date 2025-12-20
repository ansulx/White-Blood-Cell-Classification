# WBC-Bench-2026 Competition Solution

A comprehensive solution for the WBC-Bench-2026 Kaggle competition focused on White Blood Cell (WBC) image classification.

## 🎯 Competition Overview

This competition involves classifying white blood cell images into 13 different classes:
- BA (Basophil)
- BL (Blast)
- BNE (Band Neutrophil)
- EO (Eosinophil)
- LY (Lymphocyte)
- MMY (Metamyelocyte)
- MO (Monocyte)
- MY (Myelocyte)
- PC (Plasma Cell)
- PLY (Prolymphocyte)
- PMY (Promyelocyte)
- SNE (Segmented Neutrophil)
- VLY (Variant Lymphocyte)

## 📁 Dataset Structure

```
wbc-bench-2026/
├── phase1/              # Phase 1 training images
├── phase1_label.csv     # Phase 1 labels
├── phase2/
│   ├── train/          # Phase 2 training images
│   ├── test/           # Phase 2 test images (no labels)
│   └── eval/           # Phase 2 eval images (no labels)
├── phase2_train.csv    # Phase 2 training labels
├── phase2_test.csv     # Phase 2 test IDs
└── phase2_eval.csv     # Phase 2 eval IDs
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Explore the Dataset

```bash
python explore_data.py
```

This will analyze the dataset distribution, class imbalance, and image statistics.

### 3. Train Models

```bash
python train.py
```

This will:
- Combine Phase 1 and Phase 2 training data
- Perform 5-fold cross-validation
- Train EfficientNet-B4 models with advanced data augmentation
- Handle class imbalance with weighted loss
- Save best models for each fold

### 4. Make Predictions

```bash
python inference.py
```

This will:
- Load all trained models
- Create ensemble predictions
- Apply Test Time Augmentation (TTA)
- Generate predictions for test and eval sets
- Save results to `outputs/predictions/`

## 🏗️ Solution Architecture

### Models
- **Primary**: EfficientNet-B4 (pretrained on ImageNet)
- **Ensemble**: Multiple EfficientNet variants + ConvNeXt
- **Loss**: Focal Loss + Class Weights for handling imbalance

### Data Augmentation
- Horizontal/Vertical flips
- Random rotations
- Shift/Scale/Rotate
- Brightness/Contrast adjustments
- CLAHE (Contrast Limited Adaptive Histogram Equalization)
- Gaussian noise and blur
- Coarse dropout

### Training Strategy
- **5-Fold Stratified Cross-Validation**: Ensures robust validation
- **Class Balancing**: Weighted sampling + class weights in loss
- **Learning Rate**: Cosine annealing schedule
- **Early Stopping**: Prevents overfitting
- **Image Size**: 384x384 for optimal performance

### Inference Strategy
- **Ensemble**: Average predictions from multiple models
- **Test Time Augmentation (TTA)**: 5 different augmentations
- **Soft Voting**: Average probabilities before final prediction

## 📊 Key Features

1. **Handles Class Imbalance**: Uses weighted loss and sampling
2. **Robust Validation**: 5-fold stratified cross-validation
3. **Advanced Augmentation**: Albumentations library for medical imaging
4. **Model Ensembling**: Combines multiple models for better accuracy
5. **TTA**: Test Time Augmentation for improved predictions
6. **Efficient Training**: Uses mixed precision and optimized data loading

## ⚙️ Configuration

Edit `config.py` to customize:
- Model architecture
- Training hyperparameters
- Data augmentation settings
- Image size
- Batch size
- Learning rate

## 📈 Expected Performance

With this solution, you should achieve:
- High accuracy on validation set (>95%)
- Robust generalization to test set
- Competitive leaderboard position

## 🔧 Tips for Improvement

1. **Try Different Models**: Experiment with ConvNeXt, Swin Transformers
2. **Increase Image Size**: Try 512x512 or 640x640 for better accuracy
3. **More Augmentation**: Add more aggressive augmentations
4. **Pseudo-labeling**: Use confident predictions to expand training set
5. **External Data**: If allowed, use additional WBC datasets
6. **Hyperparameter Tuning**: Use Optuna or similar tools

## 📝 Submission Format

The predictions will be saved in CSV format:
```csv
ID,labels
01447013.jpg,SNE
01447014.jpg,LY
...
```

## 🎓 Best Practices Used

- Stratified K-Fold for proper validation
- Class weights for imbalanced data
- Advanced data augmentation
- Model ensembling
- Test Time Augmentation
- Early stopping
- Learning rate scheduling
- Mixed precision training (if GPU available)

## 📚 References

- EfficientNet: https://arxiv.org/abs/1905.11946
- Focal Loss: https://arxiv.org/abs/1708.02002
- Albumentations: https://albumentations.ai/

## 🤝 Contributing

Feel free to experiment and improve upon this solution!

## 📄 License

This solution is provided for educational purposes in the context of the Kaggle competition.

