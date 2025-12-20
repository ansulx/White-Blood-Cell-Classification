# ⚡ Quick Reference Card

## 🎯 Competition Summary
- **Task**: Classify 13 White Blood Cell types
- **Classes**: BA, BL, BNE, EO, LY, MMY, MO, MY, PC, PLY, PMY, SNE, VLY
- **Challenge**: Severe class imbalance (1157x ratio)
- **Data**: ~33K train, ~16K test, ~5K eval images

## 📋 Key Insights from Data
- **Most common**: SNE (15,048 samples)
- **Rare classes**: PLY (13), PC (58), PMY (118)
- **Image size**: 368x370 pixels
- **Total classes**: 13

## 🚀 Quick Start Commands

```bash
# 1. Check everything works
python check_setup.py

# 2. Explore data
python explore_data.py

# 3. Visualize samples
python visualize_samples.py

# 4. Train baseline (quick test)
# Edit config.py: MODEL_NAME='efficientnet_b0', IMG_SIZE=224
python train.py

# 5. Train better model
# Edit config.py: MODEL_NAME='efficientnet_b4', IMG_SIZE=384
python train.py

# 6. Make predictions
python inference.py

# 7. Track experiments
python EXPERIMENT_TRACKER.py
```

## 🎯 Priority Actions (Do in Order)

### ✅ Week 1: Foundation
1. **Day 1**: Setup + data exploration
2. **Day 2**: Baseline model (EfficientNet-B0)
3. **Day 3-4**: Better model (EfficientNet-B4)
4. **Day 5-6**: Handle class imbalance
5. **Day 7**: First ensemble

### ✅ Week 2: Optimization
6. **Day 8-9**: Advanced techniques
7. **Day 10-11**: Hyperparameter tuning
8. **Day 12-13**: Final ensemble
9. **Day 14**: Submission

## 🔧 Config Changes for Experiments

### Quick Test (Fast)
```python
MODEL_NAME = 'efficientnet_b0'
IMG_SIZE = 224
BATCH_SIZE = 64
NUM_EPOCHS = 10
```

### Good Performance
```python
MODEL_NAME = 'efficientnet_b4'
IMG_SIZE = 384
BATCH_SIZE = 32
NUM_EPOCHS = 50
```

### Best Performance (Slow)
```python
MODEL_NAME = 'efficientnet_b5'
IMG_SIZE = 512
BATCH_SIZE = 16
NUM_EPOCHS = 100
```

## 📊 Expected Results

| Model | Image Size | Val Acc | Training Time |
|-------|------------|---------|---------------|
| EfficientNet-B0 | 224px | 85-90% | 1-2 hours |
| EfficientNet-B4 | 384px | 92-95% | 4-6 hours |
| EfficientNet-B5 | 512px | 93-96% | 8-10 hours |
| Ensemble (3 models) | - | 94-97% | - |

## 🎯 Key Experiments

1. **Model Architecture** (High Impact)
   - EfficientNet-B4 → EfficientNet-B5
   - Try ConvNeXt, Swin Transformer

2. **Image Size** (Medium Impact)
   - 224 → 384 → 512 → 640

3. **Class Imbalance** (High Impact)
   - Weighted loss
   - Focal loss
   - Oversampling

4. **Ensembling** (High Impact)
   - 3-5 diverse models
   - Weighted averaging

5. **TTA** (Medium Impact)
   - Already implemented
   - Try more augmentations

## 🚫 What NOT to Do

- ❌ Don't create custom architectures (initially)
- ❌ Don't spend days on one experiment
- ❌ Don't ignore class imbalance
- ❌ Don't submit without ensemble
- ❌ Don't forget to validate format

## ✅ What TO Do

- ✅ Start with baseline
- ✅ Improve iteratively
- ✅ Track all experiments
- ✅ Focus on rare classes
- ✅ Ensemble diverse models
- ✅ Test submission format

## 📁 File Structure

```
wbc-bench-2026/
├── config.py              # Configuration (edit this!)
├── train.py               # Training script
├── inference.py           # Prediction script
├── dataset.py             # Dataset handling
├── models.py              # Model definitions
├── explore_data.py        # Data analysis
├── visualize_samples.py   # Visualize classes
├── EXPERIMENT_TRACKER.py  # Track experiments
├── COMPLETE_GUIDE.md      # Full guide
├── STEP_BY_STEP_START_HERE.md  # Start here!
└── outputs/
    ├── models/            # Trained models
    └── predictions/       # Submission files
```

## 🎓 Pro Tips

1. **Rare classes are key** - PLY, PC, PMY differentiate winners
2. **Diverse ensemble** - Mix CNN + Transformer
3. **Strong TTA** - Use 10+ augmentations
4. **Iterate fast** - Don't overthink
5. **Track everything** - Know what works

## 🆘 Common Issues

**Out of memory?**
→ Reduce batch_size to 16 or 8

**Training too slow?**
→ Use EfficientNet-B0, 224px for testing

**Low accuracy?**
→ Check class imbalance handling
→ Try larger model/image size

**Submission rejected?**
→ Check CSV format (ID, labels)
→ Verify all images predicted
→ Check label names match exactly

## 📞 Next Steps

1. Read: `STEP_BY_STEP_START_HERE.md`
2. Read: `COMPLETE_GUIDE.md` (for details)
3. Run: `python check_setup.py`
4. Start: Follow Day 1 checklist

**Good luck! 🚀**

