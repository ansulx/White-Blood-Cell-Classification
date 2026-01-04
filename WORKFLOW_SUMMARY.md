# Complete Integrated Workflow - Session 1 & 2

## 🚀 ONE COMMAND TO RULE THEM ALL

```bash
python scripts/train.py
```

That's it! Everything happens automatically.

---

## 📋 What Happens Automatically

### SESSION 1: Foundation Models (5-6 hours)

1. **Trains 3 Best Models Sequentially:**
   - ConvNeXt V2 Large (LR=5e-5, Batch=64)
   - Swin V2 Large (LR=4e-5, Batch=48)
   - MaxViT XLarge (LR=3e-5, Batch=32)

2. **Each Model:**
   - 5-fold cross-validation
   - Domain adaptation augmentation enabled
   - Class-aware augmentation for rare classes
   - BL class monitoring
   - Phase 2 Eval set included in training

3. **Output:**
   - 15 model files (3 models × 5 folds)
   - Training metrics and plots
   - CV performance summary

### SESSION 2: Pseudo-Labeling & Optimization (Automatic)

1. **Pseudo-Label Generation:**
   - Uses all trained models for ensemble predictions
   - Confidence threshold: 0.95 (very high)
   - Generates high-confidence labels from eval set
   - Merges with original training data

2. **Retraining:**
   - Automatically identifies best 2 models
   - Retrains them with pseudo-labels
   - Uses merged training data (original + pseudo-labels)

3. **Output:**
   - Additional 10 model files (2 models × 5 folds, retrained)
   - Total: 25 model files
   - Improved performance with pseudo-labels

---

## ⚙️ Configuration

All settings in `src/config.py`:

```python
# Session 1
TRAIN_ALL_ENSEMBLE = True  # Auto-train all models

# Session 2
RUN_PSEUDO_LABELING = True  # Generate pseudo-labels after Session 1
USE_PSEUDO_LABELS = True    # Use pseudo-labels if available
RETRAIN_WITH_PSEUDO = True  # Retrain best models with pseudo-labels
PSEUDO_LABEL_THRESHOLD = 0.95  # Confidence threshold
```

---

## 📊 Expected Results

| Stage | Models | Expected Macro F1 | Time |
|-------|--------|-------------------|------|
| Session 1 | 15 models | 0.68-0.72 | 5-6 hours |
| Session 2 | +10 models | 0.72-0.75 | +2-3 hours |
| **Total** | **25 models** | **0.72-0.75** | **7-9 hours** |

---

## 🔧 Optional: Hyperparameter Optimization

After Session 1 & 2, optionally run:

```bash
# Install Optuna
pip install optuna

# Run optimization (~1 hour)
python scripts/hyperparameter_opt.py

# Update config.py with best hyperparameters
# Then retrain if needed
```

---

## 📁 Output Files

### Models
- `outputs/models/*_best.pth` - All trained models

### Predictions
- `outputs/predictions/pseudo_labels_thresh0.95.csv` - Pseudo-labels
- `outputs/predictions/merged_train_with_pseudo.csv` - Merged training data
- `outputs/predictions/eval_predictions.csv` - Eval set predictions
- `outputs/predictions/test_predictions.csv` - Test set predictions

### Metrics & Logs
- `outputs/logs/metrics/*.csv` - Training metrics
- `outputs/logs/metrics/*.json` - Summary files
- `outputs/logs/plots/*.png` - Training plots

---

## 🎯 Final Inference

After training completes:

```bash
# Run inference with all models (including retrained)
python scripts/inference.py

# This uses:
# - All 25 models (Session 1 + Session 2)
# - Classy Ensemble (per-class weighted)
# - Enhanced TTA (15 transforms)
```

---

## ✅ Complete Workflow Summary

1. **Run:** `python scripts/train.py`
2. **Wait:** 7-9 hours (automatic)
3. **Run:** `python scripts/inference.py`
4. **Submit:** `outputs/predictions/test_predictions.csv`

**That's it! No manual steps needed.**

---

## 🛠️ Troubleshooting

### If OOM (Out of Memory):
- Reduce `BATCH_SIZE` in `MODEL_SPECIFIC_SETTINGS`
- Or reduce `IMG_SIZE` to 448

### If Pseudo-Labeling Fails:
- Check that Session 1 models exist
- Verify eval set CSV has labels
- Check disk space

### If You Want to Skip Session 2:
```python
# In config.py:
RUN_PSEUDO_LABELING = False
```

---

## 📈 Performance Tracking

- BL class F1 tracked in all logs
- Per-class metrics saved
- CV vs Eval comparison available
- Use `scripts/verify_eval_performance.py` for detailed analysis

---

**Everything is automated. Just run `python scripts/train.py` and wait!** 🚀

