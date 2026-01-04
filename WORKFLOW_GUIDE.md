# Complete Workflow Guide - WBC-Bench-2026

## 🚀 One-Command Workflow (Recommended)

### Main Command:
```bash
python scripts/train.py
```

**That's it!** This single command runs everything automatically:

### What Happens Automatically:

#### **Session 1: Auto-Train All Models** (2-4 hours)
- Trains all 3 ensemble models sequentially:
  - ConvNeXt V2 Large
  - Swin V2 Large
  - MaxViT XLarge
- Each model uses optimized hyperparameters
- 5-fold cross-validation for each model
- Saves best models to `outputs/models/`

#### **Session 2: Pseudo-Labeling & Retraining** (1-2 hours)
- Generates pseudo-labels from test set using ensemble
- Merges pseudo-labels with training data
- Retrains top 2 best models with expanded dataset
- Saves improved models

#### **Session 3: Final Submission** (30-60 minutes)
- Optimizes ensemble weights (tests 4 methods)
- Generates predictions on eval set (for validation)
- Generates final predictions on test set
- Validates submission format
- Saves final submission to `outputs/predictions/final_submission.csv`

### Final Output:
- **Submission file**: `outputs/predictions/final_submission.csv`
- **Metrics**: `outputs/logs/final_submission_metrics.json`
- **Optimization results**: `outputs/logs/ensemble_optimization_results.json`

---

## 📋 Manual Workflow (If Needed)

### Option 1: Train Only (Session 1)
```bash
# In config.py, set:
TRAIN_ALL_ENSEMBLE = True
RUN_PSEUDO_LABELING = False
RUN_FINAL_SUBMISSION = False

python scripts/train.py
```

### Option 2: Train + Pseudo-Labeling (Session 1 & 2)
```bash
# In config.py, set:
TRAIN_ALL_ENSEMBLE = True
RUN_PSEUDO_LABELING = True
RUN_FINAL_SUBMISSION = False

python scripts/train.py
```

### Option 3: Generate Submission Only (After Training)
```bash
# After models are trained, run:
python scripts/final_submission.py
```

### Option 4: Optimize Ensemble Weights Only
```bash
# After models are trained, run:
python scripts/ensemble_optimizer.py
```

### Option 5: Validate Submission
```bash
# Validate existing submission:
python scripts/validate_submission.py
```

---

## 🔧 About `scripts/inference.py`

**Status**: Now a **utility library** (not meant to be run directly)

### What it does:
- Provides inference functions used by Session 3:
  - `predict_test_set()` - Used by `final_submission.py`
  - `predict_eval_set()` - Used by `final_submission.py`
  - `predict_ensemble()` - Used by `ensemble_optimizer.py`
  - `load_model()` - Used by all inference scripts

### When to run it directly:
- **Quick inference** without full training pipeline
- **Testing specific models** individually
- **Debugging** inference issues
- **On-demand predictions** for analysis

### Example standalone usage:
```bash
python scripts/inference.py
# This will:
# 1. Load all trained models
# 2. Generate eval predictions
# 3. Generate test predictions
# 4. Save to outputs/predictions/
```

---

## ✅ Complete Checklist

### Before Running:
- [ ] Data paths configured in `src/config.py`
- [ ] GPU available (recommended) or CPU
- [ ] Sufficient disk space (~50GB for models + data)
- [ ] Python dependencies installed

### After Running `train.py`:
- [ ] Check `outputs/predictions/final_submission.csv` exists
- [ ] Validate submission: `python scripts/validate_submission.py`
- [ ] Check metrics: `outputs/logs/final_submission_metrics.json`
- [ ] Review eval performance (should be > 0.75 macro F1)
- [ ] Submit to Kaggle!

---

## 🎯 Expected Results

### Performance Targets:
- **Eval Macro F1**: 0.750 - 0.780
- **Test Macro F1** (estimated): 0.780 - 0.800
- **BL Class F1**: > 0.50 (critical rare class)

### Time Estimates:
- **Full workflow**: 4-7 hours (depending on GPU)
- **Session 1 only**: 2-4 hours
- **Session 2 only**: 1-2 hours
- **Session 3 only**: 30-60 minutes

---

## 🆘 Troubleshooting

### If training fails:
1. Check GPU memory (reduce `BATCH_SIZE` in config)
2. Check disk space
3. Verify data paths in `config.py`

### If submission generation fails:
1. Ensure models exist in `outputs/models/`
2. Check eval/test CSV files exist
3. Run validation: `python scripts/validate_submission.py`

### If you need to resume:
- Models are saved after each fold
- Can resume from any point
- Pseudo-labels are saved to `outputs/predictions/`

---

## 📊 Summary

**For Competition Submission:**
```bash
python scripts/train.py
# Wait for completion
# Check: outputs/predictions/final_submission.csv
# Submit to Kaggle!
```

**For Development/Testing:**
- Use individual scripts as needed
- `inference.py` for quick testing
- `ensemble_optimizer.py` for weight tuning
- `validate_submission.py` for format checks

---

**Everything is automated. Just run `train.py` and wait!** 🎉

