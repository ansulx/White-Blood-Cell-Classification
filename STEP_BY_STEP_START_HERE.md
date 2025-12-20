# 🚀 START HERE: Step-by-Step Action Plan

## Day 1: Setup & Understanding (2-3 hours)

### ✅ Step 1: Verify Setup
```bash
python check_setup.py
```
**Expected**: All dependencies installed, data files found

### ✅ Step 2: Explore Data
```bash
python explore_data.py
```
**What you'll learn:**
- 13 classes total
- Severe imbalance (SNE: 15,048 vs PLY: 13)
- Images are 368x370 pixels

### ✅ Step 3: Visualize Samples
```bash
python visualize_samples.py
```
**What you'll see:**
- What each cell type looks like
- Visual differences between classes
- Image quality

### ✅ Step 4: Read Competition Rules
Visit: https://www.kaggle.com/competitions/wbc-bench-2026/rules
- Note the evaluation metric
- Check submission format
- Understand any restrictions

---

## Day 2: First Baseline (3-4 hours)

### ✅ Step 1: Train Quick Baseline
Edit `config.py`:
```python
MODEL_NAME = 'efficientnet_b0'  # Fast model
IMG_SIZE = 224  # Smaller for speed
BATCH_SIZE = 64
NUM_EPOCHS = 10  # Quick test
```

Run:
```bash
python train.py
```

**Expected Result:**
- Training takes 1-2 hours
- Validation accuracy: ~85-90%
- Models saved in `outputs/models/`

### ✅ Step 2: Make First Predictions
```bash
python inference.py
```

**Check:**
- Predictions saved in `outputs/predictions/`
- Format is correct (ID, labels)
- All test images have predictions

### ✅ Step 3: Log Experiment
```bash
python EXPERIMENT_TRACKER.py
```
Log your baseline results

---

## Day 3-4: Improve Model (6-8 hours)

### ✅ Step 1: Upgrade to Better Model
Edit `config.py`:
```python
MODEL_NAME = 'efficientnet_b4'
IMG_SIZE = 384
BATCH_SIZE = 32
NUM_EPOCHS = 50
```

Run:
```bash
python train.py
```

**Expected:**
- Training: 4-6 hours
- Validation accuracy: ~92-95%
- Better than baseline!

### ✅ Step 2: Check Class-Wise Performance
Look at validation results - are rare classes (PLY, PC, PMY) performing poorly?

### ✅ Step 3: Log This Experiment
Compare with baseline

---

## Day 5-6: Handle Class Imbalance (6-8 hours)

### ✅ Step 1: Verify Class Weights Are Working
Check training logs - are rare classes being sampled more?

### ✅ Step 2: Try Focal Loss
In `config.py` or `models.py`, ensure Focal Loss is being used

### ✅ Step 3: Train with Better Balancing
The code already has weighted sampling - verify it's working

**Expected:**
- Rare class accuracy improves
- Overall accuracy might drop slightly, but that's OK
- Better generalization

---

## Day 7: First Ensemble (4-6 hours)

### ✅ Step 1: Train Second Model
Try ConvNeXt:
```python
MODEL_NAME = 'convnext_base'
```

### ✅ Step 2: Ensemble Predictions
```bash
python inference.py
```
This will automatically ensemble if multiple models exist

### ✅ Step 3: Compare Results
- Single model vs ensemble
- Should see 1-2% improvement

---

## Day 8-10: Advanced Improvements (12-15 hours)

### ✅ Step 1: Try Larger Image Size
```python
IMG_SIZE = 512
```
Train EfficientNet-B4 with 512px

### ✅ Step 2: Hyperparameter Tuning
Try different:
- Learning rates: 5e-5, 1e-4, 2e-4
- Batch sizes: 16, 32, 64
- Dropout: 0.3, 0.4, 0.5

### ✅ Step 3: More Augmentation
Add more aggressive augmentations in `dataset.py`

### ✅ Step 4: Train Multiple Models
- EfficientNet-B4 (384px)
- EfficientNet-B5 (456px)
- ConvNeXt-Base (384px)

---

## Day 11-12: Final Ensemble (8-10 hours)

### ✅ Step 1: Train All Best Models
Train 3-5 different models with best configs

### ✅ Step 2: Create Ensemble
```bash
python inference.py
```

### ✅ Step 3: Test Different Ensemble Strategies
- Simple average
- Weighted average (by validation score)
- Try different combinations

---

## Day 13-14: Submission Prep (4-6 hours)

### ✅ Step 1: Generate Final Predictions
```bash
python inference.py
```

### ✅ Step 2: Verify Submission Format
Check CSV file:
- Has ID and labels columns
- All test images included
- No missing values
- Labels match competition classes

### ✅ Step 3: Create Multiple Submissions
- Single best model
- Ensemble of 3
- Ensemble of 5
- Different TTA strategies

### ✅ Step 4: Submit to Kaggle
1. Go to competition page
2. Click "Submit Predictions"
3. Upload CSV file
4. Wait for score

### ✅ Step 5: Iterate
- If score is lower than expected, check:
  - Are predictions correct format?
  - Are all images predicted?
  - Try different ensemble weights

---

## 🎯 Quick Reference Commands

```bash
# Check setup
python check_setup.py

# Explore data
python explore_data.py

# Visualize samples
python visualize_samples.py

# Train model
python train.py

# Make predictions
python inference.py

# Track experiments
python EXPERIMENT_TRACKER.py

# Check what's in outputs
ls -lh outputs/models/
ls -lh outputs/predictions/
```

---

## 📊 What Success Looks Like

### Week 1 Goals:
- ✅ Baseline working: 85-90% accuracy
- ✅ Better model: 92-95% accuracy
- ✅ Class imbalance handled

### Week 2 Goals:
- ✅ Ensemble: 94-96% accuracy
- ✅ Multiple models trained
- ✅ Final predictions ready

### Final Goals:
- ✅ Top 10% on leaderboard
- ✅ All classes performing well
- ✅ Robust ensemble predictions

---

## 🆘 Troubleshooting

### Problem: Training is too slow
**Solution:**
- Use smaller model (EfficientNet-B0)
- Reduce image size (224px)
- Reduce batch size
- Use fewer epochs for testing

### Problem: Out of memory
**Solution:**
- Reduce batch size (16 or 8)
- Use smaller image size
- Use gradient accumulation

### Problem: Low accuracy on rare classes
**Solution:**
- Increase class weights
- Use more aggressive oversampling
- Focus augmentation on rare classes

### Problem: Overfitting
**Solution:**
- Increase dropout
- Add more augmentation
- Use early stopping
- Reduce model size

---

## 📝 Daily Checklist

**Every day, do:**
- [ ] Check training progress
- [ ] Review validation results
- [ ] Log experiments
- [ ] Plan next experiment
- [ ] Backup models and code

**Before submission:**
- [ ] All models trained
- [ ] Predictions generated
- [ ] Format verified
- [ ] Multiple submissions ready
- [ ] Code pushed to GitHub

---

## 🎓 Remember

1. **Start simple** - Get baseline working first
2. **One change at a time** - So you know what works
3. **Track everything** - Use experiment tracker
4. **Focus on high-impact** - Class imbalance, better models, ensemble
5. **Don't overthink** - The provided code is already good
6. **Iterate fast** - Don't spend days on one thing
7. **Test thoroughly** - Before final submission

**You've got this! 🚀**

