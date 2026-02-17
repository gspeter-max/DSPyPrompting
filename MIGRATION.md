# DSPyPrompting Migration Guide

This guide helps existing users migrate to the new optimizer selection feature.

## Overview

**What's New:**
- Added MIPROv2 optimizer as an alternative to BootstrapFewShot
- CLI arguments for optimizer selection
- Separate model files for each optimizer
- No breaking changes to existing functionality

**Breaking Changes:** None

---

## Quick Migration Checklist

- [x] No code changes required
- [x] Existing scripts continue to work
- [x] Default behavior unchanged
- [x] Optional new features available

---

## What Changed

### Before (Previous Version)

```bash
# Only one way to train
python train.py

# Only one optimizer (BootstrapFewShot)
# Only one model file (trained_qa_model.json)
```

### After (This Version)

```bash
# Default behavior unchanged (BootstrapFewShot)
python train.py

# New option: Use MIPROv2
python train.py --optimizer miprov2

# Separate model files
# trained_qa_model_bootstrap.json (default)
# trained_qa_model_miprov2.json (when using MIPROv2)
```

---

## Migration Scenarios

### Scenario 1: No Changes Required

**You:** Just want to continue using the system as before

**Action:** None! Default behavior is unchanged.

```bash
# This still works exactly the same
python train.py
```

**What happens:**
- Uses BootstrapFewShot (same as before)
- Model saved to `trained_qa_model_bootstrap.json`
- All existing tests pass
- No action required

---

### Scenario 2: Try MIPROv2 (Optional)

**You:** Want to try the new MIPROv2 optimizer

**Action:** Use the `--optimizer` flag

```bash
# Train with MIPROv2
python train.py --optimizer miprov2

# Train with MIPROv2 (medium mode)
python train.py --optimizer miprov2 --auto medium
```

**What happens:**
- Uses MIPROv2 optimizer
- Model saved to `trained_qa_model_miprov2.json`
- Better quality (usually)
- Slower training time

**Reverting if needed:**

```bash
# Just use default again
python train.py
```

---

### Scenario 3: Update Scripts

**You:** Have custom scripts that call `train.py`

**Before:**
```bash
#!/bin/bash
# train.sh
python train.py
python test.py
```

**After (Optional):**
```bash
#!/bin/bash
# train.sh

# Option A: Use default (BootstrapFewShot) - no change
python train.py

# Option B: Use MIPROv2 for better quality
python train.py --optimizer miprov2 --auto medium

python test.py
```

**Impact:** None - both options work

---

### Scenario 4: Update Model Loading

**You:** Load `trained_qa_model.json` in your code

**Before:**
```python
# Load model
qa = dspy.QAModule()
qa.load("trained_qa_model.json")
```

**After:**
```python
# Option A: Load BootstrapFewShot model (default)
qa = dspy.QAModule()
qa.load("trained_qa_model_bootstrap.json")

# Option B: Load MIPROv2 model
qa = dspy.QAModule()
qa.load("trained_qa_model_miprov2.json")
```

**Migration:**
- If you trained with the default, rename your model file:
  ```bash
  mv trained_qa_model.json trained_qa_model_bootstrap.json
  ```
- If you want to use MIPROv2, train new model and update path

---

### Scenario 5: CI/CD Pipeline

**You:** Have CI/CD that trains the model

**Before:**
```yaml
# .github/workflows/train.yml
- name: Train model
  run: python train.py
```

**After (Optional):**
```yaml
# .github/workflows/train.yml

# Option A: Keep using BootstrapFewShot (no change)
- name: Train model
  run: python train.py

# Option B: Use MIPROv2 for production
- name: Train model
  run: python train.py --optimizer miprov2 --auto medium
```

---

## Feature Comparison

| Feature | Before | After |
|---------|--------|-------|
| Default optimizer | BootstrapFewShot | BootstrapFewShot (unchanged) |
| Optimizer choice | 1 | 2 (BootstrapFewShot, MIPROv2) |
| CLI arguments | None | --optimizer, --auto, --num-threads |
| Model filename | `trained_qa_model.json` | `trained_qa_model_{optimizer}.json` |
| Breaking changes | N/A | **None** |
| Required migration | N/A | **None** |

---

## Model File Migration

### Old Model File

If you have an existing `trained_qa_model.json`:

**Option 1: Keep using it (no changes)**
```bash
# Rename to match new convention
mv trained_qa_model.json trained_qa_model_bootstrap.json

# Update your code to use new filename
qa.load("trained_qa_model_bootstrap.json")
```

**Option 2: Train new model**
```bash
# Train with default (BootstrapFewShot)
python train.py
# Creates: trained_qa_model_bootstrap.json

# Or train with MIPROv2
python train.py --optimizer miprov2
# Creates: trained_qa_model_miprov2.json
```

---

## API Changes

### train.py

**Before:**
```python
# train.py - no CLI arguments
# Always used BootstrapFewShot
```

**After:**
```python
# train.py - optional CLI arguments
# Default: BootstrapFewShot (no change)
python train.py

# New: MIPROv2
python train.py --optimizer miprov2
```

### qa_module.py

**No changes** - QAModule and metrics work with both optimizers

### dataset.py

**No changes** - Same dataset for both optimizers

---

## Testing Migration

### Verify Existing Tests

All existing tests should pass without modification:

```bash
# Run all tests
python -m pytest tests/ -v

# Expected: 250 tests pass
```

### Test New Optimizer

```bash
# Train with MIPROv2
python train.py --optimizer miprov2 --auto light

# Run tests
python test.py

# Verify model file exists
ls -lh trained_qa_model_miprov2.json
```

---

## Rollback Plan

If you encounter any issues:

### Step 1: Revert to Default

```bash
# Use BootstrapFewShot (default behavior)
python train.py
```

### Step 2: Rename Model File

```bash
# If you need the old filename
mv trained_qa_model_bootstrap.json trained_qa_model.json
```

### Step 3: Update Code

```python
# Use old model path in your code
qa.load("trained_qa_model.json")
```

---

## Common Questions

### Q: Do I have to change my code?

**A:** No. Default behavior is unchanged. If you want to use MIPROv2, that's optional.

### Q: Will my existing model file work?

**A:** Yes. Just rename it to `trained_qa_model_bootstrap.json` to match the new convention.

### Q: Which optimizer should I use?

**A:**
- **BootstrapFewShot:** Quick iteration, development
- **MIPROv2:** Production, maximum accuracy

See [USAGE.md](USAGE.md) for detailed guidance.

### Q: Can I use both optimizers?

**A:** Yes! Train both models and compare:
```bash
python train.py                           # BootstrapFewShot
python train.py --optimizer miprov2       # MIPROv2
python compare_optimizers.py              # Compare results
```

### Q: What happens to my existing scripts?

**A:** They continue to work. Default behavior is unchanged.

---

## Summary

**Migration Difficulty:** Easy (No changes required)

**Breaking Changes:** None

**Recommended Actions:**
1. **Optional:** Try MIPROv2 with `python train.py --optimizer miprov2`
2. **Optional:** Rename existing model to `trained_qa_model_bootstrap.json`
3. **Optional:** Update model loading paths if needed
4. **Recommended:** Run tests to verify everything works

**Support:**
- See [USAGE.md](USAGE.md) for detailed usage instructions
- See [README.md](README.md) for optimizer comparison
- Run `python train.py --help` for CLI options

---

## Migration Checklist

Use this checklist to ensure smooth migration:

- [ ] Run existing tests: `python -m pytest tests/ -v`
- [ ] Verify default training works: `python train.py`
- [ ] (Optional) Try MIPROv2: `python train.py --optimizer miprov2 --auto light`
- [ ] (Optional) Rename existing model: `mv trained_qa_model.json trained_qa_model_bootstrap.json`
- [ ] (Optional) Update model loading paths in your code
- [ ] (Optional) Train both models and compare
- [ ] Read USAGE.md for new features

---

**Need Help?**

- Run `python train.py --help` for CLI options
- See [README.md](README.md) for overview
- See [USAGE.md](USAGE.md) for detailed usage guide
