# DSPyPrompting Usage Guide

This guide provides detailed usage instructions for the DSPyPrompting QA system.

## Table of Contents

- [Quick Start](#quick-start)
- [Training](#training)
- [Optimizer Selection](#optimizer-selection)
- [Testing](#testing)
- [Common Workflows](#common-workflows)
- [Troubleshooting](#troubleshooting)

---

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd DSPyPrompting

# Install dependencies
pip install -r requirements.txt

# Configure API key
echo "GROQ_API_KEY=your_api_key_here" > .env
```

### 2. Basic Usage

```bash
# Train with default optimizer (BootstrapFewShot)
python train.py

# Test the trained model
python test.py
```

---

## Training

### BootstrapFewShot (Recommended for Development)

**Fast training for quick iteration:**

```bash
python train.py
```

**What it does:**
- Optimizes few-shot examples only
- Uses hallucination-aware metric
- Saves to `trained_qa_model_bootstrap.json`
- Takes ~2-3 minutes

**When to use:**
- Rapid prototyping
- Development and testing
- When you need quick results
- Limited API quota

### MIPROv2 (Recommended for Production)

**High-quality training with instruction optimization:**

```bash
# Light mode (fastest MIPROv2)
python train.py --optimizer miprov2

# Medium mode (balanced)
python train.py --optimizer miprov2 --auto medium

# Heavy mode (best quality)
python train.py --optimizer miprov2 --auto heavy
```

**What it does:**
- Optimizes both instructions AND few-shot examples
- Uses hallucination-aware metric
- Saves to `trained_qa_model_miprov2.json`
- Takes ~5-15 minutes depending on auto mode

**When to use:**
- Production deployments
- Maximum accuracy required
- When you have time for longer training
- Sufficient API quota available

### Parallel Training

**Speed up MIPROv2 with multi-threading:**

```bash
# Use 8 threads (adjust based on your CPU)
python train.py --optimizer miprov2 --num-threads 8
```

**Tips:**
- Set `--num-threads` to your CPU core count for best performance
- More threads = more API calls (watch your quota)
- Typical values: 4, 8, 16

---

## Optimizer Selection

### Choosing the Right Optimizer

| Use Case | Recommended Optimizer | Command |
|----------|----------------------|---------|
| Quick iteration | BootstrapFewShot | `python train.py` |
| Development | BootstrapFewShot | `python train.py` |
| Production | MIPROv2 medium | `python train.py --optimizer miprov2 --auto medium` |
| Maximum accuracy | MIPROv2 heavy | `python train.py --optimizer miprov2 --auto heavy` |
| Limited time | BootstrapFewShot | `python train.py` |
| Limited quota | BootstrapFewShot | `python train.py` |

### Understanding Auto Modes

MIPROv2's `--auto` parameter controls optimization intensity:

**Light (Default)**
- Fastest MIPROv2 training
- Good for initial MIPROv2 experiments
- ~5 minutes

**Medium**
- Balanced quality vs speed
- Recommended for most use cases
- ~10 minutes

**Heavy**
- Best possible quality
- Longest training time
- ~15+ minutes

### Command Reference

```bash
python train.py [OPTIONS]

Options:
  --optimizer {bootstrap,miprov2}  Optimizer to use (default: bootstrap)
  --auto {light,medium,heavy}       MIPROv2 auto mode (only for --optimizer miprov2)
  --num-threads INT                Number of threads for MIPROv2 (only for --optimizer miprov2)
  -h, --help                       Show help message
```

**Examples:**

```bash
# Get help
python train.py --help

# BootstrapFewShot (default)
python train.py
python train.py --optimizer bootstrap

# MIPROv2 with defaults (auto=light)
python train.py --optimizer miprov2

# MIPROv2 with medium auto
python train.py --optimizer miprov2 --auto medium

# MIPROv2 with heavy auto and 8 threads
python train.py --optimizer miprov2 --auto heavy --num-threads 8
```

---

## Testing

### Run All Tests

```bash
# Run complete test suite
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ --cov=. --cov-report=html
```

### Run Specific Test Categories

```bash
# Unit tests only
python -m pytest tests/unit/ -v

# Integration tests only
python -m pytest tests/integration/ -v

# Edge case tests only
python -m pytest tests/edge_cases/ -v
```

### Run Specific Test Files

```bash
# Test optimizer selection
python -m pytest tests/unit/test_optimizer_selection.py -v

# Test optimizer integration
python -m pytest tests/integration/test_optimizer_integration.py -v

# Test optimizer comparison
python -m pytest tests/integration/test_optimizer_comparison.py -v
```

### Expected Test Results

- **Total tests:** 250
- **Expected pass rate:** 100%
- **Test execution time:** ~1-2 seconds

---

## Common Workflows

### Workflow 1: Train and Test (BootstrapFewShot)

```bash
# Step 1: Train with BootstrapFewShot
python train.py

# Step 2: Run tests
python test.py

# Step 3: Check results
ls -lh trained_qa_model_bootstrap.json
```

### Workflow 2: Train and Test (MIPROv2)

```bash
# Step 1: Train with MIPROv2 (medium mode)
python train.py --optimizer miprov2 --auto medium

# Step 2: Run tests
python test.py

# Step 3: Compare with BootstrapFewShot
ls -lh trained_qa_model_*.json
```

### Workflow 3: Compare Optimizers

```bash
# Step 1: Train BootstrapFewShot
python train.py

# Step 2: Train MIPROv2
python train.py --optimizer miprov2 --auto medium

# Step 3: Compare models
python compare_optimizers.py
```

### Workflow 4: Development Cycle

```bash
# 1. Make code changes
vim qa_module.py

# 2. Run tests
python -m pytest tests/ -v

# 3. Quick train with BootstrapFewShot
python train.py

# 4. Verify quality
python test.py

# 5. Production train with MIPROv2 (when satisfied)
python train.py --optimizer miprov2 --auto medium
```

---

## Troubleshooting

### Issue: Training Fails with API Error

**Symptoms:**
```
ValueError: GROQ_API_KEY not found in environment variables
```

**Solution:**
```bash
# Check if .env file exists
ls -la .env

# If not, create it
echo "GROQ_API_KEY=your_actual_key_here" > .env

# Verify key is set
cat .env
```

### Issue: Slow Training

**Symptoms:** Training takes much longer than expected

**Solutions:**

1. **Use faster optimizer:**
   ```bash
   # Switch from MIPROv2 to BootstrapFewShot
   python train.py
   ```

2. **Use lighter MIPROv2 mode:**
   ```bash
   # Use light instead of medium/heavy
   python train.py --optimizer miprov2 --auto light
   ```

3. **Reduce dataset size:**
   ```python
   # In dataset.py, temporarily reduce trainset size
   trainset = trainset[:10]  # Use only 10 examples
   ```

### Issue: Out of Memory

**Symptoms:** Python crashes or shows memory error

**Solutions:**

1. **Use smaller model:**
   ```python
   # In train.py, change model to smaller variant
   llm = dspy.LM("groq/llama-3.1-8b-instant", api_key=api_key)
   ```

2. **Reduce demo count:**
   ```python
   # In train.py, reduce max_bootstrapped_demos
   "max_bootstrapped_demos": 2  # Instead of 4
   ```

### Issue: Model Quality Poor

**Symptoms:** Model gives wrong answers or hallucinates

**Solutions:**

1. **Switch to MIPROv2:**
   ```bash
   python train.py --optimizer miprov2 --auto medium
   ```

2. **Use heavier auto mode:**
   ```bash
   python train.py --optimizer miprov2 --auto heavy
   ```

3. **Check metric function:**
   ```python
   # In qa_module.py, verify hallucination_aware_metric is correct
   print(hallucination_aware_metric(gold_example, pred_example))
   ```

### Issue: Tests Failing

**Symptoms:** pytest shows failures

**Solutions:**

1. **Run tests in verbose mode:**
   ```bash
   python -m pytest tests/ -v --tb=short
   ```

2. **Run specific test to debug:**
   ```bash
   python -m pytest tests/unit/test_optimizer_selection.py::TestCLIArgumentParsing::test_default_optimizer_is_bootstrap -v
   ```

3. **Check for dependency issues:**
   ```bash
   pip install -r requirements.txt --upgrade
   ```

### Issue: Help Message Not Clear

**Symptoms:** Don't know which arguments to use

**Solution:**
```bash
python train.py --help
```

This shows all available options with descriptions.

---

## Advanced Usage

### Custom Training Configuration

**Modify optimizer parameters in train.py:**

```python
# For BootstrapFewShot
optimizer = dspy.BootstrapFewShot(
    metric=hallucination_aware_metric,
    max_labeled_demos=6,      # Increase for more examples
    max_bootstrapped_demos=4,  # Increase for more diversity
    max_rounds=1,              # Keep at 1 for this project
    max_errors=10              # Increase to allow more errors
)

# For MIPROv2
optimizer = dspy.MIPROv2(
    metric=hallucination_aware_metric,
    auto="medium",              # light/medium/heavy
    max_bootstrapped_demos=4,   # Increase for more examples
    max_labeled_demos=6,        # Increase for more examples
    max_errors=10,              # Increase to allow more errors
    num_threads=8               # Adjust based on CPU cores
)
```

### Using Different Models

**Switch to a different Groq model:**

```python
# In train.py, change the model
llm = dspy.LM(
    "groq/llama-3.1-70b-versatile",  # Larger model
    api_key=api_key
)
```

**Available Groq models:**
- `groq/llama-3.1-8b-instant` (default, fast)
- `groq/llama-3.1-70b-versatile` (slower, better quality)
- `groq/mixtral-8x7b-32768` (good balance)

---

## Tips and Best Practices

1. **Start with BootstrapFewShot** for rapid iteration
2. **Use MIPROv2 medium** for production deployments
3. **Monitor API quota** when using MIPROv2 with many threads
4. **Run tests** before and after code changes
5. **Compare models** using compare_optimizers.py
6. **Save multiple models** with different optimizers for comparison
7. **Use version control** to track model performance changes

---

## Getting Help

- **Usage:** `python train.py --help`
- **Tests:** `python -m pytest tests/ -v`
- **Documentation:** See README.md and MIGRATION.md
- **Issues:** Check GitHub Issues or create a new one
