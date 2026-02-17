# DSPy Context-Based QA System

A hands-on implementation of **DSPy framework** demonstrating context-based question answering using BootstrapFewShot optimization.

## 🎯 Purpose

Learn DSPy concepts by building a working QA system:
- **Declarative programming** over prompt engineering
- **Automatic prompt optimization** using BootstrapFewShot
- **Metric-driven training** with labeled examples
- **Model-agnostic design** using Groq's free Llama API

## 📋 Prerequisites

- Python 3.8+
- Groq API key (free tier: [console.groq.com](https://console.groq.com))

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API Key

Create a `.env` file with your Groq API key:

```bash
GROQ_API_KEY=your_api_key_here
```

### 3. Train the Model

**Option A: Quick training (BootstrapFewShot - default)**
```bash
python train.py
```

**Option B: Best quality (MIPROv2)**
```bash
python train.py --optimizer miprov2 --auto medium
```

**Expected output:**
- Training samples: 15
- Demonstrations selected: 4-6
- Model saved to: `trained_qa_model_bootstrap.json` or `trained_qa_model_miprov2.json`

### 4. Test the Model

```bash
python test.py
```

## 🔄 Optimizer Selection

DSPyPrompting supports two optimizers with different trade-offs:

### BootstrapFewShot (Default)

**Best for:** Quick iteration, baseline models, development

**Method:** Optimizes few-shot examples only

**Speed:** Fast (~2-3 minutes)

**Usage:**
```bash
python train.py                    # Uses BootstrapFewShot (default)
python train.py --optimizer bootstrap  # Explicit BootstrapFewShot
```

### MIPROv2

**Best for:** Production models, maximum accuracy, refined prompts

**Method:** Optimizes both instructions (system prompts) AND few-shot examples

**Speed:** Slower (~5-15 minutes depending on auto mode)

**Usage:**
```bash
# Light mode (fastest MIPROv2)
python train.py --optimizer miprov2

# Medium mode (balanced)
python train.py --optimizer miprov2 --auto medium

# Heavy mode (best quality)
python train.py --optimizer miprov2 --auto heavy

# With threading (faster on multi-core)
python train.py --optimizer miprov2 --num-threads 8
```

### Comparison

| Feature | BootstrapFewShot | MIPROv2 |
|---------|------------------|---------|
| Optimizes Examples | ✅ | ✅ |
| Optimizes Instructions | ❌ | ✅ |
| Training Speed | Fast | Medium-Slow |
| Model Quality | Good | Better |
| API Calls | Fewer | More |
| Parallel Processing | ❌ | ✅ (via num_threads) |

## 📁 Project Structure

```
dataset.py              # Training data (9 positive, 6 negative examples)
qa_module.py            # Model definition + metrics
train.py                # Training script
test.py                 # Test suite
trained_qa_model.json   # Saved trained model
```

## 🔑 Key Concepts

- **Few-shot learning**: Model learns from demonstrations
- **Chain-of-Thought**: Model generates reasoning before answering
- **BootstrapFewShot**: Optimizer selects best examples for demonstrations
- **Hallucination-aware metric**: Penalizes model for making up answers

## 📊 Results

- Training accuracy: ≥80%
- Sample 11 (hallucination fix): ✅ PASSING
- Negative examples: 4/6 correctly refusing (67% - up from 0%)
