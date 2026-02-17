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

```bash
python train.py
```

**Expected output:**
- Training samples: 15
- Demonstrations selected: 6
- Model saved to: trained_qa_model.json

### 4. Test the Model

```bash
python test.py
```

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
