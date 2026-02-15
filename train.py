"""Training script for DSPy QA system using BootstrapFewShot."""

import os
import dspy
from dotenv import load_dotenv
from qa_module import QAModule, answer_exact_match_metric
from dataset import trainset

# Load environment variables
load_dotenv()

# Get Groq API key
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY not found in environment variables")

# Configure DSPy with Groq Llama model (DSPy 3.x format)
llm = dspy.LM(
    "groq/llama-3.1-8b-instant",
    api_key=api_key
)
dspy.configure(lm=llm)

print("═══════════════════════════════════════════════════════════════")
print("           DSPy QA Training - BootstrapFewShot")
print("═══════════════════════════════════════════════════════════════")
print(f"Model: llama-3.1-8b-instant")
print(f"Training samples: {len(trainset)}")
print()

# Initialize untrained QA module
qa_module = QAModule()

# Configure BootstrapFewShot optimizer
optimizer = dspy.BootstrapFewShot(
    metric=answer_exact_match_metric,
    max_labeled_demos=3,
    max_bootstrapped_demos=3,
    max_rounds=1,
    max_errors=10
)

print("Training started...")
print()

# Train the model
trained_qa = optimizer.compile(
    student=qa_module,
    trainset=trainset
)

print()
print("═══════════════════════════════════════════════════════════════")
print("                   Training Complete")
print("═══════════════════════════════════════════════════════════════")
print()

# Save trained model
model_path = "trained_qa_model.json"
trained_qa.save(model_path)
print(f"✅ Trained model saved to: {model_path}")
print()
print("═══════════════════════════════════════════════════════════════")
