"""Training script for DSPy QA system using BootstrapFewShot."""

import os
import dspy
from dotenv import load_dotenv
from qa_module import QAModule, context_adherence_metric
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
    metric=context_adherence_metric,
    max_labeled_demos=4,
    max_bootstrapped_demos=4,
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

# Print demonstrations used
demos = trained_qa.generate_answer.predict.demos
print(f"Demonstrations selected by optimizer: {len(demos)}")
print()

for i, demo in enumerate(demos, 1):
    print(f"  Demo {i}:")
    print(f"    Context: {demo['context'][:50]}...")
    print(f"    Question: {demo['question']}")
    print(f"    Answer: {demo['answer']}")
    print()

print("═══════════════════════════════════════════════════════════════")
