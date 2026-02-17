"""Training script for DSPy QA system using BootstrapFewShot."""

import os
import argparse
import dspy
from dotenv import load_dotenv
from qa_module import QAModule, hallucination_aware_metric
from dataset import trainset

# Load environment variables
load_dotenv()

# Get Gemini API key
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not found in environment variables")

# Configure DSPy with Gemini 1.5 Flash (DSPy 3.x format)
llm = dspy.LM(
    "google/gemini-flash-1.5",
    api_key=api_key
)
dspy.configure(lm=llm)

# Parse command-line arguments
parser = argparse.ArgumentParser(
    description="Train DSPy QA model with configurable optimizer",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
Examples:
  python train.py                          # Use BootstrapFewShot (default)
  python train.py --optimizer miprov2      # Use MIPROv2 optimizer
  python train.py --optimizer miprov2 --auto medium  # MIPROv2 with medium auto
  python train.py --optimizer bootstrap    # Explicit BootstrapFewShot
    """
)

parser.add_argument(
    "--optimizer",
    choices=["bootstrap", "miprov2"],
    default="bootstrap",
    help="Optimizer to use (default: bootstrap)"
)

parser.add_argument(
    "--auto",
    choices=["light", "medium", "heavy"],
    default=None,
    help="MIPROv2 auto mode: light, medium, or heavy (only for --optimizer miprov2)"
)

parser.add_argument(
    "--num-threads",
    type=int,
    default=None,
    help="Number of threads for parallel optimization (only for MIPROv2)"
)

if __name__ == "__main__":
    args = parser.parse_args()

    optimizer_name = "MIPROv2" if args.optimizer == "miprov2" else "BootstrapFewShot"
    print("═══════════════════════════════════════════════════════════════")
    print(f"           DSPy QA Training - {optimizer_name}")
    if args.optimizer == "miprov2":
        auto_mode = args.auto or "light"
        print(f"           Auto Mode: {auto_mode}")
    print("═══════════════════════════════════════════════════════════════")
    print(f"Model: gemini-flash-1.5")
    print(f"Training samples: {len(trainset)}")
    positive_count = sum(1 for s in trainset if "not provided" not in s.answer)
    negative_count = sum(1 for s in trainset if "not provided" in s.answer)
    print(f"  Positive: {positive_count}")
    print(f"  Negative: {negative_count}")
    print()

    # Initialize untrained QA module
    qa_module = QAModule()

    # Configure optimizer based on user selection
    if args.optimizer == "miprov2":
        # Validate auto parameter for MIPROv2
        auto_mode = args.auto or "light"

        # Build MIPROv2 kwargs
        mipro_kwargs = {
            "metric": hallucination_aware_metric,
            "auto": auto_mode,
            "max_bootstrapped_demos": 4,
            "max_labeled_demos": 6,
            "max_errors": 10
        }

        # Add optional parameters
        if args.num_threads:
            mipro_kwargs["num_threads"] = args.num_threads

        print(f"Configuring MIPROv2 optimizer:")
        print(f"  Auto mode: {auto_mode}")
        if args.num_threads:
            print(f"  Threads: {args.num_threads}")

        optimizer = dspy.MIPROv2(**mipro_kwargs)

    else:  # bootstrap (default)
        print("Configuring BootstrapFewShot optimizer")
        optimizer = dspy.BootstrapFewShot(
            metric=hallucination_aware_metric,
            max_labeled_demos=6,
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
    optimizer_suffix = "miprov2" if args.optimizer == "miprov2" else "bootstrap"
    model_path = f"trained_qa_model_{optimizer_suffix}.json"
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
