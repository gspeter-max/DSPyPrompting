"""Compare trained models from different optimizers."""

import os
import dspy
from dotenv import load_dotenv
from qa_module import QAModule
from dataset import trainset

# Load environment variables
load_dotenv()

# Get Groq API key
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY not found in environment variables")

# Configure DSPy
llm = dspy.LM("groq/llama-3.1-8b-instant", api_key=api_key)
dspy.configure(lm=llm)


def load_model(model_path: str) -> QAModule:
    """Load a trained model from disk."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    qa = QAModule()
    qa.load(model_path)
    return qa


def evaluate_model(model: QAModule, testset) -> dict:
    """Evaluate a model on a test set and return metrics."""
    from qa_module import hallucination_aware_metric

    correct = 0
    total = len(testset)
    scores = []

    for example in testset:
        # Get prediction
        pred = model(context=example.context, question=example.question)

        # Calculate score
        score = hallucination_aware_metric(example, pred)
        scores.append(score)

        # Count exact matches
        if pred.answer == example.answer:
            correct += 1

    accuracy = correct / total
    avg_score = sum(scores) / len(scores) if scores else 0

    return {
        "accuracy": accuracy,
        "avg_metric_score": avg_score,
        "total_examples": total,
        "exact_matches": correct
    }


def main():
    """Compare BootstrapFewShot and MIPROv2 models."""
    print("═══════════════════════════════════════════════════════════════")
    print("                 Optimizer Comparison Tool")
    print("═══════════════════════════════════════════════════════════════")
    print()

    # Model paths
    bootstrap_model_path = "trained_qa_model_bootstrap.json"
    miprov2_model_path = "trained_qa_model_miprov2.json"

    # Check which models exist
    bootstrap_exists = os.path.exists(bootstrap_model_path)
    miprov2_exists = os.path.exists(miprov2_model_path)

    if not bootstrap_exists and not miprov2_exists:
        print("❌ No trained models found!")
        print()
        print("Please train models first:")
        print("  python train.py                           # Train BootstrapFewShot")
        print("  python train.py --optimizer miprov2       # Train MIPROv2")
        return

    print("Model Status:")
    print(f"  BootstrapFewShot: {'✅ Found' if bootstrap_exists else '❌ Not found'}")
    print(f"  MIPROv2:           {'✅ Found' if miprov2_exists else '❌ Not found'}")
    print()

    # Load available models
    models = {}
    if bootstrap_exists:
        print(f"Loading {bootstrap_model_path}...")
        models['BootstrapFewShot'] = load_model(bootstrap_model_path)

    if miprov2_exists:
        print(f"Loading {miprov2_model_path}...")
        models['MIPROv2'] = load_model(miprov2_model_path)

    print()

    # Evaluate on training set
    print("Evaluating models on training set...")
    print()

    results = {}
    for name, model in models.items():
        print(f"Evaluating {name}...")
        results[name] = evaluate_model(model, trainset)

    print()
    print("═══════════════════════════════════════════════════════════════")
    print("                      Results")
    print("═══════════════════════════════════════════════════════════════")
    print()

    # Print comparison table
    print(f"{'Metric':<30} {'BootstrapFewShot':<20} {'MIPROv2':<20}")
    print("─" * 70)

    # Accuracy
    bootstrap_acc = results.get('BootstrapFewShot', {}).get('accuracy', None)
    miprov2_acc = results.get('MIPROv2', {}).get('accuracy', None)

    bootstrap_acc_str = f"{bootstrap_acc:.1%}" if bootstrap_acc is not None else "N/A"
    miprov2_acc_str = f"{miprov2_acc:.1%}" if miprov2_acc is not None else "N/A"

    print(f"{'Exact Match Accuracy':<30} {bootstrap_acc_str:<20} {miprov2_acc_str:<20}")

    # Average metric score
    bootstrap_score = results.get('BootstrapFewShot', {}).get('avg_metric_score', None)
    miprov2_score = results.get('MIPROv2', {}).get('avg_metric_score', None)

    bootstrap_score_str = f"{bootstrap_score:.3f}" if bootstrap_score is not None else "N/A"
    miprov2_score_str = f"{miprov2_score:.3f}" if miprov2_score is not None else "N/A"

    print(f"{'Avg Metric Score':<30} {bootstrap_score_str:<20} {miprov2_score_str:<20}")

    # Exact matches
    bootstrap_matches = results.get('BootstrapFewShot', {}).get('exact_matches', None)
    miprov2_matches = results.get('MIPROv2', {}).get('exact_matches', None)

    bootstrap_matches_str = f"{bootstrap_matches}/15" if bootstrap_matches is not None else "N/A"
    miprov2_matches_str = f"{miprov2_matches}/15" if miprov2_matches is not None else "N/A"

    print(f"{'Exact Matches':<30} {bootstrap_matches_str:<20} {miprov2_matches_str:<20}")

    print()
    print("═══════════════════════════════════════════════════════════════")
    print()

    # Determine winner
    if bootstrap_exists and miprov2_exists:
        print("Comparison:")
        print()

        if bootstrap_acc and miprov2_acc:
            if miprov2_acc > bootstrap_acc:
                diff = (miprov2_acc - bootstrap_acc) * 100
                print(f"  ✅ MIPROv2 is {diff:.1f}% more accurate")
            elif bootstrap_acc > miprov2_acc:
                diff = (bootstrap_acc - miprov2_acc) * 100
                print(f"  ✅ BootstrapFewShot is {diff:.1f}% more accurate")
            else:
                print(f"  ⚖️  Both optimizers have equal accuracy")

        if bootstrap_score and miprov2_score:
            if miprov2_score > bootstrap_score:
                diff = miprov2_score - bootstrap_score
                print(f"  ✅ MIPROv2 has {diff:.3f} higher metric score")
            elif bootstrap_score > miprov2_score:
                diff = bootstrap_score - miprov2_score
                print(f"  ✅ BootstrapFewShot has {diff:.3f} higher metric score")
            else:
                print(f"  ⚖️  Both optimizers have equal metric scores")

        print()
        print("Recommendation:")

        # Simple recommendation logic
        if miprov2_acc and bootstrap_acc:
            if miprov2_acc >= bootstrap_acc:
                print("  → Use MIPROv2 for production (better or equal accuracy)")
            else:
                print("  → BootstrapFewShot performed better - consider using it")

        print("  → Use BootstrapFewShot for development (faster training)")
        print()

    print("═══════════════════════════════════════════════════════════════")


if __name__ == "__main__":
    main()
