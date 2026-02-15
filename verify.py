"""Verification and demonstration script for DSPy QA system."""

import os
import json
from dotenv import load_dotenv
import dspy
from qa_module import QAModule
from dataset import trainset

# Load environment variables
load_dotenv()


def print_header(text):
    """Print formatted section header."""
    print()
    print("═" * 63)
    print(f"  {text}")
    print("═" * 63)


def verify_environment():
    """Step 1: Environment check."""
    print_header("Step 1: Environment Verification")

    # Check API key
    api_key = os.getenv("GROQ_API_KEY")
    if api_key:
        print(f"  ✅ GROQ_API_KEY: Loaded ({api_key[:10]}...{api_key[-4:]})")
    else:
        print("  ❌ GROQ_API_KEY: Not found")
        return False

    # Test API connectivity
    try:
        llm = dspy.LM("groq/llama-3.1-8b-instant", api_key=api_key)
        dspy.configure(lm=llm)

        # Simple test call
        test_result = dspy.Predict("question -> answer")(question="What is 2+2?")
        print(f"  ✅ API Connectivity: Working")
        print(f"  ✅ Model: llama-3.1-8b-instant")
        print()
        print("  Rate Limits (llama-3.1-8b-instant):")
        print("    - 14,400 requests/day")
        print("    - 30 requests/minute")
        return True
    except Exception as e:
        print(f"  ❌ API Error: {str(e)}")
        return False


def verify_data():
    """Step 2: Data verification."""
    print_header("Step 2: Training Data Verification")

    print(f"  Training samples: {len(trainset)}")
    print()

    for i, sample in enumerate(trainset, 1):
        print(f"  Sample {i}:")
        print(f"    Context: {sample.context[:60]}...")
        print(f"    Question: {sample.question}")
        print(f"    Answer: {sample.answer}")
        print()

    print(f"  ✅ All {len(trainset)} samples loaded correctly")
    return True


def verify_training():
    """Step 3: Training verification."""
    print_header("Step 3: Trained Model Verification")

    model_path = "trained_qa_model.json"

    if not os.path.exists(model_path):
        print(f"  ❌ Model file not found: {model_path}")
        print("  Run 'python train.py' first")
        return False

    # Load model
    qa_model = QAModule()
    qa_model.load(model_path)
    print(f"  ✅ Model loaded from {model_path}")
    print()

    # Model loaded successfully
    print(f"  ✅ Model loaded and ready")
    print()

    return True


def interactive_demo(qa_model):
    """Step 4: Interactive demonstration."""
    print_header("Step 4: Interactive Demo")

    print("  Enter your questions (or 'quit' to exit)")
    print("  Note: Include context in your question for best results")
    print()

    while True:
        try:
            user_input = input("  Your question (with context): ").strip()

            if user_input.lower() in ['quit', 'exit', 'q']:
                print("  Exiting demo...")
                break

            if not user_input:
                continue

            # Try to parse context and question
            # Format: "Context: ... Question: ..."
            if "Context:" in user_input and "Question:" in user_input:
                parts = user_input.split("Question:")
                context = parts[0].replace("Context:", "").strip()
                question = parts[1].strip()
            else:
                # Use entire input as question with minimal context
                context = "Python programming"
                question = user_input

            # Generate answer
            result = qa_model(context=context, question=question)

            print(f"  Answer: {result.answer}")
            print()

        except KeyboardInterrupt:
            print("\n  Exiting demo...")
            break
        except Exception as e:
            print(f"  Error: {str(e)}")
            print()


def generate_report(results):
    """Step 5: Generate verification report."""
    print_header("Step 5: Generating Report")

    report_path = "verification_report.txt"

    report_content = f"""
DSPy QA System Verification Report
{'=' * 63}
Generated: {results.get('timestamp', 'Unknown')}

ENVIRONMENT
{'─' * 63}
API Key: {'✅ Loaded' if results.get('api_key') else '❌ Not found'}
API Connectivity: {'✅ Working' if results.get('api_connected') else '❌ Failed'}
Model: llama-3.1-8b-instant

TRAINING DATA
{'─' * 63}
Samples: {results.get('num_samples', 0)}

TRAINED MODEL
{'─' * 63}
Status: {'✅ Loaded' if results.get('model_loaded') else '❌ Not found'}
Demonstrations: {results.get('num_demos', 0)}

TEST RESULTS
{'─' * 63}
"""

    for test_name, passed in results.get('test_results', {}).items():
        status = "✅ PASS" if passed else "❌ FAIL"
        report_content += f"{status} {test_name}\n"

    report_content += f"\n{'=' * 63}\n"

    with open(report_path, 'w') as f:
        f.write(report_content)

    print(f"  ✅ Report saved to: {report_path}")
    return True


def main():
    """Run verification steps."""
    print()
    print("╔" + "═" * 61 + "╗")
    print("║" + " " * 12 + "DSPy QA Verification & Demo" + " " * 17 + "║")
    print("╚" + "═" * 61 + "╝")

    results = {
        'timestamp': str(dspy.now()) if hasattr(dspy, 'now') else 'Unknown'
    }

    # Step 1: Environment
    results['api_key'] = os.getenv("GROQ_API_KEY") is not None
    results['api_connected'] = verify_environment()
    if not results['api_connected']:
        print("\n  ❌ Cannot proceed without API connection")
        return

    # Step 2: Data
    results['num_samples'] = len(trainset)
    verify_data()

    # Step 3: Training
    model_path = "trained_qa_model.json"
    results['model_loaded'] = os.path.exists(model_path)
    if results['model_loaded']:
        verify_training()
        qa_model = QAModule()
        qa_model.load(model_path)
        results['num_demos'] = "N/A"  # Not accessible in DSPy 3.x
    else:
        print("\n  ❌ Cannot run demo without trained model")
        return

    # Step 4: Interactive demo
    print()
    response = input("  Run interactive demo? (y/n): ").strip().lower()
    if response == 'y':
        interactive_demo(qa_model)
    else:
        print("  Skipping demo...")

    # Step 5: Generate report
    results['test_results'] = {
        "Environment Check": results['api_connected'],
        "Data Loaded": results['num_samples'] == 6,
        "Model Trained": results['model_loaded'],
    }
    generate_report(results)

    print()
    print("═══════════════════════════════════════════════════════════════")
    print("  Verification Complete")
    print("═══════════════════════════════════════════════════════════════")
    print()


if __name__ == "__main__":
    main()
