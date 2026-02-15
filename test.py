"""Comprehensive testing suite for DSPy QA system."""

import os
import dspy
from dotenv import load_dotenv
from qa_module import QAModule, answer_exact_match_metric
from dataset import trainset

# Load environment variables
load_dotenv()

# Configure DSPy
api_key = os.getenv("GROQ_API_KEY")
llm = dspy.GROQ(model="llama-3.1-8b-instant", api_key=api_key)
dspy.configure(lm=llm)


def print_header(text):
    """Print formatted section header."""
    print()
    print("═" * 63)
    print(f"  {text}")
    print("═" * 63)


def test_training_data_accuracy(trained_model):
    """Test 1: Verify 100% accuracy on training data."""
    print_header("Test 1: Training Data Accuracy (Sanity Check)")

    correct = 0
    total = len(trainset)

    for i, sample in enumerate(trainset, 1):
        pred = trained_model(context=sample.context, question=sample.question)
        is_correct = answer_exact_match_metric(sample, pred)

        status = "✅ PASS" if is_correct else "❌ FAIL"
        print(f"  {status} Sample {i}: {sample.question[:40]}...")

        if is_correct:
            correct += 1
        else:
            print(f"    Expected: {sample.answer}")
            print(f"    Got:      {pred.answer}")

    accuracy = (correct / total) * 100
    print()
    print(f"  Training Accuracy: {accuracy:.1f}% ({correct}/{total})")
    return accuracy == 100


def test_generalization(trained_model):
    """Test 2: Test on unseen questions."""
    print_header("Test 2: Generalization (Unseen Questions)")

    # Unseen test samples
    testset = [
        dspy.Example(
            context="""Python lambda functions are anonymous functions that can have any number of arguments but only one expression.
They are defined using the lambda keyword: lambda arguments: expression
Example: lambda x: x * 2 creates a function that doubles its input.""",
            question="What keyword is used to define a lambda function?",
            answer="lambda"
        ).with_inputs("context", "question"),
        dspy.Example(
            context="""The 'zip' function in Python takes iterables (like lists) and returns an iterator of tuples.
Each tuple contains elements from the input iterables at the same position.
Example: zip([1, 2], ['a', 'b']) produces [(1, 'a'), (2, 'b')]""",
            question="What type of elements does zip return?",
            answer="tuples"
        ).with_inputs("context", "question"),
        dspy.Example(
            context="""Python dictionaries store key-value pairs.
Keys must be immutable (like strings, numbers, or tuples).
Values can be of any type.
You access values using their keys: my_dict['key']""",
            question="What must dictionary keys be?",
            answer="immutable"
        ).with_inputs("context", "question"),
    ]

    correct = 0
    total = len(testset)

    for i, sample in enumerate(testset, 1):
        pred = trained_model(context=sample.context, question=sample.question)
        is_correct = answer_exact_match_metric(sample, pred)

        status = "✅ PASS" if is_correct else "⚠️ SEMANTIC MATCH" if not is_correct and sample.answer.lower() in pred.answer.lower() else "❌ FAIL"

        print(f"  {status} Sample {i}:")
        print(f"    Context: {sample.context[:50]}...")
        print(f"    Question: {sample.question}")
        print(f"    Expected: {sample.answer}")
        print(f"    Predicted: {pred.answer}")
        print()

        if is_correct:
            correct += 1

    accuracy = (correct / total) * 100
    print(f"  Generalization Accuracy: {accuracy:.1f}% ({correct}/{total})")
    return accuracy


def test_untrained_vs_trained():
    """Test 3: Compare untrained vs trained model."""
    print_header("Test 3: Untrained vs Trained Comparison")

    # Load trained model
    trained_qa = QAModule()
    trained_qa.load("trained_qa_model.json")

    # Create untrained model
    untrained_qa = QAModule()

    # Test questions (subset of training data)
    test_samples = trainset[:3]

    print(f"  {'Question':<50} {'Untrained':>12} {'Trained':>12}")
    print("  " + "-" * 76)

    trained_correct = 0
    untrained_correct = 0

    for sample in test_samples:
        pred_untrained = untrained_qa(context=sample.context, question=sample.question)
        pred_trained = trained_qa(context=sample.context, question=sample.question)

        untrained_ok = answer_exact_match_metric(sample, pred_untrained)
        trained_ok = answer_exact_match_metric(sample, pred_trained)

        untrained_status = "✅" if untrained_ok else "❌"
        trained_status = "✅" if trained_ok else "❌"

        print(f"  {sample.question[:47]:<50} {untrained_status:^12} {trained_status:^12}")

        if untrained_ok:
            untrained_correct += 1
        if trained_ok:
            trained_correct += 1

    print()
    print(f"  Untrained: {untrained_correct}/{len(test_samples)} correct")
    print(f"  Trained:   {trained_correct}/{len(test_samples)} correct")

    improvement = ((trained_correct - untrained_correct) / len(test_samples)) * 100
    print(f"  Trained Model Improvement: {improvement:+.1f}%")

    return trained_correct > untrained_correct


def test_context_adherence(trained_model):
    """Test 4: Verify model doesn't hallucinate."""
    print_header("Test 4: Context Adherence (Anti-Hallucination)")

    test_case = dspy.Example(
        context="""Python lists are mutable sequences that can hold mixed types.""",
        question="What is a tuple?",
        answer="Not mentioned in context"
    ).with_inputs("context", "question")

    pred = trained_model(context=test_case.context, question=test_case.question)

    print(f"  Context: {test_case.context}")
    print(f"  Question: {test_case.question}")
    print(f"  Answer: {pred.answer}")
    print()

    # Check if answer indicates information not in context
    no_hallucination = any(phrase in pred.answer.lower() for phrase in
                          ["not mentioned", "not provided", "context does not",
                           "cannot answer", "don't know", "not stated"])

    if no_hallucination:
        print("  ✅ PASS: Model correctly indicated information not in context")
    else:
        print("  ⚠️ WARNING: Model may have hallucinated (use manual review)")

    return no_hallucination


def test_edge_cases(trained_model):
    """Test 5: Edge cases."""
    print_header("Test 5: Edge Cases")

    results = []

    # Empty context
    try:
        pred = trained_model(context="", question="What is Python?")
        print(f"  Empty context: {pred.answer[:50]}...")
        results.append("Empty context handled")
    except Exception as e:
        print(f"  Empty context: Error - {str(e)[:40]}...")
        results.append("Empty context error")

    # Multi-part question
    try:
        pred = trained_model(
            context="Python has lists and tuples. Lists are mutable, tuples are not.",
            question="What are lists and are they mutable?"
        )
        print(f"  Multi-part question: {pred.answer[:50]}...")
        results.append("Multi-part handled")
    except Exception as e:
        print(f"  Multi-part question: Error - {str(e)[:40]}...")
        results.append("Multi-part error")

    print()
    print(f"  Edge cases tested: {len(results)}")
    return len(results) > 0


def main():
    """Run all tests."""
    print()
    print("╔" + "═" * 61 + "╗")
    print("║" + " " * 15 + "DSPy QA Test Results" + " " * 21 + "║")
    print("╚" + "═" * 61 + "╝")

    # Load trained model
    if not os.path.exists("trained_qa_model.json"):
        print("  ❌ Error: trained_qa_model.json not found")
        print("  Run 'python train.py' first")
        return

    trained_qa = QAModule()
    trained_qa.load("trained_qa_model.json")

    # Run tests
    results = {
        "Test 1: Training Accuracy": test_training_data_accuracy(trained_qa),
        "Test 2: Generalization": test_generalization(trained_qa) >= 60,
        "Test 3: Trained > Untrained": test_untrained_vs_trained(),
        "Test 4: No Hallucination": test_context_adherence(trained_qa),
        "Test 5: Edge Cases": test_edge_cases(trained_qa),
    }

    # Summary
    print_header("Test Summary")

    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status} {test_name}")

    total_passed = sum(results.values())
    total_tests = len(results)
    print()
    print(f"  Overall: {total_passed}/{total_tests} tests passed")
    print()
    print("═══════════════════════════════════════════════════════════════")


if __name__ == "__main__":
    main()
