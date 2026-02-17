"""Custom assertion helpers for tests.

These functions provide more readable and specific assertions
for common testing scenarios in the DSPy QA system.
"""


def assert_refusal(pred_answer):
    """Assert answer indicates information not in context.

    Args:
        pred_answer: The predicted answer text

    Raises:
        AssertionError: If answer doesn't indicate refusal
    """
    refusal_phrases = [
        "not provided in context",
        "not mentioned",
        "cannot answer",
        "not in the context",
        "don't know",
        "information is not"
    ]

    pred_lower = pred_answer.lower()

    has_refusal = any(phrase in pred_lower for phrase in refusal_phrases)

    assert has_refusal, (
        f"Expected refusal answer (containing phrases like 'not provided in context'), "
        f"but got: '{pred_answer}'"
    )


def assert_no_refusal(pred_answer):
    """Assert answer does NOT indicate information not in context.

    Args:
        pred_answer: The predicted answer text

    Raises:
        AssertionError: If answer indicates refusal
    """
    refusal_phrases = [
        "not provided in context",
        "not mentioned",
        "cannot answer",
        "not in the context",
        "don't know"
    ]

    pred_lower = pred_answer.lower()

    has_refusal = any(phrase in pred_lower for phrase in refusal_phrases)

    assert not has_refusal, (
        f"Expected actual answer, but got refusal: '{pred_answer}'"
    )


def assert_semantic_match(score, threshold=0.5):
    """Assert semantic score meets threshold.

    Args:
        score: The semantic similarity score
        threshold: Minimum acceptable score (default: 0.5)

    Raises:
        AssertionError: If score is below threshold
    """
    assert score >= threshold, (
        f"Semantic score {score:.3f} is below threshold {threshold:.3f}"
    )


def assert_exact_match(score):
    """Assert score indicates exact match.

    Args:
        score: The similarity score

    Raises:
        AssertionError: If score is not 1.0
    """
    assert score == 1.0, (
        f"Expected exact match (score=1.0), but got {score:.3f}"
    )


def assert_no_match(score):
    """Assert score indicates no match.

    Args:
        score: The similarity score

    Raises:
        AssertionError: If score is not 0.0 or very close
    """
    assert score < 0.3, (
        f"Expected no match (score < 0.3), but got {score:.3f}"
    )


def assert_valid_prediction(pred):
    """Assert prediction has required fields and valid values.

    Args:
        pred: dspy.Prediction object

    Raises:
        AssertionError: If prediction is invalid
    """
    assert hasattr(pred, 'answer'), "Prediction missing 'answer' field"
    assert isinstance(pred.answer, str), "Answer must be a string"

    # Answer should not be None (empty string is ok)
    assert pred.answer is not None, "Answer cannot be None"


def assert_valid_example(example):
    """Assert example has required fields.

    Args:
        example: dspy.Example object

    Raises:
        AssertionError: If example is invalid
    """
    assert hasattr(example, 'context'), "Example missing 'context' field"
    assert hasattr(example, 'question'), "Example missing 'question' field"
    assert hasattr(example, 'answer'), "Example missing 'answer' field"

    assert isinstance(example.context, str), "Context must be a string"
    assert isinstance(example.question, str), "Question must be a string"
    assert isinstance(example.answer, str), "Answer must be a string"


def assert_positive_example(example):
    """Assert example is a positive example (answer in context).

    Args:
        example: dspy.Example object

    Raises:
        AssertionError: If example is not positive
    """
    assert_no_refusal(example.answer)

    # Positive examples should have substantive answers
    assert len(example.answer) > 10, (
        f"Positive example answer too short: '{example.answer}'"
    )


def assert_negative_example(example):
    """Assert example is a negative example (answer not in context).

    Args:
        example: dspy.Example object

    Raises:
        AssertionError: If example is not negative
    """
    assert_refusal(example.answer)


def assert_score_range(score):
    """Assert score is in valid range [0.0, 1.0].

    Args:
        score: The score to validate

    Raises:
        AssertionError: If score is out of range
    """
    assert 0.0 <= score <= 1.0, (
        f"Score {score:.3f} is out of valid range [0.0, 1.0]"
    )


def assert_high_quality_match(score, min_score=0.8):
    """Assert score indicates high quality match.

    Args:
        score: The similarity score
        min_score: Minimum score for "high quality" (default: 0.8)

    Raises:
        AssertionError: If score is below high quality threshold
    """
    assert score >= min_score, (
        f"Score {score:.3f} is below high quality threshold {min_score:.3f}"
    )


def assert_improvement(old_score, new_score, min_improvement=0.1):
    """Assert new score shows improvement over old score.

    Args:
        old_score: The original score
        new_score: The improved score
        min_improvement: Minimum improvement required (default: 0.1)

    Raises:
        AssertionError: If improvement is insufficient
    """
    improvement = new_score - old_score

    assert improvement >= min_improvement, (
        f"Score improved by {improvement:.3f}, "
        f"but needed at least {min_improvement:.3f} "
        f"(old: {old_score:.3f} -> new: {new_score:.3f})"
    )


def assert_demos_count(qa_module, expected_count):
    """Assert QAModule has expected number of demonstrations.

    Args:
        qa_module: QAModule instance
        expected_count: Expected number of demos

    Raises:
        AssertionError: If demo count doesn't match
    """
    actual_count = len(qa_module.generate_answer.predict.demos)

    assert actual_count == expected_count, (
        f"Expected {expected_count} demonstrations, but found {actual_count}"
    )


def assert_demos_structure(demos):
    """Assert demonstrations have correct structure.

    Args:
        demos: List of demonstration dictionaries

    Raises:
        AssertionError: If structure is invalid
    """
    assert isinstance(demos, list), "Demos must be a list"

    for i, demo in enumerate(demos):
        assert isinstance(demo, dict), f"Demo {i} is not a dict"
        assert 'context' in demo, f"Demo {i} missing 'context' field"
        assert 'question' in demo, f"Demo {i} missing 'question' field"
        assert 'answer' in demo, f"Demo {i} missing 'answer' field"


def assert_dataset_size(dataset, expected_size):
    """Assert dataset has expected size.

    Args:
        dataset: Dataset (list of examples)
        expected_size: Expected number of examples

    Raises:
        AssertionError: If size doesn't match
    """
    actual_size = len(dataset)

    assert actual_size == expected_size, (
        f"Expected dataset size {expected_size}, but got {actual_size}"
    )


def assert_training_improvement(untrained_scores, trained_scores):
    """Assert trained model performs better than untrained.

    Args:
        untrained_scores: List of scores from untrained model
        trained_scores: List of scores from trained model

    Raises:
        AssertionError: If no improvement detected
    """
    assert len(untrained_scores) == len(trained_scores), (
        "Score lists must be same length"
    )

    untrained_avg = sum(untrained_scores) / len(untrained_scores)
    trained_avg = sum(trained_scores) / len(trained_scores)

    assert trained_avg > untrained_avg, (
        f"Trained model average ({trained_avg:.3f}) is not better than "
        f"untrained average ({untrained_avg:.3f})"
    )


def assert_context_adherence(gold, pred):
    """Assert prediction adheres to context (no hallucination for negative examples).

    Args:
        gold: Ground truth example
        pred: Prediction

    Raises:
        AssertionError: If hallucination detected
    """
    gold_lower = gold.answer.lower()

    # Check if this is a negative example
    is_negative = (
        "not provided" in gold_lower or
        "not mentioned" in gold_lower or
        "cannot answer" in gold_lower
    )

    if is_negative:
        # For negative examples, prediction should refuse
        assert_refusal(pred.answer)
    else:
        # For positive examples, prediction should provide answer
        assert_no_refusal(pred.answer)
