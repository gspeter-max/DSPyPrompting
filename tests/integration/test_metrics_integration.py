"""Integration tests for metrics with real model outputs."""

import pytest
import dspy
from qa_module import semantic_f1_metric, hallucination_aware_metric, _fallback_metric


class TestSemanticF1Integration:
    """Test SemanticF1 metric with realistic outputs."""

    def test_exact_match_with_real_output(self):
        """Test exact match scenario."""
        gold = dspy.Example(
            context="Python lists are mutable sequences.",
            question="Are lists mutable?",
            answer="Yes, they are mutable"
        ).with_inputs("context", "question")

        pred_exact = dspy.Prediction(answer="Yes, they are mutable")

        score = semantic_f1_metric(gold, pred_exact)
        assert score == 1.0

    def test_semantic_match_with_real_output(self):
        """Test semantic match with word overlap."""
        gold = dspy.Example(
            context="Python lists are mutable sequences.",
            question="Are lists mutable?",
            answer="Yes, they are mutable"
        ).with_inputs("context", "question")

        # Similar but not exact
        pred_semantic = dspy.Prediction(answer="Yes, Python lists are mutable")

        score = semantic_f1_metric(gold, pred_semantic)
        # Should be high due to word overlap
        assert score >= 0.5

    def test_no_match_with_real_output(self):
        """Test no match scenario."""
        gold = dspy.Example(
            context="Python lists are mutable.",
            question="Are lists mutable?",
            answer="Yes, they are mutable"
        ).with_inputs("context", "question")

        pred_wrong = dspy.Prediction(answer="No, they are immutable")

        score = semantic_f1_metric(gold, pred_wrong)
        # Should be low
        assert score < 0.5

    def test_short_answer_uses_fallback(self):
        """Test short answers use fallback metric."""
        gold = dspy.Example(
            context="Test",
            question="Test?",
            answer="Yes"
        ).with_inputs("context", "question")

        pred = dspy.Prediction(answer="No")

        # Short answers use fallback
        score = semantic_f1_metric(gold, pred)
        assert 0.0 <= score <= 1.0

    def test_various_answer_formats(self):
        """Test metric handles various answer formats."""
        gold = dspy.Example(
            context="Python is a programming language.",
            question="What is Python?",
            answer="Python is a high-level programming language"
        ).with_inputs("context", "question")

        # Various formats
        formats = [
            "Python is a high-level programming language",  # Exact
            "Python is a programming language",  # Subset
            "a high-level programming language",  # Partial
        ]

        for fmt in formats:
            pred = dspy.Prediction(answer=fmt)
            score = semantic_f1_metric(gold, pred)
            assert 0.0 <= score <= 1.0


class TestHallucinationDetectionIntegration:
    """Test hallucination detection with real outputs."""

    def test_positive_example_normal_answer(self):
        """Test positive example with normal answer."""
        gold = dspy.Example(
            context="Python lists are mutable.",
            question="Are lists mutable?",
            answer="Yes, they are mutable"
        ).with_inputs("context", "question")

        pred = dspy.Prediction(answer="Yes, Python lists are mutable")

        score = hallucination_aware_metric(gold, pred)
        # Should use semantic_f1 and return good score
        assert score >= 0.5

    def test_negative_example_correct_refusal(self):
        """Test negative example with correct refusal."""
        gold = dspy.Example(
            context="Python lists exist.",
            question="What is a tuple?",
            answer="This information is not provided in the context"
        ).with_inputs("context", "question")

        pred = dspy.Prediction(answer="This information is not provided in the context")

        score = hallucination_aware_metric(gold, pred)
        # Correct refusal = 1.0
        assert score == 1.0

    def test_negative_example_hallucination(self):
        """Test negative example with hallucination."""
        gold = dspy.Example(
            context="Python lists exist.",
            question="What is a tuple?",
            answer="This information is not provided in the context"
        ).with_inputs("context", "question")

        pred = dspy.Prediction(answer="A tuple is an immutable sequence")

        score = hallucination_aware_metric(gold, pred)
        # Hallucination = 0.0
        assert score == 0.0

    def test_refusal_phrase_variations(self):
        """Test various refusal phrases all detected."""
        gold = dspy.Example(
            context="Test context",
            question="Test?",
            answer="Not provided"
        ).with_inputs("context", "question")

        refusals = [
            "This information is not provided in the context",
            "Not mentioned in the provided context",
            "Cannot answer from the given context",
            "This is not stated in the context",
            "Not in context"
        ]

        for refusal in refusals:
            pred = dspy.Prediction(answer=refusal)
            score = hallucination_aware_metric(gold, pred)
            assert score == 1.0, f"Failed for: {refusal}"

    def test_mixed_positive_negative(self):
        """Test mix of positive and negative examples."""
        examples = [
            # Positive
            (dspy.Example(
                context="Lists are mutable.",
                question="Mutable?",
                answer="Yes"
            ).with_inputs("context", "question"),
             dspy.Prediction(answer="Yes")),
            # Negative
            (dspy.Example(
                context="Lists exist.",
                question="What are tuples?",
                answer="Not provided"
            ).with_inputs("context", "question"),
             dspy.Prediction(answer="Not in context")),
        ]

        for gold, pred in examples:
            score = hallucination_aware_metric(gold, pred)
            assert 0.0 <= score <= 1.0


class TestFallbackMetricIntegration:
    """Test fallback metric with various scenarios."""

    def test_exact_match_fallback(self):
        """Test fallback with exact match."""
        gold = dspy.Example(
            context="Test",
            question="Test?",
            answer="Answer"
        ).with_inputs("context", "question")

        pred = dspy.Prediction(answer="Answer")

        score = _fallback_metric(gold, pred)
        assert score == 1.0

    def test_refusal_matching_fallback(self):
        """Test refusal phrase matching in fallback."""
        gold = dspy.Example(
            context="Test",
            question="Test?",
            answer="Not provided in context"
        ).with_inputs("context", "question")

        refusals = [
            "Not provided",
            "Not mentioned",
            "Cannot answer",
            "Don't know",
        ]

        for refusal in refusals:
            pred = dspy.Prediction(answer=refusal)
            score = _fallback_metric(gold, pred)
            assert score == 1.0, f"Failed for: {refusal}"

    def test_substring_matching_fallback(self):
        """Test substring matching in fallback."""
        gold = dspy.Example(
            context="Test",
            question="Test?",
            answer="@"
        ).with_inputs("context", "question")

        pred = dspy.Prediction(answer="the @ symbol")

        score = _fallback_metric(gold, pred)
        # Should match via substring
        assert score == 1.0

    def test_word_overlap_threshold_fallback(self):
        """Test 80% word overlap threshold in fallback."""
        gold = dspy.Example(
            context="Test",
            question="Test?",
            answer="Python lists are mutable sequences"
        ).with_inputs("context", "question")

        # 4/5 words = 80%
        pred = dspy.Prediction(answer="Python lists are mutable")

        score = _fallback_metric(gold, pred)
        assert score == 1.0

    def test_below_threshold_fallback(self):
        """Test below 80% overlap returns 0.0."""
        gold = dspy.Example(
            context="Test",
            question="Test?",
            answer="Python is a great language"
        ).with_inputs("context", "question")

        pred = dspy.Prediction(answer="Java is nice")

        # Only 1/5 overlap = 20%
        score = _fallback_metric(gold, pred)
        assert score == 0.0


class TestMetricConsistency:
    """Test consistency between metrics."""

    def test_semantic_f1_consistent_results(self):
        """Test semantic_f1 returns consistent scores for same input."""
        gold = dspy.Example(
            context="Test",
            question="Test?",
            answer="Test answer"
        ).with_inputs("context", "question")

        pred = dspy.Prediction(answer="Test answer")

        # Run multiple times
        scores = [semantic_f1_metric(gold, pred) for _ in range(3)]

        # All should be identical
        assert all(s == scores[0] for s in scores)
        assert scores[0] == 1.0

    def test_hallucination_metric_consistent_results(self):
        """Test hallucination_aware_metric returns consistent scores."""
        gold = dspy.Example(
            context="Test",
            question="Test?",
            answer="Not provided"
        ).with_inputs("context", "question")

        pred = dspy.Prediction(answer="Not in context")

        # Run multiple times
        scores = [hallucination_aware_metric(gold, pred) for _ in range(3)]

        # All should be identical
        assert all(s == scores[0] for s in scores)
        assert scores[0] == 1.0


class TestRealDatasetIntegration:
    """Test metrics with actual dataset examples."""

    def test_metrics_with_dataset_positive_examples(self):
        """Test metrics work with positive examples from dataset."""
        from dataset import trainset

        # Test first 5 (positive examples)
        for i in range(5):
            gold = trainset[i]
            pred = dspy.Prediction(answer=gold.answer)

            # Test both metrics
            score1 = semantic_f1_metric(gold, pred)
            score2 = hallucination_aware_metric(gold, pred)

            assert 0.0 <= score1 <= 1.0
            assert 0.0 <= score2 <= 1.0

    def test_metrics_with_dataset_negative_examples(self):
        """Test metrics work with negative examples from dataset."""
        from dataset import trainset

        # Test last 5 (negative examples)
        for i in range(10, 15):
            gold = trainset[i]
            pred = dspy.Prediction(answer=gold.answer)

            # Test hallucination metric
            score = hallucination_aware_metric(gold, pred)

            # Should be 1.0 for correct refusal
            assert score == 1.0

    def test_partial_match_scenarios(self):
        """Test partial match scenarios with dataset-like data."""
        gold = dspy.Example(
            context="Python lists are mutable sequences that can hold mixed types.",
            question="What are Python lists?",
            answer="Python lists are mutable sequences that can hold mixed types"
        ).with_inputs("context", "question")

        partial_answers = [
            "mutable sequences",  # Partial
            "Python lists are mutable",  # Subset
            "sequences that can hold mixed types",  # Partial
        ]

        for answer in partial_answers:
            pred = dspy.Prediction(answer=answer)
            score = semantic_f1_metric(gold, pred)
            # All should have some match
            assert score > 0.0
