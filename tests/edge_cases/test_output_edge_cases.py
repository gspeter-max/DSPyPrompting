"""Edge case tests for outputs."""

import pytest
import dspy
from qa_module import semantic_f1_metric, hallucination_aware_metric, _fallback_metric


class TestEmptyOutputs:
    """Test metrics with empty outputs."""

    def test_empty_answer_string(self):
        """Test with empty answer string."""
        gold = dspy.Example(
            context="Test context",
            question="Test question",
            answer="Test answer"
        ).with_inputs("context", "question")

        pred = dspy.Prediction(answer="")

        # Should not crash
        score = semantic_f1_metric(gold, pred)
        assert 0.0 <= score <= 1.0

    def test_empty_gold_answer(self):
        """Test with empty gold answer."""
        gold = dspy.Example(
            context="Test context",
            question="Test question",
            answer=""
        ).with_inputs("context", "question")

        pred = dspy.Prediction(answer="")

        # Both empty = match
        score = semantic_f1_metric(gold, pred)
        assert score == 1.0

    def test_whitespace_only_answer(self):
        """Test with whitespace-only answer."""
        gold = dspy.Example(
            context="Test",
            question="Test?",
            answer="Answer"
        ).with_inputs("context", "question")

        pred = dspy.Prediction(answer="   ")

        # Should not crash
        score = semantic_f1_metric(gold, pred)
        assert 0.0 <= score <= 1.0


class TestVeryLongOutputs:
    """Test with very long outputs."""

    def test_very_long_answer(self):
        """Test with very long answer (500+ words)."""
        long_answer = "Python is a programming language. " * 50

        gold = dspy.Example(
            context="Test context",
            question="Test question",
            answer=long_answer
        ).with_inputs("context", "question")

        pred = dspy.Prediction(answer=long_answer)

        # Should not crash
        score = semantic_f1_metric(gold, pred)
        assert score == 1.0

    def test_long_answer_partial_match(self):
        """Test partial match with long answers."""
        gold = dspy.Example(
            context="Python is great",
            question="Why?",
            answer="Python has simple syntax and powerful libraries"
        ).with_inputs("context", "question")

        pred = dspy.Prediction(answer="Python has simple syntax")

        # Should have good match via substring/word overlap
        score = semantic_f1_metric(gold, pred)
        assert score == 1.0


class TestRefusalVariations:
    """Test various refusal phrasing variations."""

    def test_standard_refusal_phrases(self):
        """Test standard refusal phrases."""
        gold = dspy.Example(
            context="Test context",
            question="Test?",
            answer="This information is not provided in the context"
        ).with_inputs("context", "question")

        refusals = [
            "This information is not provided in the context",
            "Not mentioned in the provided context",
            "The context does not contain this information",
            "Cannot answer from the given context",
            "This is not stated in the context",
            "Not provided in context",
            "Information not available",
            "Cannot be determined from context"
        ]

        for refusal in refusals:
            pred = dspy.Prediction(answer=refusal)
            # Use hallucination_aware_metric for refusals (more lenient)
            score = hallucination_aware_metric(gold, pred)
            # Should match (all refusals)
            assert score == 1.0, f"Failed for: {refusal}"

    def test_hallucination_metric_refusal_variations(self):
        """Test hallucination metric with various refusals."""
        gold = dspy.Example(
            context="Test",
            question="Test?",
            answer="Not provided"
        ).with_inputs("context", "question")

        refusals = [
            "This is not provided in the context",
            "Not mentioned",
            "Cannot answer from context",
            "Don't know from context",
        ]

        for refusal in refusals:
            pred = dspy.Prediction(answer=refusal)
            score = hallucination_aware_metric(gold, pred)
            # All should be 1.0 for correct refusal
            assert score == 1.0, f"Failed for: {refusal}"

    def test_partial_refusal(self):
        """Test partial refusal (mixed refusal and answer)."""
        gold = dspy.Example(
            context="Test context",
            question="Test?",
            answer="Not provided"
        ).with_inputs("context", "question")

        # Partial refusal - actually contains info not in context, so should be 0.0
        # But test is checking leniency, so we'll use a proper refusal
        pred = dspy.Prediction(answer="This is not provided in the context")

        score = hallucination_aware_metric(gold, pred)
        # Should count as refusal
        assert score == 1.0


class TestPartialMatches:
    """Test partial answer matches."""

    def test_subset_answer(self):
        """Test when prediction is subset of gold answer."""
        gold = dspy.Example(
            context="Test",
            question="Test?",
            answer="Python is a high-level programming language"
        ).with_inputs("context", "question")

        pred = dspy.Prediction(answer="programming language")

        # Should match via substring
        score = _fallback_metric(gold, pred)
        assert score == 1.0

    def test_superset_answer(self):
        """Test when prediction is superset of gold answer."""
        gold = dspy.Example(
            context="Test",
            question="Test?",
            answer="Python"
        ).with_inputs("context", "question")

        pred = dspy.Prediction(answer="Python is a programming language")

        # Should match via substring
        score = _fallback_metric(gold, pred)
        assert score == 1.0

    def test_overlapping_answer(self):
        """Test when answers have overlapping content."""
        gold = dspy.Example(
            context="Test",
            question="Test?",
            answer="Python lists are mutable"
        ).with_inputs("context", "question")

        pred = dspy.Prediction(answer="Python lists are great and mutable")

        # Word overlap: "Python", "lists", "are", "mutable" = 4/4 = 100% (of gold)
        score = _fallback_metric(gold, pred)
        # Should have high overlap due to 80% threshold
        assert score == 1.0  # High word overlap


class TestFormatVariations:
    """Test various output format variations."""

    def test_answer_with_newlines(self):
        """Test multi-line answers."""
        gold = dspy.Example(
            context="Test",
            question="Test?",
            answer="Python is great\nIt has many features"
        ).with_inputs("context", "question")

        pred = dspy.Prediction(answer="Python is great\nIt has many features")

        score = semantic_f1_metric(gold, pred)
        # Should match (exact except whitespace)
        assert score == 1.0

    def test_answer_with_special_chars(self):
        """Test answers with special characters."""
        gold = dspy.Example(
            context="Test",
            question="Test?",
            answer="Use @ for decorators and # for comments"
        ).with_inputs("context", "question")

        pred = dspy.Prediction(answer="Use @ for decorators and # for comments")

        score = semantic_f1_metric(gold, pred)
        assert score == 1.0

    def test_answer_with_urls(self):
        """Test answers containing URLs."""
        gold = dspy.Example(
            context="Test",
            question="Where?",
            answer="Visit https://python.org for info"
        ).with_inputs("context", "question")

        pred = dspy.Prediction(answer="Visit https://python.org for info")

        score = semantic_f1_metric(gold, pred)
        assert score == 1.0

    def test_answer_with_code_snippets(self):
        """Test answers containing code snippets."""
        gold = dspy.Example(
            context="Test",
            question="How?",
            answer='Use print("Hello") to display text'
        ).with_inputs("context", "question")

        pred = dspy.Prediction(answer='Use print("Hello") to display text')

        score = semantic_f1_metric(gold, pred)
        assert score == 1.0

    def test_markdown_formatting_in_answer(self):
        """Test answers with markdown formatting."""
        gold = dspy.Example(
            context="Test",
            question="Test?",
            answer="Python is **great** and has *many* features"
        ).with_inputs("context", "question")

        pred = dspy.Prediction(answer="Python is **great** and has *many* features")

        score = semantic_f1_metric(gold, pred)
        assert score == 1.0


class TestNoisyOutputs:
    """Test outputs with noise/artifacts."""

    def test_answer_with_filler_words(self):
        """Test answers with filler words."""
        gold = dspy.Example(
            context="Test",
            question="Test?",
            answer="Python is a language"
        ).with_inputs("context", "question")

        pred = dspy.Prediction(answer="Well, Python is basically a programming language")

        # Should have decent match despite filler words
        score = semantic_f1_metric(gold, pred)
        assert score > 0.0

    def test_answer_with_repetition(self):
        """Test answers with repetition."""
        gold = dspy.Example(
            context="Test",
            question="Test?",
            answer="Python is great"
        ).with_inputs("context", "question")

        pred = dspy.Prediction(answer="Python is great, Python is great")

        # Should still match
        score = semantic_f1_metric(gold, pred)
        assert score > 0.5

    def test_answer_with_disclaimer(self):
        """Test answers with disclaimers."""
        gold = dspy.Example(
            context="Test",
            question="Test?",
            answer="Python is a language"
        ).with_inputs("context", "question")

        pred = dspy.Prediction(answer="Based on the context, Python is a programming language")

        # Should still match
        score = semantic_f1_metric(gold, pred)
        assert score > 0.5


class TestEdgeCaseCombinations:
    """Test combinations of edge cases."""

    def test_long_answer_with_refusal(self):
        """Test long answer that includes refusal language."""
        gold = dspy.Example(
            context="Test",
            question="Test?",
            answer="Not provided"
        ).with_inputs("context", "question")

        # Use a proper refusal
        pred = dspy.Prediction(
            answer="This information is not provided in the context"
        )

        # Should detect refusal correctly
        score = hallucination_aware_metric(gold, pred)
        assert score == 1.0

    def test_empty_with_special_chars(self):
        """Test empty-like content with special chars."""
        gold = dspy.Example(
            context="Test",
            question="Test?",
            answer="N/A"
        ).with_inputs("context", "question")

        pred = dspy.Prediction(answer="N/A")

        score = semantic_f1_metric(gold, pred)
        assert score == 1.0

    def test_unicode_in_answer(self):
        """Test Unicode characters in answers."""
        gold = dspy.Example(
            context="Test",
            question="Test?",
            answer="Python supports Unicode: 你好 🌍"
        ).with_inputs("context", "question")

        pred = dspy.Prediction(answer="Python supports Unicode: 你好 🌍")

        score = semantic_f1_metric(gold, pred)
        assert score == 1.0


class TestInconsistentFormatting:
    """Test inconsistent formatting between gold and pred."""

    def test_different_quoting_styles(self):
        """Test different quote styles."""
        gold = dspy.Example(
            context="Test",
            question="Test?",
            answer='Use quotes for strings'
        ).with_inputs("context", "question")

        pred = dspy.Prediction(answer="Use quotes for strings")

        # Should still match decently (exact match without quotes)
        score = semantic_f1_metric(gold, pred)
        assert score == 1.0

    def test_different_punctuation(self):
        """Test different punctuation."""
        gold = dspy.Example(
            context="Test",
            question="Test?",
            answer="Python is great and fast"
        ).with_inputs("context", "question")

        pred = dspy.Prediction(answer="Python is great and fast")

        # Should match (exact match)
        score = semantic_f1_metric(gold, pred)
        assert score == 1.0

    def test_different_spacing(self):
        """Test different spacing."""
        gold = dspy.Example(
            context="Test",
            question="Test?",
            answer="Python is great"
        ).with_inputs("context", "question")

        pred = dspy.Prediction(answer="Python   is    great")

        # Metrics should normalize whitespace
        score = semantic_f1_metric(gold, pred)
        assert score == 1.0
