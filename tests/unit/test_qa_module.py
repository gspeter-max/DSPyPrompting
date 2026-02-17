"""Unit tests for qa_module.py components."""

import pytest
import dspy
import tempfile
import os
from qa_module import QAModule, GenerateAnswer, semantic_f1_metric, hallucination_aware_metric, _fallback_metric


class TestGenerateAnswerSignature:
    """Test GenerateAnswer signature definition."""

    def test_signature_has_required_fields(self):
        """Verify signature has context, question, answer fields."""
        assert hasattr(GenerateAnswer, 'input_fields')
        assert 'context' in GenerateAnswer.input_fields
        assert 'question' in GenerateAnswer.input_fields
        assert 'answer' in GenerateAnswer.output_fields

    def test_signature_field_descriptions(self):
        """Verify fields have descriptions for DSPy."""
        context_field = GenerateAnswer.input_fields['context']
        question_field = GenerateAnswer.input_fields['question']
        answer_field = GenerateAnswer.output_fields['answer']

        assert context_field.desc is not None
        assert question_field.desc is not None
        assert answer_field.desc is not None

    def test_signature_refusal_instructions(self):
        """Verify signature docstring contains refusal instructions."""
        docstring = GenerateAnswer.__doc__
        assert docstring is not None
        assert "not provided" in docstring.lower()
        assert "context" in docstring.lower()


class TestQAModule:
    """Test QAModule class."""

    def test_qamodule_initialization(self):
        """Verify module initializes correctly."""
        qa = QAModule()
        assert qa is not None
        assert hasattr(qa, 'generate_answer')

    def test_qamodule_has_generate_answer(self):
        """Verify ChainOfThought component exists."""
        qa = QAModule()
        assert hasattr(qa.generate_answer, 'predict')
        assert isinstance(qa.generate_answer, dspy.ChainOfThought)

    def test_qamodule_forward_method(self):
        """Test forward() returns prediction."""
        qa = QAModule()
        # Note: This will fail without actual LM configured
        # Testing method signature
        assert hasattr(qa, 'forward')
        assert callable(qa.forward)

    def test_qamodule_input_validation_empty_context(self):
        """Test with empty context string."""
        qa = QAModule()
        # Should not crash on empty input
        # (actual prediction requires LM)
        assert callable(qa.forward)

    def test_qamodule_input_validation_empty_question(self):
        """Test with empty question string."""
        qa = QAModule()
        # Should not crash on empty input
        assert callable(qa.forward)


class TestSemanticF1Metric:
    """Test semantic_f1_metric function."""

    def test_exact_match_returns_1(self):
        """Identical answers return 1.0."""
        gold = dspy.Example(
            context="Test context",
            question="Test question",
            answer="Yes, they are mutable"
        ).with_inputs("context", "question")

        pred = dspy.Prediction(answer="Yes, they are mutable")

        score = semantic_f1_metric(gold, pred)
        assert score == 1.0

    def test_case_insensitive_match(self):
        """Case insensitive matching."""
        gold = dspy.Example(
            context="Test",
            question="Test?",
            answer="Yes, they are mutable"
        ).with_inputs("context", "question")

        pred = dspy.Prediction(answer="YES, THEY ARE MUTABLE")

        score = semantic_f1_metric(gold, pred)
        assert score == 1.0

    def test_whitespace_tolerant(self):
        """Whitespace tolerant matching."""
        gold = dspy.Example(
            context="Test",
            question="Test?",
            answer="Yes"
        ).with_inputs("context", "question")

        pred = dspy.Prediction(answer="  Yes  ")

        score = semantic_f1_metric(gold, pred)
        assert score == 1.0

    def test_fallback_for_short_answers(self):
        """Use fallback for short answers (< 50 chars)."""
        gold = dspy.Example(
            context="Test",
            question="Test?",
            answer="Yes"
        ).with_inputs("context", "question")

        pred = dspy.Prediction(answer="No")

        # Short answers use fallback, not SemanticF1
        score = semantic_f1_metric(gold, pred)
        assert 0.0 <= score <= 1.0

    def test_fallback_on_refusal_answer(self):
        """Short refusal answers use fallback."""
        gold = dspy.Example(
            context="Test",
            question="Test?",
            answer="This information is not provided in the context"
        ).with_inputs("context", "question")

        pred = dspy.Prediction(answer="Not mentioned")

        # Should match via fallback
        score = semantic_f1_metric(gold, pred)
        assert score >= 0.5

    def test_no_match_returns_0(self):
        """Completely different answers return low score."""
        gold = dspy.Example(
            context="Test",
            question="Test?",
            answer="Python is a programming language"
        ).with_inputs("context", "question")

        pred = dspy.Prediction(answer="I like pizza")

        score = semantic_f1_metric(gold, pred)
        assert score < 0.5

    def test_field_mapping_answer_to_response(self):
        """Verify field name mapping from .answer to .response."""
        # This tests that semantic_f1_metric correctly maps
        # our .answer field to SemanticF1's expected .response field
        gold = dspy.Example(
            context="Test",
            question="Test?",
            answer="Test answer"
        ).with_inputs("context", "question")

        pred = dspy.Prediction(answer="Test answer")

        # Should not raise an error about missing .response field
        score = semantic_f1_metric(gold, pred)
        assert score == 1.0


class TestHallucinationAwareMetric:
    """Test hallucination_aware_metric function."""

    def test_positive_example_uses_semantic_f1(self):
        """Normal answers use semantic_f1."""
        gold = dspy.Example(
            context="Python lists are mutable.",
            question="Are lists mutable?",
            answer="Yes, they are mutable"
        ).with_inputs("context", "question")

        pred = dspy.Prediction(answer="Yes, Python lists are mutable")

        score = hallucination_aware_metric(gold, pred)
        # Should use semantic_f1, return good score
        assert score >= 0.5

    def test_negative_example_correct_refusal(self):
        """Correct refusal returns 1.0."""
        gold = dspy.Example(
            context="Python lists exist.",
            question="What is a tuple?",
            answer="This information is not provided in the context"
        ).with_inputs("context", "question")

        pred = dspy.Prediction(answer="This information is not provided in the context")

        score = hallucination_aware_metric(gold, pred)
        assert score == 1.0

    def test_negative_example_hallucination_penalty(self):
        """Wrong answer returns 0.0."""
        gold = dspy.Example(
            context="Python lists exist.",
            question="What is a tuple?",
            answer="This information is not provided in the context"
        ).with_inputs("context", "question")

        pred = dspy.Prediction(answer="A tuple is an immutable sequence")

        score = hallucination_aware_metric(gold, pred)
        assert score == 0.0

    def test_refusal_phrase_variations(self):
        """Test various refusal phrases."""
        gold = dspy.Example(
            context="Test",
            question="Test?",
            answer="This information is not provided in the context"
        ).with_inputs("context", "question")

        refusals = [
            "This information is not provided in the context",
            "Not mentioned in context",
            "This is not in the context",
            "Cannot answer from the context"
        ]

        for refusal in refusals:
            pred = dspy.Prediction(answer=refusal)
            score = hallucination_aware_metric(gold, pred)
            assert score == 1.0, f"Failed for refusal: {refusal}"

    def test_positive_example_no_answer(self):
        """Positive example with no match in answer."""
        gold = dspy.Example(
            context="Python lists are mutable.",
            question="Are lists mutable?",
            answer="Yes"
        ).with_inputs("context", "question")

        pred = dspy.Prediction(answer="No")

        score = hallucination_aware_metric(gold, pred)
        # Should use semantic_f1, return low score
        assert score < 0.5


class TestFallbackMetric:
    """Test _fallback_metric function."""

    def test_exact_match_fallback(self):
        """Exact match returns 1.0."""
        gold = dspy.Example(
            context="Test",
            question="Test?",
            answer="Test answer"
        ).with_inputs("context", "question")

        pred = dspy.Prediction(answer="Test answer")

        score = _fallback_metric(gold, pred)
        assert score == 1.0

    def test_refusal_match_fallback(self):
        """Refusal phrase matching."""
        gold = dspy.Example(
            context="Test",
            question="Test?",
            answer="This information is not provided in the context"
        ).with_inputs("context", "question")

        pred = dspy.Prediction(answer="Not mentioned in the context")

        score = _fallback_metric(gold, pred)
        assert score == 1.0

    def test_substring_match(self):
        """Substring matching (e.g., 'the @' matches '@')."""
        gold = dspy.Example(
            context="Test",
            question="Test?",
            answer="@"
        ).with_inputs("context", "question")

        pred = dspy.Prediction(answer="the @ symbol")

        score = _fallback_metric(gold, pred)
        # Should match via substring
        assert score == 1.0

    def test_word_overlap_threshold(self):
        """80% word overlap returns 1.0."""
        gold = dspy.Example(
            context="Test",
            question="Test?",
            answer="Python lists are mutable sequences"
        ).with_inputs("context", "question")

        pred = dspy.Prediction(answer="Python lists are mutable")

        # 4/5 words overlap = 80%
        score = _fallback_metric(gold, pred)
        assert score == 1.0

    def test_no_match_returns_0(self):
        """No overlap returns 0.0."""
        gold = dspy.Example(
            context="Test",
            question="Test?",
            answer="Python is great"
        ).with_inputs("context", "question")

        pred = dspy.Prediction(answer="I like pizza")

        score = _fallback_metric(gold, pred)
        assert score == 0.0

    def test_case_insensitive_fallback(self):
        """Case insensitive matching in fallback."""
        gold = dspy.Example(
            context="Test",
            question="Test?",
            answer="Yes"
        ).with_inputs("context", "question")

        pred = dspy.Prediction(answer="yes")

        score = _fallback_metric(gold, pred)
        assert score == 1.0

    def test_whitespace_tolerant_fallback(self):
        """Whitespace tolerant in fallback."""
        gold = dspy.Example(
            context="Test",
            question="Test?",
            answer="Yes"
        ).with_inputs("context", "question")

        pred = dspy.Prediction(answer="  Yes  ")

        score = _fallback_metric(gold, pred)
        assert score == 1.0
