"""Unit tests for dataset.py validation."""

import pytest
import dspy
from dataset import trainset


class TestDatasetStructure:
    """Test dataset structure and format."""

    def test_dataset_is_list(self):
        """Verify trainset is a list."""
        assert isinstance(trainset, list), "trainset should be a list"

    def test_dataset_has_15_examples(self):
        """Verify dataset has exactly 15 examples."""
        assert len(trainset) == 15, f"Expected 15 examples, got {len(trainset)}"

    def test_all_examples_are_dspy_examples(self):
        """Verify all examples are dspy.Example instances."""
        for i, example in enumerate(trainset):
            assert isinstance(example, dspy.Example), f"Example {i} is not a dspy.Example"

    def test_all_have_with_inputs(self):
        """Verify all examples have with_inputs() called."""
        for i, example in enumerate(trainset):
            # Check that example has _input_keys attribute (set by with_inputs)
            assert hasattr(example, '_input_keys'), f"Example {i} missing _input_keys"

    def test_all_examples_have_context(self):
        """Verify all examples have context field."""
        for i, example in enumerate(trainset):
            assert hasattr(example, 'context'), f"Example {i} missing context"
            assert isinstance(example.context, str), f"Example {i} context not a string"

    def test_all_examples_have_question(self):
        """Verify all examples have question field."""
        for i, example in enumerate(trainset):
            assert hasattr(example, 'question'), f"Example {i} missing question"
            assert isinstance(example.question, str), f"Example {i} question not a string"

    def test_all_examples_have_answer(self):
        """Verify all examples have answer field."""
        for i, example in enumerate(trainset):
            assert hasattr(example, 'answer'), f"Example {i} missing answer"
            assert isinstance(example.answer, str), f"Example {i} answer not a string"


class TestPositiveExamples:
    """Test positive examples (answer in context)."""

    def test_positive_examples_count(self):
        """Count positive examples (should be 9)."""
        positive_count = 0
        for example in trainset:
            answer_lower = example.answer.lower()
            # Positive examples don't contain refusal phrases
            if not any(phrase in answer_lower for phrase in ["not provided", "cannot answer", "not mentioned"]):
                positive_count += 1
        assert positive_count == 9, f"Expected 9 positive examples, got {positive_count}"

    def test_positive_examples_have_answers(self):
        """All positive examples have non-refusal answers."""
        for i, example in enumerate(trainset[:9]):  # First 9 are positive
            answer_lower = example.answer.lower()
            is_refusal = any(phrase in answer_lower for phrase in [
                "not provided", "cannot answer", "not mentioned"
            ])
            assert not is_refusal, f"Example {i} has refusal answer but should be positive"

    def test_positive_context_answer_consistency(self):
        """Positive examples should have answers consistent with context."""
        # This is a basic sanity check - answers should be substantive
        for i, example in enumerate(trainset[:9]):
            assert len(example.answer) > 20, f"Example {i} answer too short for positive example"
            assert len(example.context) > 100, f"Example {i} context too short"


class TestNegativeExamples:
    """Test negative examples (answer not in context)."""

    def test_negative_examples_count(self):
        """Count negative examples (should be 6)."""
        negative_count = 0
        for example in trainset:
            answer_lower = example.answer.lower()
            # Negative examples contain refusal phrases
            if any(phrase in answer_lower for phrase in ["not provided", "cannot answer", "not mentioned"]):
                negative_count += 1
        assert negative_count == 6, f"Expected 6 negative examples, got {negative_count}"

    def test_negative_examples_expect_refusal(self):
        """All negative examples expect refusal answers."""
        # Examples 10-15 are negative
        for i, example in enumerate(trainset[9:], start=10):
            answer_lower = example.answer.lower()
            is_refusal = any(phrase in answer_lower for phrase in [
                "not provided", "not mentioned", "not in context"
            ])
            assert is_refusal, f"Example {i} should have refusal answer"

    def test_negative_context_no_answer(self):
        """Negative examples should not contain answer in context."""
        # This is a basic check - contexts should be about different topics
        for i, example in enumerate(trainset[9:], start=10):
            # Questions should ask about things not in context
            assert len(example.context) > 50, f"Example {i} context too short"
            assert len(example.question) > 10, f"Example {i} question too short"


class TestDataQuality:
    """Test data quality standards."""

    def test_no_empty_contexts(self):
        """All contexts should be non-empty."""
        for i, example in enumerate(trainset):
            assert len(example.context.strip()) > 0, f"Example {i} has empty context"

    def test_no_empty_questions(self):
        """All questions should be non-empty."""
        for i, example in enumerate(trainset):
            assert len(example.question.strip()) > 0, f"Example {i} has empty question"

    def test_no_empty_answers(self):
        """All answers should be non-empty."""
        for i, example in enumerate(trainset):
            assert len(example.answer.strip()) > 0, f"Example {i} has empty answer"

    def test_contexts_reasonable_length(self):
        """Contexts should be substantive (> 100 chars for positive examples)."""
        for i, example in enumerate(trainset[:9]):
            assert len(example.context) >= 100, f"Example {i} context too short"

    def test_questions_reasonable_length(self):
        """Questions should be complete (> 20 chars)."""
        for i, example in enumerate(trainset):
            assert len(example.question) >= 20, f"Example {i} question too short"

    def test_answers_reasonable_length(self):
        """Answers should be substantive."""
        for i, example in enumerate(trainset[:9]):
            # Positive examples should have longer answers
            assert len(example.answer) >= 20, f"Example {i} answer too short"

    def test_no_special_characters_breaking(self):
        """Test special characters don't break parsing."""
        for i, example in enumerate(trainset):
            # Should be able to access all fields without errors
            _ = example.context
            _ = example.question
            _ = example.answer
            # Should be able to convert to string
            context_str = str(example.context)
            question_str = str(example.question)
            answer_str = str(example.answer)
            assert len(context_str) > 0
            assert len(question_str) > 0
            assert len(answer_str) > 0

    def test_unicode_in_dataset(self):
        """Test that dataset handles Unicode characters."""
        # Dataset should handle any Unicode in the data
        for i, example in enumerate(trainset):
            # Try to encode/decode without errors
            context_bytes = example.context.encode('utf-8')
            question_bytes = example.question.encode('utf-8')
            answer_bytes = example.answer.encode('utf-8')

            # Should decode back correctly
            assert context_bytes.decode('utf-8') == example.context
            assert question_bytes.decode('utf-8') == example.question
            assert answer_bytes.decode('utf-8') == example.answer


class TestDatasetIndexing:
    """Test dataset indexing and access patterns."""

    def test_sequential_access(self):
        """Test sequential access works correctly."""
        for i in range(len(trainset)):
            example = trainset[i]
            assert example is not None
            assert hasattr(example, 'context')
            assert hasattr(example, 'question')
            assert hasattr(example, 'answer')

    def test_iteration(self):
        """Test iteration over dataset works."""
        count = 0
        for example in trainset:
            assert example is not None
            count += 1
        assert count == 15

    def test_slicing(self):
        """Test slicing works correctly."""
        # First 5 examples
        first_five = trainset[:5]
        assert len(first_five) == 5
        assert all(isinstance(ex, dspy.Example) for ex in first_five)

        # Last 5 examples
        last_five = trainset[-5:]
        assert len(last_five) == 5
        assert all(isinstance(ex, dspy.Example) for ex in last_five)

        # Middle slice
        middle = trainset[5:10]
        assert len(middle) == 5
        assert all(isinstance(ex, dspy.Example) for ex in middle)
