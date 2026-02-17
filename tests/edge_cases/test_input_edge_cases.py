"""Edge case tests for inputs."""

import pytest
import dspy
from qa_module import QAModule, semantic_f1_metric, hallucination_aware_metric


class TestEmptyInputs:
    """Test model behavior with empty inputs."""

    def test_empty_context_string(self):
        """Test with empty context string."""
        gold = dspy.Example(
            context="",
            question="What is Python?",
            answer="This information is not provided in the context"
        ).with_inputs("context", "question")

        pred = dspy.Prediction(answer="This information is not provided in the context")

        # Should not crash
        score = semantic_f1_metric(gold, pred)
        assert 0.0 <= score <= 1.0

    def test_empty_question_string(self):
        """Test with empty question string."""
        gold = dspy.Example(
            context="Python is a language",
            question="",
            answer="Empty question"
        ).with_inputs("context", "question")

        pred = dspy.Prediction(answer="Empty question")

        # Should not crash
        score = semantic_f1_metric(gold, pred)
        assert score == 1.0

    def test_empty_answer(self):
        """Test metrics with empty answer."""
        gold = dspy.Example(
            context="Test",
            question="Test?",
            answer="Answer"
        ).with_inputs("context", "question")

        pred = dspy.Prediction(answer="")

        # Should not crash
        score = semantic_f1_metric(gold, pred)
        assert 0.0 <= score <= 1.0

    def test_all_empty_strings(self):
        """Test with all empty strings."""
        gold = dspy.Example(
            context="",
            question="",
            answer=""
        ).with_inputs("context", "question")

        pred = dspy.Prediction(answer="")

        # Should not crash
        score = semantic_f1_metric(gold, pred)
        assert score == 1.0  # Both empty = match


class TestVeryShortInputs:
    """Test with very short inputs."""

    def test_very_short_context_one_word(self):
        """Test with one-word context."""
        gold = dspy.Example(
            context="Python",
            question="What is Python?",
            answer="Programming language"
        ).with_inputs("context", "question")

        pred = dspy.Prediction(answer="Programming language")

        score = semantic_f1_metric(gold, pred)
        assert score == 1.0

    def test_very_short_question(self):
        """Test with one-word question."""
        gold = dspy.Example(
            context="Python is great",
            question="Python?",
            answer="Yes"
        ).with_inputs("context", "question")

        pred = dspy.Prediction(answer="Yes")

        score = semantic_f1_metric(gold, pred)
        assert score == 1.0

    def test_very_short_answer(self):
        """Test with one-word answer."""
        gold = dspy.Example(
            context="Python lists are mutable",
            question="Are lists mutable?",
            answer="Yes"
        ).with_inputs("context", "question")

        pred = dspy.Prediction(answer="Yes")

        score = semantic_f1_metric(gold, pred)
        assert score == 1.0


class TestVeryLongInputs:
    """Test with very long inputs."""

    def test_very_long_context(self):
        """Test with very long context (1000+ words)."""
        long_context = "Python is a programming language. " * 100

        gold = dspy.Example(
            context=long_context,
            question="What is Python?",
            answer="A programming language"
        ).with_inputs("context", "question")

        pred = dspy.Prediction(answer="A programming language")

        # Should not crash
        score = semantic_f1_metric(gold, pred)
        assert score == 1.0

    def test_very_long_question(self):
        """Test with very long question."""
        long_question = "What is Python? " * 50

        gold = dspy.Example(
            context="Python is a language",
            question=long_question,
            answer="A language"
        ).with_inputs("context", "question")

        pred = dspy.Prediction(answer="A language")

        score = semantic_f1_metric(gold, pred)
        assert score == 1.0

    def test_very_long_answer(self):
        """Test with very long answer."""
        long_answer = "Python is a programming language that is widely used. " * 20

        gold = dspy.Example(
            context="Python is a language",
            question="What is Python?",
            answer=long_answer
        ).with_inputs("context", "question")

        pred = dspy.Prediction(answer=long_answer)

        # Should not crash
        score = semantic_f1_metric(gold, pred)
        assert score == 1.0


class TestSpecialCharacters:
    """Test with special characters."""

    def test_unicode_characters(self):
        """Test with Unicode characters."""
        unicode_context = "Python支持Unicode: 你好世界 🌍"

        gold = dspy.Example(
            context=unicode_context,
            question="Does Python support Unicode?",
            answer="Yes, Python supports Unicode"
        ).with_inputs("context", "question")

        pred = dspy.Prediction(answer="Yes, Python supports Unicode")

        # Should not crash
        score = semantic_f1_metric(gold, pred)
        assert 0.0 <= score <= 1.0

    def test_newlines_and_tabs(self):
        """Test with newlines and tabs."""
        context_with_whitespace = "Python is great.\n\tIt has many features.\n\tPeople love it."

        gold = dspy.Example(
            context=context_with_whitespace,
            question="What is Python?",
            answer="A programming language"
        ).with_inputs("context", "question")

        pred = dspy.Prediction(answer="A programming language")

        score = semantic_f1_metric(gold, pred)
        assert score == 1.0

    def test_quotes_and_escaping(self):
        """Test with quotes and special characters."""
        context_with_quotes = 'Python uses "quotes" and \'apostrophes\' for strings.'

        gold = dspy.Example(
            context=context_with_quotes,
            question="What does Python use?",
            answer="Quotes and apostrophes"
        ).with_inputs("context", "question")

        pred = dspy.Prediction(answer="Quotes and apostrophes")

        score = semantic_f1_metric(gold, pred)
        assert score == 1.0

    def test_markdown_formatting(self):
        """Test with markdown in context."""
        markdown_context = """# Python

Python is **great** because it has:
- Simple syntax
- Powerful libraries

__Code example__:
```python
print("Hello")
```"""

        gold = dspy.Example(
            context=markdown_context,
            question="What is Python?",
            answer="A programming language"
        ).with_inputs("context", "question")

        pred = dspy.Prediction(answer="A programming language")

        # Should not crash
        score = semantic_f1_metric(gold, pred)
        assert 0.0 <= score <= 1.0

    def test_urls_in_text(self):
        """Test with URLs in text."""
        context_with_urls = "Visit https://python.org for more info. Also check https://pypi.org"

        gold = dspy.Example(
            context=context_with_urls,
            question="Where to get Python info?",
            answer="https://python.org"
        ).with_inputs("context", "question")

        pred = dspy.Prediction(answer="https://python.org")

        score = semantic_f1_metric(gold, pred)
        assert score == 1.0

    def test_special_symbols(self):
        """Test with special symbols."""
        symbols = "@#$%^&*()_+-=[]{}|;':\",./<>?"

        gold = dspy.Example(
            context=f"Python uses symbols like {symbols}",
            question="What symbols?",
            answer="@#$%"
        ).with_inputs("context", "question")

        pred = dspy.Prediction(answer="@#$%")

        score = semantic_f1_metric(gold, pred)
        assert score == 1.0


class TestWhitespaceVariations:
    """Test various whitespace patterns."""

    def test_leading_trailing_spaces(self):
        """Test with leading/trailing spaces."""
        gold = dspy.Example(
            context="  Python is great  ",
            question="  What is Python?  ",
            answer="  A language  "
        ).with_inputs("context", "question")

        pred = dspy.Prediction(answer="A language")

        # Metrics should handle whitespace
        score = semantic_f1_metric(gold, pred)
        assert score == 1.0

    def test_multiple_spaces(self):
        """Test with multiple consecutive spaces."""
        gold = dspy.Example(
            context="Python  is  great",
            question="What  is  Python?",
            answer="A  language"
        ).with_inputs("context", "question")

        pred = dspy.Prediction(answer="A language")

        score = semantic_f1_metric(gold, pred)
        assert score == 1.0

    def test_tabs_vs_spaces(self):
        """Test with tabs instead of spaces."""
        gold = dspy.Example(
            context="Python\tis\tgreat",
            question="What\tis\tPython?",
            answer="A\tlanguage"
        ).with_inputs("context", "question")

        pred = dspy.Prediction(answer="A language")

        # Should handle tabs
        score = semantic_f1_metric(gold, pred)
        assert 0.0 <= score <= 1.0


class TestNoneHandling:
    """Test None value handling (if applicable)."""

    def test_missing_optional_fields(self):
        """Test behavior when optional fields might be missing."""
        # dspy.Example should handle this gracefully
        gold = dspy.Example(
            context="Test",
            question="Test?",
            answer="Answer"
        ).with_inputs("context", "question")

        pred = dspy.Prediction(answer="Answer")

        # Should work normally
        score = semantic_f1_metric(gold, pred)
        assert score == 1.0


class testCaseSensitivity:
    """Test case sensitivity handling."""

    def test_all_uppercase(self):
        """Test with all uppercase."""
        gold = dspy.Example(
            context="PYTHON IS GREAT",
            question="WHAT IS PYTHON?",
            answer="A PROGRAMMING LANGUAGE"
        ).with_inputs("context", "question")

        pred = dspy.Prediction(answer="A PROGRAMMING LANGUAGE")

        score = semantic_f1_metric(gold, pred)
        assert score == 1.0

    def test_all_lowercase(self):
        """Test with all lowercase."""
        gold = dspy.Example(
            context="python is great",
            question="what is python?",
            answer="a programming language"
        ).with_inputs("context", "question")

        pred = dspy.Prediction(answer="a programming language")

        score = semantic_f1_metric(gold, pred)
        assert score == 1.0

    def test_mixed_case(self):
        """Test with mixed case."""
        gold = dspy.Example(
            context="PyThOn Is GrEaT",
            question="WhAt Is PyThOn?",
            answer="A LaNgUaGe"
        ).with_inputs("context", "question")

        pred = dspy.Prediction(answer="A language")

        # Metrics should be case-insensitive
        score = semantic_f1_metric(gold, pred)
        assert score == 1.0


class TestNumericInputs:
    """Test with numeric content."""

    def test_numbers_in_text(self):
        """Test with numbers in text."""
        gold = dspy.Example(
            context="Python 3.11 was released in 2023",
            question="When was Python 3.11 released?",
            answer="2023"
        ).with_inputs("context", "question")

        pred = dspy.Prediction(answer="2023")

        score = semantic_f1_metric(gold, pred)
        assert score == 1.0

    def test_version_numbers(self):
        """Test with version numbers."""
        gold = dspy.Example(
            context="Python 3.12.0 is the latest",
            question="What version?",
            answer="3.12.0"
        ).with_inputs("context", "question")

        pred = dspy.Prediction(answer="3.12.0")

        score = semantic_f1_metric(gold, pred)
        assert score == 1.0
