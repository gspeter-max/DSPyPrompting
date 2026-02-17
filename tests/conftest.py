"""Pytest configuration and shared fixtures."""

import os
import sys
import pytest
import dspy

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Test environment flag
os.environ['TESTING'] = 'true'


@pytest.fixture
def mock_lm():
    """Mock DSPy LM for testing without API calls."""
    # Mock LM that returns predictable responses
    class MockLM:
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, prompt, **kwargs):
            # Return predictable mock response
            return dspy.Prediction(
                reasoning="Mock reasoning for testing",
                answer="mock answer"
            )

    return MockLM()


@pytest.fixture
def sample_context():
    """Sample context for testing."""
    return "Python lists are mutable sequences that can hold mixed types."


@pytest.fixture
def sample_question():
    """Sample question for testing."""
    return "What is a Python list?"


@pytest.fixture
def trained_model_path(tmp_path):
    """Path to trained model for testing."""
    return tmp_path / "test_model.json"
