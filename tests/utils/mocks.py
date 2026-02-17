"""Mock objects for testing."""

import dspy
from unittest.mock import MagicMock


class MockLLM:
    """Mock DSPy LM for testing without API calls.

    This mock returns predictable responses without making actual API calls.
    Useful for unit tests where you want to avoid network dependencies.
    """

    def __init__(self, response="mock answer", reasoning="Mock reasoning"):
        self.response = response
        self.reasoning = reasoning
        self.call_count = 0
        self.call_history = []

    def __call__(self, prompt, **kwargs):
        """Simulate LM call."""
        self.call_count += 1
        self.call_history.append({
            'prompt': prompt,
            'kwargs': kwargs,
            'call_number': self.call_count
        })

        return dspy.Prediction(
            reasoning=f"{self.reasoning} (call #{self.call_count})",
            answer=self.response
        )

    def reset(self):
        """Reset call history."""
        self.call_count = 0
        self.call_history = []


class MockQAModule:
    """Mock QAModule for testing.

    This mock provides deterministic answers without needing actual LLM calls.
    """

    def __init__(self, fixed_answer="test answer"):
        self.fixed_answer = fixed_answer
        self.call_count = 0

    def forward(self, context, question):
        """Mock forward method."""
        self.call_count += 1
        return dspy.Prediction(
            reasoning=f"Mock reasoning for: {question[:50]}",
            answer=self.fixed_answer
        )

    def __call__(self, context, question):
        """Allow calling instance directly."""
        return self.forward(context, question)


class MockChainOfThought:
    """Mock dspy.ChainOfThought for testing."""

    def __init__(self, signature):
        self.signature = signature
        self.demos = []
        self.call_count = 0

    def __call__(self, **kwargs):
        """Mock prediction."""
        self.call_count += 1
        return dspy.Prediction(
            reasoning="Mock ChainOfThought reasoning",
            answer="Mock answer"
        )


def mock_api_call(prompt, **kwargs):
    """Mock API call that returns predictable response.

    Args:
        prompt: The prompt text
        **kwargs: Additional arguments

    Returns:
        dspy.Prediction with mock response
    """
    return dspy.Prediction(
        reasoning="Mock reasoning from API",
        answer="Mock answer from API"
    )


def mock_prediction(answer=None, reasoning=None):
    """Create a mock dspy.Prediction.

    Args:
        answer: The answer text (default: "mock answer")
        reasoning: The reasoning text (default: "mock reasoning")

    Returns:
        dspy.Prediction instance
    """
    return dspy.Prediction(
        reasoning=reasoning or "mock reasoning",
        answer=answer or "mock answer"
    )


def mock_example(context=None, question=None, answer=None):
    """Create a mock dspy.Example.

    Args:
        context: The context text
        question: The question text
        answer: The answer text

    Returns:
        dspy.Example instance with with_inputs() called
    """
    return dspy.Example(
        context=context or "mock context",
        question=question or "mock question",
        answer=answer or "mock answer"
    ).with_inputs("context", "question")


class MockOptimizer:
    """Mock BootstrapFewShot optimizer for testing."""

    def __init__(self, **kwargs):
        self.config = kwargs
        self.compile_call_count = 0
        self.mock_trained_module = None

    def compile(self, student, trainset):
        """Mock compile method."""
        self.compile_call_count += 1

        # Return a mock trained module
        trained = MagicMock()
        trained.generate_answer.predict.demos = [
            {
                'context': 'Mock demo context',
                'question': 'Mock demo question',
                'answer': 'Mock demo answer'
            }
        ]

        return trained


def mock_trained_qa():
    """Create a mock trained QAModule with demonstrations.

    Returns:
        MagicMock object mimicking a trained QAModule
    """
    trained = MagicMock()
    trained.generate_answer.predict.demos = [
        {
            'context': 'Python lists are mutable',
            'question': 'Are lists mutable?',
            'answer': 'Yes, they are mutable'
        },
        {
            'context': 'Tuples are immutable',
            'question': 'Are tuples mutable?',
            'answer': 'No, they are immutable'
        }
    ]

    return trained


class MockMetricTracker:
    """Track metric calls during testing."""

    def __init__(self):
        self.calls = []

    def __call__(self, gold, pred, trace=None):
        """Track metric call."""
        self.calls.append({
            'gold': gold,
            'pred': pred,
            'trace': trace
        })

        # Return a default score
        if gold.answer.lower() == pred.answer.lower():
            return 1.0
        return 0.0

    def reset(self):
        """Reset tracking."""
        self.calls = []

    def get_call_count(self):
        """Get number of times metric was called."""
        return len(self.calls)
