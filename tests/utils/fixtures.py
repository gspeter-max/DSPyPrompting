"""Test data fixtures."""

import dspy


# Sample positive example (answer in context)
POSITIVE_EXAMPLE = dspy.Example(
    context="Python lists are mutable sequences that can hold mixed types. They are one of the most commonly used data structures in Python programming.",
    question="Are Python lists mutable?",
    answer="Yes, Python lists are mutable"
).with_inputs("context", "question")

# Sample negative example (answer not in context)
NEGATIVE_EXAMPLE = dspy.Example(
    context="Python lists exist in the language. They are widely used for storing collections of items.",
    question="What is a tuple?",
    answer="This information is not provided in the context"
).with_inputs("context", "question")

# Edge case samples
EMPTY_CONTEXT = dspy.Example(
    context="",
    question="What is Python?",
    answer="This information is not provided in the context"
).with_inputs("context", "question")

UNICODE_CONTEXT = dspy.Example(
    context="Python支持Unicode字符: 你好世界 🌍 Python fully supports international text and emojis.",
    question="Does Python support Unicode?",
    answer="Yes, Python supports Unicode characters including emojis and international text"
).with_inputs("context", "question")

LONG_CONTEXT = dspy.Example(
    context="Python is a programming language. " * 100,
    question="What is Python?",
    answer="Python is a high-level programming language"
).with_inputs("context", "question")

# Short examples
SHORT_ANSWER = dspy.Example(
    context="Lists are mutable",
    question="Are lists mutable?",
    answer="Yes"
).with_inputs("context", "question")

LONG_ANSWER = dspy.Example(
    context="Python is a versatile language",
    question="Tell me about Python",
    answer="Python is a high-level, interpreted programming language known for its simple syntax. " * 10
).with_inputs("context", "question")

# Special character examples
SPECIAL_CHARS_CONTEXT = dspy.Example(
    context='Python uses special characters: @ for decorators, # for comments, $ in regex, % for modulo.',
    question="What does @ do?",
    answer="@ is used for decorators in Python"
).with_inputs("context", "question")

# Refusal variations
REFUSAL_STANDARD = dspy.Example(
    context="Python exists",
    question="What about other languages?",
    answer="This information is not provided in the context"
).with_inputs("context", "question")

# Partial match examples
PARTIAL_MATCH_GOLD = dspy.Example(
    context="Python lists are mutable sequences that can hold mixed types of data",
    question="What are Python lists?",
    answer="Python lists are mutable sequences"
).with_inputs("context", "question")

# Code snippet example
CODE_SNIPPET = dspy.Example(
    context="Python uses print() to output text. Example: print('Hello, World!') displays a message.",
    question="How do you print in Python?",
    answer="Use the print() function: print('Hello, World!')"
).with_inputs("context", "question")

# Number-heavy example
NUMBERS_EXAMPLE = dspy.Example(
    context="Python 3.11 was released in October 2022, with version 3.12 following in October 2023.",
    question="When was Python 3.11 released?",
    answer="October 2022"
).with_inputs("context", "question")


def get_positive_examples(count=5):
    """Get first N positive examples from trainset."""
    from dataset import trainset
    return trainset[:count]


def get_negative_examples(count=3):
    """Get last N negative examples from trainset."""
    from dataset import trainset
    return trainset[-count:]


def get_mixed_examples(count=5):
    """Get N mixed examples from trainset."""
    from dataset import trainset
    return trainset[:count]
