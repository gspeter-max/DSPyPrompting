"""Training dataset for DSPy Context-Based QA system.

Contains 6 labeled examples for Python programming Q&A.
"""

import dspy

trainset = [
    # 1. List comprehension
    dspy.Example(
        context="""List comprehensions in Python provide a concise way to create lists.
The syntax is: [expression for item in iterable if condition]
For example: [x**2 for x in range(10) if x % 2 == 0] creates squares of even numbers from 0 to 9.""",
        question="What is the syntax for a list comprehension in Python?",
        answer="[expression for item in iterable if condition]"
    ).with_inputs("context", "question"),

    # 2. Decorators
    dspy.Example(
        context="""Decorators in Python are functions that modify the behavior of other functions.
They wrap a function, allowing you to execute code before and after the wrapped function runs.
Decorators are denoted with the @ symbol placed before the function definition.""",
        question="What symbol is used to denote a decorator in Python?",
        answer="@"
    ).with_inputs("context", "question"),

    # 3. List vs tuple
    dspy.Example(
        context="""Lists and tuples are both sequence types in Python, but they have a key difference.
Lists are mutable, meaning you can modify them after creation (add, remove, or change elements).
Tuples are immutable, meaning once created, they cannot be changed.""",
        question="Can you modify a tuple after it is created?",
        answer="No"
    ).with_inputs("context", "question"),

    # 4. *args and **kwargs
    dspy.Example(
        context="""*args and **kwargs allow you to pass a variable number of arguments to a function.
*args collects positional arguments into a tuple.
**kwargs collects keyword arguments into a dictionary.""",
        question="What does **kwargs collect keyword arguments into?",
        answer="A dictionary"
    ).with_inputs("context", "question"),

    # 5. Context managers
    dspy.Example(
        context="""Context managers in Python manage resources automatically using the 'with' statement.
They ensure that resources are properly cleaned up after use, even if an exception occurs.
The most common example is opening files: with open('file.txt') as f:""",
        question="What statement is used to work with context managers in Python?",
        answer="with"
    ).with_inputs("context", "question"),

    # 6. Generator expressions
    dspy.Example(
        context="""Generator expressions are similar to list comprehensions but use parentheses instead of brackets.
They produce items one at a time and are memory efficient.
The syntax is: (expression for item in iterable if condition)
Unlike list comprehensions, generator expressions don't create the entire list in memory.""",
        question="Do generator expressions create the entire list in memory?",
        answer="No"
    ).with_inputs("context", "question"),
]
