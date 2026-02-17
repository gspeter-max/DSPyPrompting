"""Production-quality training dataset for DSPy Context-Based QA system.

Contains 9 positive examples (answer in context) and 3 negative examples (answer not in context).

This dataset uses production-realistic examples with:
- Longer, detailed contexts (3-5 sentences)
- Multi-sentence explanatory answers
- Nuanced questions requiring deep understanding
- Real Python documentation topics

Designed to work well with SemanticF1 metric for long-form answer evaluation.
"""

import dspy

trainset = [
    # 1. Dataclasses with field() parameters
    dspy.Example(
        context="""Python's dataclasses module provides a decorator and functions for automatically adding generated special methods to user-defined classes. The @dataclass decorator automatically generates __init__, __repr__, and __eq__ methods based on the class attributes with type annotations. For more control over individual fields, you can use the field() function which accepts parameters like default (for default values), default_factory (for callable defaults), init (include in __init__), repr (include in string representation), compare (use in comparisons), and kw_only (keyword-only argument). The field() function is particularly useful for mutable default values like lists or dictionaries, where you should use default_factory=list instead of default=[] to avoid shared state between instances.""",
        question="How do you handle mutable default values in Python dataclasses without causing shared state between instances?",
        answer="You should use the field() function with the default_factory parameter. For example, instead of default=[], you use default_factory=list. This ensures each instance gets its own separate list rather than sharing a single list across all instances, which prevents unexpected mutations affecting multiple objects."
    ).with_inputs("context", "question"),

    # 2. Async/await in comprehensions
    dspy.Example(
        context="""Python 3.9 and later support using await expressions directly in list, set, and dictionary comprehensions. This feature allows you to asynchronously iterate over iterables and build data structures in a concise syntax. When using await in comprehensions, you must be within an async function or async context. The await expression can be used in the comprehension's iterable part (before 'for') or in the expression part (after 'for'). However, each await operation is sequential within the comprehension, meaning if you need to await multiple independent operations concurrently, you should use asyncio.gather() or asyncio.TaskGroup() instead for better performance. The syntax works like: result = [await fetch_item(id) for id in item_ids] where fetch_item is an async function.""",
        question="What should you use instead of await in comprehensions when you need to execute multiple independent async operations concurrently?",
        answer="You should use asyncio.gather() or asyncio.TaskGroup() instead of await in comprehensions. This is because await operations in comprehensions execute sequentially, which is inefficient for independent operations. asyncio.gather() schedules all coroutines to run concurrently and collects their results, while TaskGroup provides structured concurrency with automatic exception handling."
    ).with_inputs("context", "question"),

    # 3. Context managers with @contextmanager
    dspy.Example(
        context="""The contextlib module provides the @contextmanager decorator which simplifies creating context managers from generator functions. When you use this decorator, the code before the yield statement serves as the __enter__ method (setup phase), and the code after yield serves as the __exit__ method (teardown phase). The value yielded becomes what's assigned to the variable in the with statement. A key advantage of @contextmanager is that the finally clause in the generator ensures cleanup code always runs, even if an exception occurs in the with block. You can also catch exceptions by wrapping the yield in a try-except block, though you need to decide whether to suppress the exception or re-raise it after cleanup. This approach is more concise than creating a class with __enter__ and __exit__ methods for simple resource management scenarios.""",
        question="How does the @contextmanager decorator ensure cleanup code always runs, even when exceptions occur?",
        answer="The @contextmanager decorator uses a finally clause in the generator function to ensure cleanup code always executes. Any code placed after the yield statement runs in a finally block, which Python guarantees to execute regardless of whether the with block completes normally or raises an exception. You can also add exception handling with a try-except around the yield if you need to catch specific errors during the with block execution."
    ).with_inputs("context", "question"),

    # 4. Generator type annotations
    dspy.Example(
        context="""Type annotations for generators use the Generator[YieldType, SendType, ReturnType] type from the typing module. The first type parameter (YieldType) indicates what the generator yields, the second (SendType) indicates what type of values it can receive via send(), and the third (ReturnType) indicates the return value when the generator finishes. For generators that only yield values and never receive anything or return a specific value, you can use Generator[YieldType, None, None]. Python's type checkers use these annotations to ensure you're using generators correctly, such as preventing you from sending values of the wrong type or expecting return values from generators that don't have them. Starting with Python 3.9, you can also use collections.abc.Generator as a generic type, though the typing module version provides more explicit documentation of all three type parameters.""",
        question="What are the three type parameters in Generator[YieldType, SendType, ReturnType] used for?",
        answer="The YieldType parameter specifies what values the generator yields, the SendType parameter indicates what type of values can be sent into the generator using the send() method, and the ReturnType parameter defines what value the generator returns when it completes (if any). This three-part type signature allows type checkers to verify correct usage of generators in both yielding and receiving values."
    ).with_inputs("context", "question"),

    # 5. Decorators with parameters
    dspy.Example(
        context="""Creating decorators that accept parameters requires three levels of nested functions. The outermost function accepts the decorator parameters and returns the actual decorator function. The middle function is the decorator itself, which receives the function being decorated. The innermost function is the wrapper that replaces the decorated function, containing the code that runs before and after the original function. This nesting is necessary because the decorator parameters need to be captured in the outer function's closure and made available to the wrapper function. A common pattern is to use functools.wraps() on the wrapper function to preserve the original function's metadata like __name__, __doc__, and __annotations__. Without functools.wraps(), the decorated function would appear to have the wrapper's metadata instead of its own, which makes debugging and introspection more difficult.""",
        question="Why do parameterized decorators require three levels of nested function definitions?",
        answer="Three levels are needed because each level serves a different purpose: the outermost function accepts and stores the decorator parameters, the middle function receives the actual function being decorated, and the innermost wrapper function contains the enhanced behavior. This nesting creates closures that capture both the decorator parameters and the original function, making them available when the wrapper is eventually called."
    ).with_inputs("context", "question"),

    # 6. Type hints with Union and Optional
    dspy.Example(
        context="""Type hints in Python use the Union and Optional types from the typing module to express flexible but constrained types. Union[X, Y] indicates a value can be either type X or type Y, while Optional[X] is shorthand for Union[X, None], meaning the value can be type X or None. Starting with Python 3.10, you can use the pipe syntax (X | Y) instead of Union[X, Y] for more concise code. Type checkers use these hints to catch potential type errors before runtime, such as trying to call a method that doesn't exist on all possible types in the Union. When working with Union types, it's good practice to use isinstance() checks or hasattr() to narrow down which specific type you're dealing with before performing type-specific operations. This helps both type checkers and human readers understand the code's intent.""",
        question="What is the difference between Optional[X] and Union[X, None] in type hints?",
        answer="There is no functional difference—Optional[X] is simply shorthand notation for Union[X, None]. Both indicate that a value can be of type X or None. Optional[X] is preferred when you want to emphasize the nullability of the type, while Union[X, None] makes the union explicit. They are completely equivalent in terms of type checking behavior."
    ).with_inputs("context", "question"),

    # 7. Property decorators
    dspy.Example(
        context="""The @property decorator transforms a method into a read-only attribute that's computed when accessed rather than stored. This allows you to encapsulate computation behind attribute access syntax, making the interface cleaner while maintaining the ability to add validation or caching later. For writable properties, you can use the @property_name.setter decorator to define a method that runs when the property is assigned to. When you use @property, Python automatically creates a descriptor object that intercepts attribute access and calls your method instead. A common pattern is to make instance variables private (with a leading underscore) and expose them through properties, which lets you add validation or computed logic without breaking the public API. Properties also don't require parentheses when accessed, making them indistinguishable from regular attributes to the caller.""",
        question="How does the @property decorator change the behavior of a method when it's accessed?",
        answer="The @property decorator transforms a method so it can be accessed like an attribute without parentheses. When you access the property name, Python automatically calls the decorated method and returns its result. This allows computed or validated values to be accessed with the same syntax as stored attributes, providing a cleaner interface while keeping the implementation flexibility of a method."
    ).with_inputs("context", "question"),

    # 8. Class methods and static methods
    dspy.Example(
        context="""Python provides @classmethod and @staticmethod decorators for defining methods that aren't tied to specific instances. A @classmethod receives the class itself as the first argument (conventionally named cls) instead of the instance (self), allowing it to access and modify class-level state. This is commonly used for alternative constructors that preprocess data before creating an instance. A @staticmethod receives neither the instance nor the class as an implicit argument, making it behave like a regular function that happens to live in a class namespace. Static methods are useful for grouping related utility functions with a class when they don't need access to instance or class data. The key difference is that classmethods can be overridden in subclasses and have access to class state, while static methods are completely independent of both instance and class state.""",
        question="What is the main difference between @classmethod and @staticmethod in Python?",
        answer="The main difference is that @classmethod receives the class as the first implicit argument (cls) and can access or modify class-level state, making it useful for alternative constructors and inheritance. @staticmethod receives no implicit first argument and behaves like a regular function, making it suitable for utility operations that don't need access to instance or class data."
    ).with_inputs("context", "question"),

    # 9. Descriptors
    dspy.Example(
        context="""Descriptors are Python objects that implement __get__, __set__, or __delete__ methods to customize attribute access on a class. When you access an attribute that's a descriptor, Python calls the descriptor's __get__ method with the instance and owner class as arguments. For assignment, Python calls __set__ with the instance and the value being assigned. This mechanism is what powers properties, class methods, and static methods—they're all implemented using descriptors under the hood. A data descriptor (one that defines both __get__ and __set__) takes precedence over instance dictionaries, while a non-data descriptor (only __get__) can be shadowed by assigning to the instance. Descriptors are powerful for creating reusable attribute management logic, such as type validation, computed attributes, or lazy loading, because the descriptor code lives in one place and can be attached to any attribute in any class.""",
        question="What distinguishes a data descriptor from a non-data descriptor in Python?",
        answer="A data descriptor defines both __get__ and __set__ methods, while a non-data descriptor only defines __get__. The key practical difference is that data descriptors take precedence over instance dictionaries, meaning you can't override them by assigning to the instance. Non-data descriptors can be shadowed by instance attributes, allowing instance-specific values to replace the descriptor's behavior when needed."
    ).with_inputs("context", "question"),

    # NEGATIVE EXAMPLES (teach model to refuse when answer not in context)

    # 10. Negative: Answer not in context (metaclasses)
    dspy.Example(
        context="""Python supports multiple inheritance through classes that can have more than one parent class. When a method is called, Python uses the Method Resolution Order (MRO) to determine which class's method to use. The MRO follows the C3 linearization algorithm to ensure a consistent order that respects inheritance hierarchies.""",
        question="How do you create a metaclass in Python?",
        answer="This information is not provided in the context"
    ).with_inputs("context", "question"),

    # 11. Negative: Answer not in context (coroutine theory)
    dspy.Example(
        context="""Python 3 introduced async and await keywords for writing asynchronous code. Async functions are defined using async def and return coroutine objects. The await keyword is used to call other async functions and wait for their results.""",
        question="What is the difference between coroutines and generators in Python's implementation?",
        answer="This information is not provided in the context"
    ).with_inputs("context", "question"),

    # 12. Negative: Answer not in context (GIL and threading)
    dspy.Example(
        context="""Python's threading module allows you to run multiple threads concurrently. Each thread shares the same memory space and can access the same objects. You need to use locks or other synchronization primitives when multiple threads access shared mutable state to prevent race conditions.""",
        question="What is the Global Interpreter Lock (GIL) and how does it affect multi-threaded Python programs?",
        answer="This information is not provided in the context"
    ).with_inputs("context", "question"),

    # 13. Negative: async/await internals
    dspy.Example(
        context="""The asyncio module in Python provides infrastructure for writing single-threaded concurrent code using coroutines, multiplexing I/O access over a thread. The event loop is the core of asyncio, running asynchronous tasks and callbacks, performing network IO operations, and managing subprocesses. Tasks are used to schedule coroutines concurrently.""",
        question="How does the event loop actually schedule tasks internally at the operating system level?",
        answer="This information is not provided in the context"
    ).with_inputs("context", "question"),

    # 14. Negative: dataclasses __init__ order
    dspy.Example(
        context="""Python dataclasses automatically generate __init__, __repr__, and __eq__ methods based on the class attributes with type annotations. The @dataclass decorator inspects the class variables and their types to create these methods. Fields can have default values and default factories for mutable defaults.""",
        question="In what order are inherited dataclass fields handled in __init__ when multiple inheritance is used?",
        answer="This information is not provided in the context"
    ).with_inputs("context", "question"),

    # 15. Negative: descriptor protocol details
    dspy.Example(
        context="""Descriptors in Python allow you to customize attribute access on classes. They implement __get__, __set__, and __delete__ methods to control how attributes are accessed, modified, or deleted. Descriptors power the underlying mechanism for properties, class methods, and static methods.""",
        question="What happens when multiple descriptors in the inheritance chain access the same attribute name?",
        answer="This information is not provided in the context"
    ).with_inputs("context", "question"),
]
