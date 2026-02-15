"""DSPy QA module definition.

Contains the signature, module, and metric for the QA system.
"""

import dspy


class GenerateAnswer(dspy.Signature):
    """Answer questions about Python programming based on context."""

    context = dspy.InputField(desc="Documentation or explanation about Python")
    question = dspy.InputField(desc="Question about the context")
    answer = dspy.OutputField(desc="Answer based on the given context")


class QAModule(dspy.Module):
    """QA module using Chain-of-Thought reasoning."""

    def __init__(self):
        super().__init__()
        self.generate_answer = dspy.ChainOfThought(GenerateAnswer)

    def forward(self, context, question):
        """Generate an answer based on context and question."""
        return self.generate_answer(context=context, question=question)


def answer_exact_match_metric(gold, pred, trace=None):
    """Metric function for exact answer matching.

    Args:
        gold: Ground truth example with expected answer
        pred: Prediction with predicted answer
        trace: Optional trace of the prediction process

    Returns:
        bool: True if answers match (case-insensitive)
    """
    return pred.answer.lower().strip() == gold.answer.lower().strip()
