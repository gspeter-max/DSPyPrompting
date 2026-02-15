"""DSPy QA module definition.

Contains the signature, module, and SemanticF1 metric for the QA system.
"""

import dspy
from dspy.evaluate import SemanticF1


class GenerateAnswer(dspy.Signature):
    """Answer questions using ONLY the provided context.

    CRITICAL INSTRUCTIONS:
    - You MUST answer using ONLY the information in the context
    - If the answer is not in the context, say: "This information is not provided in the context."
    - Do NOT use outside knowledge or prior training
    - Do NOT make up information
    """

    context = dspy.InputField(desc="Documentation or explanation about Python. This is the ONLY source of information for answering.")
    question = dspy.InputField(desc="Question about the context")
    answer = dspy.OutputField(desc="Answer based ONLY on the given context. Say 'not provided in context' if information is missing.")


class QAModule(dspy.Module):
    """QA module using Chain-of-Thought reasoning."""

    def __init__(self):
        super().__init__()
        self.generate_answer = dspy.ChainOfThought(GenerateAnswer)

    def forward(self, context, question):
        """Generate an answer based on context and question."""
        return self.generate_answer(context=context, question=question)


def semantic_f1_metric(gold, pred, trace=None):
    """Semantic F1 metric using DSPy's SemanticF1 for QA evaluation.

    This metric wraps DSPy's SemanticF1 to work with our QA module's field names.
    SemanticF1 expects .response field but our module uses .answer field.

    Args:
        gold: Ground truth example with expected answer
        pred: Prediction with predicted answer
        trace: Optional trace of the prediction process

    Returns:
        float: Semantic F1 score (0.0 to 1.0)
    """
    # SemanticF1 expects examples with .response field, not .answer
    # Create wrapped examples with correct field names
    wrapped_gold = dspy.Example(
        question=gold.question,
        response=gold.answer
    )

    wrapped_pred = dspy.Example(
        response=pred.answer
    )

    metric = SemanticF1(decompositional=True)
    return metric(wrapped_gold, wrapped_pred)
