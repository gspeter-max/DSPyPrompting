"""DSPy QA module definition.

Contains the signature, module, and metric for the QA system.
"""

import dspy


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


def context_adherence_metric(gold, pred, trace=None):
    """Improved metric that validates context adherence.

    This metric checks:
    1. If answer should be "not in context", verify refusal
    2. If answer should be in context, check for semantic match
    3. More lenient with formatting than exact match

    Args:
        gold: Ground truth example with expected answer
        pred: Prediction with predicted answer
        trace: Optional trace of the prediction process

    Returns:
        bool: True if answer adheres to context guidelines
    """
    pred_answer = pred.answer.lower().strip()
    gold_answer = gold.answer.lower().strip()

    # Check for "not in context" refusal
    if "not provided in context" in gold_answer or "not mentioned" in gold_answer:
        # Model should refuse answering - check for any variation
        refusal_phrases = ["not provided in context", "not mentioned", "not in context",
                          "cannot answer", "don't know", "information not"]
        return any(phrase in pred_answer for phrase in refusal_phrases)

    # For normal answers, check if gold answer is contained in prediction
    # This handles cases like "the @ symbol" matching "@"
    if gold_answer in pred_answer:
        return True

    # Also check if prediction is contained in gold (for shorter answers)
    if pred_answer in gold_answer:
        return True

    # Check for key semantic equivalence
    # Remove common words and compare
    pred_words = set(pred_answer.split())
    gold_words = set(gold_answer.split())

    # If 80% of gold words appear in prediction, consider it a match
    if gold_words:
        overlap = len(pred_words & gold_words) / len(gold_words)
        if overlap >= 0.8:
            return True

    # Exact match as fallback
    return pred_answer == gold_answer
