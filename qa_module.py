"""DSPy QA module definition.

Contains the signature, module, and SemanticF1 metric for the QA system.
"""

import dspy
from dspy.evaluate import SemanticF1


class GenerateAnswer(dspy.Signature):
    """Answer questions with STRICT adherence to the provided context.

    CRITICAL RULES - NO EXCEPTIONS:
    1. You MAY ONLY answer if the EXACT answer is explicitly in the context
    2. If the context only MENTIONS the topic but doesn't ANSWER the question, say: "This information is not provided in the context."
    3. If ANY part of the answer requires outside knowledge, say: "This information is not provided in the context."
    4. Even if you KNOW the answer from training, if it's not in the context, say: "This information is not provided in the context."

    Examples of CORRECT refusals:
    - Context: "Python lists are mutable" → Question: "Are tuples mutable?" → Answer: "This information is not provided in the context."
    - Context: "async def creates coroutines" → Question: "How do coroutines work internally?" → Answer: "This information is not provided in the context."
    - Context: "Descriptors implement __get__" → Question: "What about multiple descriptors?" → Answer: "This information is not provided in the context."
    """

    context = dspy.InputField(desc="Documentation or explanation. This is the ONLY source of information allowed.")
    question = dspy.InputField(desc="Question about the context")
    answer = dspy.OutputField(desc="Answer based ONLY on context, or 'This information is not provided in the context.' if answer is not explicitly in context")


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

    Uses a hybrid approach:
    1. Try exact match first (most reliable for short answers)
    2. If not exact match, try SemanticF1 for nuanced evaluation
    3. Fall back to string matching if SemanticF1 fails

    Args:
        gold: Ground truth example with expected answer
        pred: Prediction with predicted answer
        trace: Optional trace of the prediction process

    Returns:
        float: Semantic F1 score (0.0 to 1.0)
    """
    pred_answer = pred.answer.lower().strip()
    gold_answer = gold.answer.lower().strip()

    # 1. Exact match (most reliable)
    if pred_answer == gold_answer:
        return 1.0

    # 2. For short answers (< 50 chars) or refusals, use fallback (SemanticF1 unreliable)
    if len(gold_answer) < 50 or len(pred_answer) < 50:
        return _fallback_metric(gold, pred)

    # 3. For longer answers, try SemanticF1 for nuanced evaluation
    try:
        wrapped_gold = dspy.Example(
            question=gold.question,
            response=gold.answer
        )

        wrapped_pred = dspy.Example(
            question=gold.question,
            response=pred.answer
        )

        metric = SemanticF1(decompositional=False)
        score = metric(wrapped_gold, wrapped_pred)
        # Ensure score is a float
        return float(score) if score is not None else 0.0
    except Exception as e:
        # If SemanticF1 fails, fall back to simple string matching
        print(f"Warning: SemanticF1 failed ({e}), using fallback metric")
        return _fallback_metric(gold, pred)


def hallucination_aware_metric(gold, pred, trace=None):
    """Metric that heavily penalizes hallucination on negative examples.

    For negative examples (where gold expects refusal):
    - Returns 1.0 if model correctly refuses
    - Returns 0.0 if model hallucinates (provides answer instead of refusing)
    - Prints debug output to identify hallucination during training

    For positive examples:
    - Uses existing semantic_f1_metric for normal evaluation

    Args:
        gold: Ground truth example
        pred: Prediction from model
        trace: Optional trace

    Returns:
        float: Score (0.0 to 1.0)
    """
    gold_lower = gold.answer.lower().strip()

    # Check if this is a negative example (expects refusal)
    is_negative = (
        "not provided" in gold_lower or
        "cannot answer" in gold_lower or
        "not mentioned" in gold_lower
    )

    if is_negative:
        # For negative examples, check if model refused
        pred_lower = pred.answer.lower().strip()
        refusal_indicators = [
            "not provided in context",
            "not mentioned in context",
            "not mentioned",
            "mentioned in the provided context",
            "information is not provided",
            "information not available",
            "not available",
            "not in the context",
            "not in context",
            "cannot answer",
            "cannot be determined",
            "don't know",
            "is not provided in the context",
            "is not provided",
            "context does not contain",
            "not in the provided context",
            "not stated in the context",
            "provided context"
        ]

        refused = any(indicator in pred_lower for indicator in refusal_indicators)

        if refused:
            # Correctly refused
            return 1.0
        else:
            # HALLUCINATION - Model answered when it should have refused
            print(f"🔴 HALLUCINATION DETECTED:")
            print(f"   Question: {gold.question[:60]}...")
            print(f"   Expected: Refusal (not provided in context)")
            print(f"   Got: {pred.answer[:80]}...")
            return 0.0  # Zero score - severe penalty

    # For positive examples, use normal semantic evaluation
    return semantic_f1_metric(gold, pred)


def _fallback_metric(gold, pred, trace=None):
    """Fallback metric when SemanticF1 fails.

    Uses simple string matching as a fallback.

    Args:
        gold: Ground truth example with expected answer
        pred: Prediction with predicted answer
        trace: Optional trace of the prediction process

    Returns:
        float: Score (0.0 or 1.0)
    """
    pred_answer = pred.answer.lower().strip()
    gold_answer = gold.answer.lower().strip()

    # Exact match (most reliable)
    if pred_answer == gold_answer:
        return 1.0

    # Check for "not in context" refusal - be more lenient
    if "not provided" in gold_answer or "not mentioned" in gold_answer or "not in context" in gold_answer:
        # Any indication that the info is not available counts as correct
        refusal_phrases = [
            "not provided in context", "not mentioned", "not in context",
            "cannot answer", "cannot be determined", "don't know", "information not",
            "information not available", "not available",
            "this information is not", "is not provided",
            "is not provided in the context", "not provided",
            "context does not contain", "not in the provided context",
            "mentioned in the provided context", "provided context"
        ]
        return 1.0 if any(phrase in pred_answer for phrase in refusal_phrases) else 0.0

    # For normal answers, check substring match (handles "the @ symbol" matching "@")
    if gold_answer in pred_answer or pred_answer in gold_answer:
        return 1.0

    # Check for word overlap
    pred_words = set(pred_answer.split())
    gold_words = set(gold_answer.split())

    if gold_words:
        overlap = len(pred_words & gold_words) / len(gold_words)
        if overlap >= 0.8:
            return 1.0

    return 0.0
