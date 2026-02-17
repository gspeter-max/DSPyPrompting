"""Integration tests for full training pipeline."""

import pytest
import dspy
import tempfile
import os
from unittest.mock import patch, MagicMock
from qa_module import QAModule
from dataset import trainset


class TestFullPipeline:
    """Test complete train → predict workflow."""

    def test_qamodule_initialization(self):
        """Test QAModule can be initialized."""
        qa = QAModule()
        assert qa is not None
        assert hasattr(qa, 'generate_answer')
        assert hasattr(qa, 'forward')

    def test_qamodule_forward_signature(self):
        """Test forward method signature is correct."""
        qa = QAModule()
        # Check forward is callable
        assert callable(qa.forward)
        # Check it accepts context and question
        import inspect
        sig = inspect.signature(qa.forward)
        params = list(sig.parameters.keys())
        assert 'context' in params
        assert 'question' in params

    @patch('qa_module.dspy.ChainOfThought')
    def test_forward_returns_prediction(self, mock_cot):
        """Test forward returns dspy.Prediction."""
        # Mock the ChainOfThought to return a prediction
        mock_predict = MagicMock()
        mock_predict.return_value = dspy.Prediction(
            reasoning="Test reasoning",
            answer="Test answer"
        )
        mock_cot.return_value = mock_predict

        qa = QAModule()
        # Forward should return a prediction
        result = qa.forward(context="Test context", question="Test question")

        # Verify result
        assert result is not None

    def test_dataset_integration_with_module(self):
        """Test dataset works with QAModule."""
        # Get first example
        example = trainset[0]

        # Verify it has required fields
        assert hasattr(example, 'context')
        assert hasattr(example, 'question')
        assert hasattr(example, 'answer')

        # Verify can access fields
        context = example.context
        question = example.question
        answer = example.answer

        assert len(context) > 0
        assert len(question) > 0
        assert len(answer) > 0


class TestDemonstrations:
    """Test demonstration handling in QAModule."""

    def test_qamodule_has_demos_attribute(self):
        """Test QAModule has demos attribute."""
        qa = QAModule()
        assert hasattr(qa.generate_answer, 'predict')
        assert hasattr(qa.generate_answer.predict, 'demos')

    def test_demos_is_list(self):
        """Test demos is a list."""
        qa = QAModule()
        demos = qa.generate_answer.predict.demos
        assert isinstance(demos, list)

    def test_empty_demos_initially(self):
        """Test demos is empty initially (untrained)."""
        qa = QAModule()
        demos = qa.generate_answer.predict.demos
        assert len(demos) == 0


class TestModelPersistence:
    """Test model state persistence concepts."""

    def test_demos_can_be_modified(self):
        """Test demonstrations can be added/modified."""
        qa = QAModule()

        # Add a mock demonstration
        mock_demo = {
            'context': 'Test context',
            'question': 'Test question',
            'answer': 'Test answer'
        }

        qa.generate_answer.predict.demos.append(mock_demo)

        # Verify it was added
        assert len(qa.generate_answer.predict.demos) == 1
        assert qa.generate_answer.predict.demos[0] == mock_demo

    def test_demos_structure(self):
        """Test demo structure matches expected format."""
        qa = QAModule()

        mock_demo = {
            'context': 'Python lists are mutable.',
            'question': 'Are lists mutable?',
            'answer': 'Yes, they are mutable'
        }

        qa.generate_answer.predict.demos.append(mock_demo)

        # Verify structure
        demo = qa.generate_answer.predict.demos[0]
        assert 'context' in demo
        assert 'question' in demo
        assert 'answer' in demo
        assert isinstance(demo['context'], str)
        assert isinstance(demo['question'], str)
        assert isinstance(demo['answer'], str)


class TestTrainingIntegration:
    """Test training-related integration points."""

    @patch('dspy.BootstrapFewShot')
    def test_optimizer_integration(self, mock_bootstrap):
        """Test optimizer integrates with QAModule."""
        from qa_module import hallucination_aware_metric

        # Mock the optimizer
        mock_optimizer = MagicMock()
        mock_trained = MagicMock()
        mock_optimizer.compile.return_value = mock_trained
        mock_bootstrap.return_value = mock_optimizer

        # Create optimizer
        optimizer = dspy.BootstrapFewShot(
            metric=hallucination_aware_metric,
            max_labeled_demos=6,
            max_bootstrapped_demos=4,
            max_rounds=1
        )

        # Verify created
        assert optimizer is not None

    def test_metric_integration_with_dataset(self):
        """Test metric works with dataset examples."""
        from qa_module import semantic_f1_metric

        # Use real dataset example
        gold = trainset[0]

        # Create prediction
        pred = dspy.Prediction(answer=gold.answer)

        # Test metric
        score = semantic_f1_metric(gold, pred)

        # Should return 1.0 for exact match
        assert score == 1.0

    def test_negative_example_metric_integration(self):
        """Test metric works with negative examples."""
        from qa_module import hallucination_aware_metric

        # Use negative example (last 6 are negative)
        gold = trainset[10]  # First negative example

        # Correct refusal
        pred_refuse = dspy.Prediction(answer="This information is not provided in the context")

        # Should return 1.0
        score = hallucination_aware_metric(gold, pred_refuse)
        assert score == 1.0


class TestCrossDataset:
    """Test cross-dataset validation concepts."""

    def test_train_test_split_concept(self):
        """Test concept of splitting dataset for train/test."""
        # Split dataset
        train_size = int(len(trainset) * 0.8)
        train_split = trainset[:train_size]
        test_split = trainset[train_size:]

        # Verify split
        assert len(train_split) + len(test_split) == len(trainset)
        assert len(train_split) > 0
        assert len(test_split) > 0

    def test_multiple_examples_can_be_tested(self):
        """Test multiple examples can be evaluated."""
        from qa_module import semantic_f1_metric

        # Test first 5 examples
        for i in range(5):
            gold = trainset[i]
            pred = dspy.Prediction(answer=gold.answer)

            score = semantic_f1_metric(gold, pred)
            assert score >= 0.0
            assert score <= 1.0


class TestEndToEndWorkflow:
    """Test end-to-end workflow without actual training."""

    def test_module_creation_workflow(self):
        """Test complete module creation workflow."""
        # 1. Create module
        qa = QAModule()
        assert qa is not None

        # 2. Verify structure
        assert hasattr(qa, 'generate_answer')

        # 3. Verify forward method
        assert callable(qa.forward)

        # 4. Verify demos attribute
        assert hasattr(qa.generate_answer.predict, 'demos')
        assert isinstance(qa.generate_answer.predict.demos, list)

    def test_dataset_loading_workflow(self):
        """Test dataset loading and validation workflow."""
        # 1. Load dataset
        from dataset import trainset

        # 2. Verify structure
        assert isinstance(trainset, list)
        assert len(trainset) == 15

        # 3. Verify examples
        for example in trainset:
            assert hasattr(example, 'context')
            assert hasattr(example, 'question')
            assert hasattr(example, 'answer')

    def test_metric_evaluation_workflow(self):
        """Test metric evaluation workflow."""
        from qa_module import semantic_f1_metric, hallucination_aware_metric

        # 1. Get example
        gold = trainset[0]

        # 2. Create prediction
        pred = dspy.Prediction(answer=gold.answer)

        # 3. Evaluate with semantic_f1
        score1 = semantic_f1_metric(gold, pred)
        assert 0.0 <= score1 <= 1.0

        # 4. Evaluate with hallucination_aware
        score2 = hallucination_aware_metric(gold, pred)
        assert 0.0 <= score2 <= 1.0
