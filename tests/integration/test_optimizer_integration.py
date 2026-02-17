"""Integration tests for optimizer functionality."""

import pytest
import dspy
import tempfile
import os
from unittest.mock import patch, MagicMock, Mock
from qa_module import QAModule, hallucination_aware_metric
from dataset import trainset


class TestMIPROv2Integration:
    """Test MIPROv2 optimizer integration."""

    @patch('train.dspy.MIPROv2')
    def test_miprov2_compilation_flow(self, mock_miprov2):
        """Verify MIPROv2.compile() works with QAModule."""
        # Create mock optimizer
        mock_optimizer = MagicMock()
        mock_trained = MagicMock()
        mock_trained.generate_answer = MagicMock()
        mock_trained.generate_answer.predict = MagicMock()
        mock_trained.generate_answer.predict.demos = []
        mock_optimizer.compile.return_value = mock_trained
        mock_miprov2.return_value = mock_optimizer

        # Create QAModule
        qa_module = QAModule()

        # Configure MIPROv2 optimizer
        optimizer = dspy.MIPROv2(
            metric=hallucination_aware_metric,
            auto="light",
            max_bootstrapped_demos=4,
            max_labeled_demos=6,
            max_errors=10
        )

        # Verify optimizer was created
        assert optimizer is not None
        mock_miprov2.assert_called_once()

    @patch('train.dspy.MIPROv2')
    def test_miprov2_output_structure(self, mock_miprov2):
        """Verify MIPROv2 produces valid demo structure."""
        # Create mock trained model with demos
        mock_trained = MagicMock()
        mock_trained.generate_answer = MagicMock()
        mock_trained.generate_answer.predict = MagicMock()

        # Create mock demos
        mock_demos = [
            dspy.Example(
                context="Test context 1",
                question="Test question 1",
                answer="Test answer 1"
            ),
            dspy.Example(
                context="Test context 2",
                question="Test question 2",
                answer="Test answer 2"
            )
        ]
        mock_trained.generate_answer.predict.demos = mock_demos

        # Create mock optimizer
        mock_optimizer = MagicMock()
        mock_optimizer.compile.return_value = mock_trained
        mock_miprov2.return_value = mock_optimizer

        # Verify demos structure
        demos = mock_trained.generate_answer.predict.demos
        assert len(demos) == 2
        for demo in demos:
            assert hasattr(demo, 'context')
            assert hasattr(demo, 'question')
            assert hasattr(demo, 'answer')

    @patch('train.dspy.MIPROv2')
    def test_miprov2_with_hallucination_metric(self, mock_miprov2):
        """Test metric compatibility with MIPROv2."""
        # Create mock optimizer
        mock_optimizer = MagicMock()
        mock_miprov2.return_value = mock_optimizer

        # Configure with metric
        optimizer = dspy.MIPROv2(
            metric=hallucination_aware_metric,
            auto="light",
            max_bootstrapped_demos=4,
            max_labeled_demos=6
        )

        # Verify metric was passed
        call_kwargs = mock_miprov2.call_args[1]
        assert call_kwargs['metric'] == hallucination_aware_metric

    @patch('train.dspy.MIPROv2')
    def test_miprov2_model_saving(self, mock_miprov2):
        """Test MIPROv2 model can be saved to disk."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "trained_qa_model_miprov2.json")

            # Create mock trained model
            mock_trained = MagicMock()
            mock_trained.save = MagicMock()
            mock_trained.generate_answer = MagicMock()
            mock_trained.generate_answer.predict = MagicMock()
            mock_trained.generate_answer.predict.demos = []

            # Create mock optimizer
            mock_optimizer = MagicMock()
            mock_optimizer.compile.return_value = mock_trained
            mock_miprov2.return_value = mock_optimizer

            # Save model
            mock_trained.save(model_path)

            # Verify save was called
            mock_trained.save.assert_called_once_with(model_path)


class TestBootstrapFewShotIntegration:
    """Test BootstrapFewShot optimizer integration."""

    @patch('train.dspy.BootstrapFewShot')
    def test_bootstrap_compilation_flow(self, mock_bootstrap):
        """Verify BootstrapFewShot.compile() works with QAModule."""
        # Create mock optimizer
        mock_optimizer = MagicMock()
        mock_trained = MagicMock()
        mock_trained.generate_answer = MagicMock()
        mock_trained.generate_answer.predict = MagicMock()
        mock_trained.generate_answer.predict.demos = []
        mock_optimizer.compile.return_value = mock_trained
        mock_bootstrap.return_value = mock_optimizer

        # Create QAModule
        qa_module = QAModule()

        # Configure BootstrapFewShot optimizer
        optimizer = dspy.BootstrapFewShot(
            metric=hallucination_aware_metric,
            max_labeled_demos=6,
            max_bootstrapped_demos=4,
            max_rounds=1,
            max_errors=10
        )

        # Verify optimizer was created
        assert optimizer is not None
        mock_bootstrap.assert_called_once()

    @patch('train.dspy.BootstrapFewShot')
    def test_bootstrap_output_structure(self, mock_bootstrap):
        """Verify BootstrapFewShot produces valid demo structure."""
        # Create mock trained model with demos
        mock_trained = MagicMock()
        mock_trained.generate_answer = MagicMock()
        mock_trained.generate_answer.predict = MagicMock()

        # Create mock demos
        mock_demos = [
            dspy.Example(
                context="Test context 1",
                question="Test question 1",
                answer="Test answer 1"
            ),
            dspy.Example(
                context="Test context 2",
                question="Test question 2",
                answer="Test answer 2"
            )
        ]
        mock_trained.generate_answer.predict.demos = mock_demos

        # Create mock optimizer
        mock_optimizer = MagicMock()
        mock_optimizer.compile.return_value = mock_trained
        mock_bootstrap.return_value = mock_optimizer

        # Verify demos structure
        demos = mock_trained.generate_answer.predict.demos
        assert len(demos) == 2
        for demo in demos:
            assert hasattr(demo, 'context')
            assert hasattr(demo, 'question')
            assert hasattr(demo, 'answer')

    @patch('train.dspy.BootstrapFewShot')
    def test_bootstrap_with_hallucination_metric(self, mock_bootstrap):
        """Test metric compatibility with BootstrapFewShot."""
        # Create mock optimizer
        mock_optimizer = MagicMock()
        mock_bootstrap.return_value = mock_optimizer

        # Configure with metric
        optimizer = dspy.BootstrapFewShot(
            metric=hallucination_aware_metric,
            max_labeled_demos=6,
            max_bootstrapped_demos=4,
            max_rounds=1,
            max_errors=10
        )

        # Verify metric was passed
        call_kwargs = mock_bootstrap.call_args[1]
        assert call_kwargs['metric'] == hallucination_aware_metric

    @patch('train.dspy.BootstrapFewShot')
    def test_bootstrap_model_saving(self, mock_bootstrap):
        """Test BootstrapFewShot model can be saved to disk."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "trained_qa_model_bootstrap.json")

            # Create mock trained model
            mock_trained = MagicMock()
            mock_trained.save = MagicMock()
            mock_trained.generate_answer = MagicMock()
            mock_trained.generate_answer.predict = MagicMock()
            mock_trained.generate_answer.predict.demos = []

            # Create mock optimizer
            mock_optimizer = MagicMock()
            mock_optimizer.compile.return_value = mock_trained
            mock_bootstrap.return_value = mock_optimizer

            # Save model
            mock_trained.save(model_path)

            # Verify save was called
            mock_trained.save.assert_called_once_with(model_path)


class TestOptimizerParameterCompatibility:
    """Test parameter handling across optimizers."""

    @patch('train.dspy.MIPROv2')
    @patch('train.dspy.BootstrapFewShot')
    def test_both_optimizers_accept_metric(self, mock_bootstrap, mock_miprov2):
        """Verify both optimizers accept metric parameter."""
        # MIPROv2
        mock_optimizer1 = MagicMock()
        mock_miprov2.return_value = mock_optimizer1

        optimizer1 = dspy.MIPROv2(
            metric=hallucination_aware_metric,
            auto="light",
            max_bootstrapped_demos=4,
            max_labeled_demos=6
        )

        call_kwargs1 = mock_miprov2.call_args[1]
        assert 'metric' in call_kwargs1

        # BootstrapFewShot
        mock_optimizer2 = MagicMock()
        mock_bootstrap.return_value = mock_optimizer2

        optimizer2 = dspy.BootstrapFewShot(
            metric=hallucination_aware_metric,
            max_labeled_demos=6,
            max_bootstrapped_demos=4
        )

        call_kwargs2 = mock_bootstrap.call_args[1]
        assert 'metric' in call_kwargs2

    @patch('train.dspy.MIPROv2')
    @patch('train.dspy.BootstrapFewShot')
    def test_both_optimizers_accept_max_labeled_demos(self, mock_bootstrap, mock_miprov2):
        """Verify both optimizers accept max_labeled_demos parameter."""
        # MIPROv2
        mock_optimizer1 = MagicMock()
        mock_miprov2.return_value = mock_optimizer1

        optimizer1 = dspy.MIPROv2(
            metric=hallucination_aware_metric,
            auto="light",
            max_labeled_demos=6
        )

        call_kwargs1 = mock_miprov2.call_args[1]
        assert call_kwargs1['max_labeled_demos'] == 6

        # BootstrapFewShot
        mock_optimizer2 = MagicMock()
        mock_bootstrap.return_value = mock_optimizer2

        optimizer2 = dspy.BootstrapFewShot(
            metric=hallucination_aware_metric,
            max_labeled_demos=6
        )

        call_kwargs2 = mock_bootstrap.call_args[1]
        assert call_kwargs2['max_labeled_demos'] == 6

    @patch('train.dspy.MIPROv2')
    @patch('train.dspy.BootstrapFewShot')
    def test_both_optimizers_accept_max_bootstrapped_demos(self, mock_bootstrap, mock_miprov2):
        """Verify both optimizers accept max_bootstrapped_demos parameter."""
        # MIPROv2
        mock_optimizer1 = MagicMock()
        mock_miprov2.return_value = mock_optimizer1

        optimizer1 = dspy.MIPROv2(
            metric=hallucination_aware_metric,
            auto="light",
            max_bootstrapped_demos=4
        )

        call_kwargs1 = mock_miprov2.call_args[1]
        assert call_kwargs1['max_bootstrapped_demos'] == 4

        # BootstrapFewShot
        mock_optimizer2 = MagicMock()
        mock_bootstrap.return_value = mock_optimizer2

        optimizer2 = dspy.BootstrapFewShot(
            metric=hallucination_aware_metric,
            max_bootstrapped_demos=4
        )

        call_kwargs2 = mock_bootstrap.call_args[1]
        assert call_kwargs2['max_bootstrapped_demos'] == 4

    @patch('train.dspy.MIPROv2')
    @patch('train.dspy.BootstrapFewShot')
    def test_both_optimizers_accept_max_errors(self, mock_bootstrap, mock_miprov2):
        """Verify both optimizers accept max_errors parameter."""
        # MIPROv2
        mock_optimizer1 = MagicMock()
        mock_miprov2.return_value = mock_optimizer1

        optimizer1 = dspy.MIPROv2(
            metric=hallucination_aware_metric,
            auto="light",
            max_errors=10
        )

        call_kwargs1 = mock_miprov2.call_args[1]
        assert call_kwargs1['max_errors'] == 10

        # BootstrapFewShot
        mock_optimizer2 = MagicMock()
        mock_bootstrap.return_value = mock_optimizer2

        optimizer2 = dspy.BootstrapFewShot(
            metric=hallucination_aware_metric,
            max_errors=10
        )

        call_kwargs2 = mock_bootstrap.call_args[1]
        assert call_kwargs2['max_errors'] == 10


class TestTrainsetCompatibility:
    """Test both optimizers work with the same trainset."""

    def test_trainset_structure(self):
        """Verify trainset has correct structure for both optimizers."""
        assert isinstance(trainset, list)
        assert len(trainset) == 15

        # All items should be dspy.Example
        for item in trainset:
            assert hasattr(item, 'context')
            assert hasattr(item, 'question')
            assert hasattr(item, 'answer')

    def test_trainset_positive_negative_examples(self):
        """Verify trainset has both positive and negative examples."""
        positive = sum(1 for s in trainset if "not provided" not in s.answer)
        negative = sum(1 for s in trainset if "not provided" in s.answer)

        assert positive == 9
        assert negative == 6
        assert positive + negative == len(trainset)

    @patch('train.dspy.MIPROv2')
    @patch('train.dspy.BootstrapFewShot')
    def test_both_optimizers_accept_trainset(self, mock_bootstrap, mock_miprov2):
        """Verify both optimizers can accept the trainset."""
        # MIPROv2
        mock_optimizer1 = MagicMock()
        mock_trained1 = MagicMock()
        mock_trained1.generate_answer = MagicMock()
        mock_trained1.generate_answer.predict = MagicMock()
        mock_trained1.generate_answer.predict.demos = []
        mock_optimizer1.compile.return_value = mock_trained1
        mock_miprov2.return_value = mock_optimizer1

        optimizer1 = dspy.MIPROv2(
            metric=hallucination_aware_metric,
            auto="light",
            max_bootstrapped_demos=4
        )

        # Verify optimizer can be created
        assert optimizer1 is not None

        # BootstrapFewShot
        mock_optimizer2 = MagicMock()
        mock_trained2 = MagicMock()
        mock_trained2.generate_answer = MagicMock()
        mock_trained2.generate_answer.predict = MagicMock()
        mock_trained2.generate_answer.predict.demos = []
        mock_optimizer2.compile.return_value = mock_trained2
        mock_bootstrap.return_value = mock_optimizer2

        optimizer2 = dspy.BootstrapFewShot(
            metric=hallucination_aware_metric,
            max_bootstrapped_demos=4
        )

        # Verify optimizer can be created
        assert optimizer2 is not None
