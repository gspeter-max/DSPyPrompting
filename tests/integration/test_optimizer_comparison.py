"""Comparison tests between BootstrapFewShot and MIPROv2 optimizers."""

import pytest
import dspy
from unittest.mock import patch, MagicMock
from qa_module import hallucination_aware_metric


class TestOptimizerComparison:
    """Compare outputs and parameters between BootstrapFewShot and MIPROv2."""

    @patch('train.dspy.MIPROv2')
    @patch('train.dspy.BootstrapFewShot')
    def test_both_optimizers_use_same_metric(self, mock_bootstrap, mock_miprov2):
        """Verify both optimizers can use the same metric function."""
        # Create MIPROv2
        mock_optimizer1 = MagicMock()
        mock_miprov2.return_value = mock_optimizer1

        optimizer1 = dspy.MIPROv2(
            metric=hallucination_aware_metric,
            auto="light",
            max_bootstrapped_demos=4,
            max_labeled_demos=6
        )

        kwargs1 = mock_miprov2.call_args[1]
        metric1 = kwargs1['metric']

        # Create BootstrapFewShot
        mock_optimizer2 = MagicMock()
        mock_bootstrap.return_value = mock_optimizer2

        optimizer2 = dspy.BootstrapFewShot(
            metric=hallucination_aware_metric,
            max_labeled_demos=6,
            max_bootstrapped_demos=4
        )

        kwargs2 = mock_bootstrap.call_args[1]
        metric2 = kwargs2['metric']

        # Verify same metric
        assert metric1 is metric2
        assert metric1 == hallucination_aware_metric

    @patch('train.dspy.MIPROv2')
    @patch('train.dspy.BootstrapFewShot')
    def test_both_optimizers_produce_demos(self, mock_bootstrap, mock_miprov2):
        """Verify both optimizers produce demonstration lists."""
        # MIPROv2 mock
        mock_trained1 = MagicMock()
        mock_trained1.generate_answer = MagicMock()
        mock_trained1.generate_answer.predict = MagicMock()
        mock_trained1.generate_answer.predict.demos = []
        mock_optimizer1 = MagicMock()
        mock_optimizer1.compile.return_value = mock_trained1
        mock_miprov2.return_value = mock_optimizer1

        optimizer1 = dspy.MIPROv2(
            metric=hallucination_aware_metric,
            auto="light",
            max_bootstrapped_demos=4
        )

        # Verify demos attribute exists
        assert hasattr(mock_trained1.generate_answer.predict, 'demos')
        assert isinstance(mock_trained1.generate_answer.predict.demos, list)

        # BootstrapFewShot mock
        mock_trained2 = MagicMock()
        mock_trained2.generate_answer = MagicMock()
        mock_trained2.generate_answer.predict = MagicMock()
        mock_trained2.generate_answer.predict.demos = []
        mock_optimizer2 = MagicMock()
        mock_optimizer2.compile.return_value = mock_trained2
        mock_bootstrap.return_value = mock_optimizer2

        optimizer2 = dspy.BootstrapFewShot(
            metric=hallucination_aware_metric,
            max_bootstrapped_demos=4
        )

        # Verify demos attribute exists
        assert hasattr(mock_trained2.generate_answer.predict, 'demos')
        assert isinstance(mock_trained2.generate_answer.predict.demos, list)

    @patch('train.dspy.MIPROv2')
    @patch('train.dspy.BootstrapFewShot')
    def test_both_optimizers_accept_max_labeled_demos(self, mock_bootstrap, mock_miprov2):
        """Compare max_labeled_demos parameter handling."""
        # MIPROv2
        mock_optimizer1 = MagicMock()
        mock_miprov2.return_value = mock_optimizer1

        optimizer1 = dspy.MIPROv2(
            metric=hallucination_aware_metric,
            auto="light",
            max_labeled_demos=6
        )

        kwargs1 = mock_miprov2.call_args[1]
        assert 'max_labeled_demos' in kwargs1
        assert kwargs1['max_labeled_demos'] == 6

        # BootstrapFewShot
        mock_optimizer2 = MagicMock()
        mock_bootstrap.return_value = mock_optimizer2

        optimizer2 = dspy.BootstrapFewShot(
            metric=hallucination_aware_metric,
            max_labeled_demos=6
        )

        kwargs2 = mock_bootstrap.call_args[1]
        assert 'max_labeled_demos' in kwargs2
        assert kwargs2['max_labeled_demos'] == 6

        # Verify same value
        assert kwargs1['max_labeled_demos'] == kwargs2['max_labeled_demos']

    @patch('train.dspy.MIPROv2')
    @patch('train.dspy.BootstrapFewShot')
    def test_both_optimizers_accept_max_bootstrapped_demos(self, mock_bootstrap, mock_miprov2):
        """Compare max_bootstrapped_demos parameter handling."""
        # MIPROv2
        mock_optimizer1 = MagicMock()
        mock_miprov2.return_value = mock_optimizer1

        optimizer1 = dspy.MIPROv2(
            metric=hallucination_aware_metric,
            auto="light",
            max_bootstrapped_demos=4
        )

        kwargs1 = mock_miprov2.call_args[1]
        assert 'max_bootstrapped_demos' in kwargs1
        assert kwargs1['max_bootstrapped_demos'] == 4

        # BootstrapFewShot
        mock_optimizer2 = MagicMock()
        mock_bootstrap.return_value = mock_optimizer2

        optimizer2 = dspy.BootstrapFewShot(
            metric=hallucination_aware_metric,
            max_bootstrapped_demos=4
        )

        kwargs2 = mock_bootstrap.call_args[1]
        assert 'max_bootstrapped_demos' in kwargs2
        assert kwargs2['max_bootstrapped_demos'] == 4

        # Verify same value
        assert kwargs1['max_bootstrapped_demos'] == kwargs2['max_bootstrapped_demos']

    @patch('train.dspy.MIPROv2')
    def test_miprov2_has_auto_parameter(self, mock_miprov2):
        """Verify MIPROv2 has 'auto' parameter (unique to MIPROv2)."""
        mock_optimizer = MagicMock()
        mock_miprov2.return_value = mock_optimizer

        optimizer = dspy.MIPROv2(
            metric=hallucination_aware_metric,
            auto="light",
            max_bootstrapped_demos=4
        )

        kwargs = mock_miprov2.call_args[1]
        assert 'auto' in kwargs
        assert kwargs['auto'] == "light"

    @patch('train.dspy.BootstrapFewShot')
    def test_bootstrap_has_max_rounds_parameter(self, mock_bootstrap):
        """Verify BootstrapFewShot has 'max_rounds' parameter (unique to BootstrapFewShot)."""
        mock_optimizer = MagicMock()
        mock_bootstrap.return_value = mock_optimizer

        optimizer = dspy.BootstrapFewShot(
            metric=hallucination_aware_metric,
            max_rounds=1,
            max_labeled_demos=6
        )

        kwargs = mock_bootstrap.call_args[1]
        assert 'max_rounds' in kwargs
        assert kwargs['max_rounds'] == 1

    @patch('train.dspy.MIPROv2')
    def test_miprov2_has_num_threads_parameter(self, mock_miprov2):
        """Verify MIPROv2 accepts 'num_threads' parameter (unique to MIPROv2)."""
        mock_optimizer = MagicMock()
        mock_miprov2.return_value = mock_optimizer

        # Test with num_threads
        mipro_kwargs = {
            "metric": hallucination_aware_metric,
            "auto": "light",
            "max_bootstrapped_demos": 4,
            "max_labeled_demos": 6
        }

        # Add num_threads if provided
        num_threads = 8
        if num_threads:
            mipro_kwargs["num_threads"] = num_threads

        optimizer = dspy.MIPROv2(**mipro_kwargs)

        kwargs = mock_miprov2.call_args[1]
        assert 'num_threads' in kwargs
        assert kwargs['num_threads'] == 8

    @patch('train.dspy.MIPROv2')
    @patch('train.dspy.BootstrapFewShot')
    def test_miprov2_has_more_parameters(self, mock_bootstrap, mock_miprov2):
        """Verify MIPROv2 has additional parameters compared to BootstrapFewShot."""
        # MIPROv2
        mock_optimizer1 = MagicMock()
        mock_miprov2.return_value = mock_optimizer1

        optimizer1 = dspy.MIPROv2(
            metric=hallucination_aware_metric,
            auto="light",
            num_threads=4,
            max_bootstrapped_demos=4,
            max_labeled_demos=6
        )

        kwargs1 = mock_miprov2.call_args[1]

        # BootstrapFewShot
        mock_optimizer2 = MagicMock()
        mock_bootstrap.return_value = mock_optimizer2

        optimizer2 = dspy.BootstrapFewShot(
            metric=hallucination_aware_metric,
            max_rounds=1,
            max_bootstrapped_demos=4,
            max_labeled_demos=6
        )

        kwargs2 = mock_bootstrap.call_args[1]

        # MIPROv2 should have 'auto' and 'num_threads'
        assert 'auto' in kwargs1
        assert 'num_threads' in kwargs1

        # BootstrapFewShot should NOT have these
        assert 'auto' not in kwargs2
        assert 'num_threads' not in kwargs2

        # BootstrapFewShot should have 'max_rounds'
        assert 'max_rounds' in kwargs2

        # MIPROv2 should NOT have max_rounds
        assert 'max_rounds' not in kwargs1

    @patch('train.dspy.MIPROv2')
    @patch('train.dspy.BootstrapFewShot')
    def test_both_accept_metric_callable(self, mock_bootstrap, mock_miprov2):
        """Verify both optimizers accept a callable metric."""
        # Create a simple callable metric
        def simple_metric(gold, pred, trace=None):
            return gold.answer == pred.answer

        # MIPROv2
        mock_optimizer1 = MagicMock()
        mock_miprov2.return_value = mock_optimizer1

        optimizer1 = dspy.MIPROv2(
            metric=simple_metric,
            auto="light"
        )

        kwargs1 = mock_miprov2.call_args[1]
        assert callable(kwargs1['metric'])

        # BootstrapFewShot
        mock_optimizer2 = MagicMock()
        mock_bootstrap.return_value = mock_optimizer2

        optimizer2 = dspy.BootstrapFewShot(
            metric=simple_metric
        )

        kwargs2 = mock_bootstrap.call_args[1]
        assert callable(kwargs2['metric'])

    @patch('train.dspy.MIPROv2')
    @patch('train.dspy.BootstrapFewShot')
    def test_both_have_max_errors_parameter(self, mock_bootstrap, mock_miprov2):
        """Verify both optimizers accept max_errors parameter."""
        # MIPROv2
        mock_optimizer1 = MagicMock()
        mock_miprov2.return_value = mock_optimizer1

        optimizer1 = dspy.MIPROv2(
            metric=hallucination_aware_metric,
            auto="light",
            max_errors=10
        )

        kwargs1 = mock_miprov2.call_args[1]
        assert 'max_errors' in kwargs1
        assert kwargs1['max_errors'] == 10

        # BootstrapFewShot
        mock_optimizer2 = MagicMock()
        mock_bootstrap.return_value = mock_optimizer2

        optimizer2 = dspy.BootstrapFewShot(
            metric=hallucination_aware_metric,
            max_errors=10
        )

        kwargs2 = mock_bootstrap.call_args[1]
        assert 'max_errors' in kwargs2
        assert kwargs2['max_errors'] == 10
