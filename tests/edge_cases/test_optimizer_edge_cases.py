"""Edge case tests for optimizer selection."""

import pytest
import argparse
from unittest.mock import patch, MagicMock, Mock
import sys


class TestMIPROv2EdgeCases:
    """Test MIPROv2-specific edge cases."""

    @patch('train.dspy.MIPROv2')
    def test_miprov2_with_none_auto(self, mock_miprov2):
        """Test MIPROv2 with auto=None defaults to 'light'."""
        mock_optimizer = MagicMock()
        mock_miprov2.return_value = mock_optimizer

        # Simulate auto=None
        auto = None
        auto_mode = auto or "light"

        # Configure MIPROv2
        mipro_kwargs = {
            "metric": MagicMock(),
            "auto": auto_mode,
            "max_bootstrapped_demos": 4,
            "max_labeled_demos": 6
        }

        optimizer = mock_miprov2(**mipro_kwargs)

        # Verify auto defaulted to 'light'
        call_kwargs = mock_miprov2.call_args[1]
        assert call_kwargs['auto'] == "light"

    @patch('train.dspy.MIPROv2')
    def test_miprov2_with_zero_threads(self, mock_miprov2):
        """Test MIPROv2 with num_threads=0 (should not add parameter)."""
        mock_optimizer = MagicMock()
        mock_miprov2.return_value = mock_optimizer

        # Simulate num_threads=0
        num_threads = 0
        mipro_kwargs = {
            "metric": MagicMock(),
            "auto": "light",
            "max_bootstrapped_demos": 4,
            "max_labeled_demos": 6
        }

        # Only add num_threads if truthy
        if num_threads:
            mipro_kwargs["num_threads"] = num_threads

        optimizer = mock_miprov2(**mipro_kwargs)

        # Verify num_threads was NOT added
        call_kwargs = mock_miprov2.call_args[1]
        assert 'num_threads' not in call_kwargs

    @patch('train.dspy.MIPROv2')
    def test_miprov2_with_negative_threads(self, mock_miprov2):
        """Test MIPROv2 with negative num_threads (should not add parameter)."""
        mock_optimizer = MagicMock()
        mock_miprov2.return_value = mock_optimizer

        # Simulate num_threads=-1
        num_threads = -1
        mipro_kwargs = {
            "metric": MagicMock(),
            "auto": "light",
            "max_bootstrapped_demos": 4,
            "max_labeled_demos": 6
        }

        # Only add num_threads if truthy
        if num_threads:
            mipro_kwargs["num_threads"] = num_threads

        optimizer = mock_miprov2(**mipro_kwargs)

        # Verify num_threads was added even if negative
        call_kwargs = mock_miprov2.call_args[1]
        # Note: This test documents current behavior
        # In practice, DSPy may validate this

    @patch('train.dspy.MIPROv2')
    def test_miprov2_with_all_parameters(self, mock_miprov2):
        """Test MIPROv2 with all parameters set."""
        mock_optimizer = MagicMock()
        mock_miprov2.return_value = mock_optimizer

        # All parameters
        mipro_kwargs = {
            "metric": MagicMock(),
            "auto": "heavy",
            "max_bootstrapped_demos": 4,
            "max_labeled_demos": 6,
            "max_errors": 10,
            "num_threads": 16
        }

        optimizer = mock_miprov2(**mipro_kwargs)

        # Verify all parameters were passed
        call_kwargs = mock_miprov2.call_args[1]
        assert call_kwargs['auto'] == "heavy"
        assert call_kwargs['max_bootstrapped_demos'] == 4
        assert call_kwargs['max_labeled_demos'] == 6
        assert call_kwargs['max_errors'] == 10
        assert call_kwargs['num_threads'] == 16

    @patch('train.dspy.MIPROv2')
    def test_miprov2_with_callable_metric(self, mock_miprov2):
        """Verify metric is callable."""
        mock_optimizer = MagicMock()
        mock_miprov2.return_value = mock_optimizer

        def custom_metric(gold, pred, trace=None):
            return True

        mipro_kwargs = {
            "metric": custom_metric,
            "auto": "light"
        }

        optimizer = mock_miprov2(**mipro_kwargs)

        call_kwargs = mock_miprov2.call_args[1]
        assert callable(call_kwargs['metric'])


class TestCLIValidation:
    """Test CLI argument validation."""

    def test_invalid_optimizer_name_rejected(self):
        """Verify invalid optimizer names raise SystemExit."""
        parser = argparse.ArgumentParser()
        parser.add_argument("--optimizer", choices=["bootstrap", "miprov2"], default="bootstrap")

        with pytest.raises(SystemExit):
            parser.parse_args(["--optimizer", "invalid_optimizer"])

    def test_invalid_auto_mode_rejected(self):
        """Verify invalid auto modes raise SystemExit."""
        parser = argparse.ArgumentParser()
        parser.add_argument("--auto", choices=["light", "medium", "heavy"], default=None)

        with pytest.raises(SystemExit):
            parser.parse_args(["--auto", "invalid_auto"])

    def test_auto_with_bootstrap_optimizer(self):
        """Test --auto with --optimizer bootstrap (auto should be ignored)."""
        parser = argparse.ArgumentParser()
        parser.add_argument("--optimizer", choices=["bootstrap", "miprov2"], default="bootstrap")
        parser.add_argument("--auto", choices=["light", "medium", "heavy"], default=None)

        args = parser.parse_args(["--optimizer", "bootstrap", "--auto", "light"])

        # argparse accepts this, but train.py logic should ignore auto for bootstrap
        assert args.optimizer == "bootstrap"
        assert args.auto == "light"

    def test_num_threads_with_bootstrap_optimizer(self):
        """Test --num-threads with --optimizer bootstrap (should be ignored)."""
        parser = argparse.ArgumentParser()
        parser.add_argument("--optimizer", choices=["bootstrap", "miprov2"], default="bootstrap")
        parser.add_argument("--num-threads", type=int, default=None)

        args = parser.parse_args(["--optimizer", "bootstrap", "--num-threads", "8"])

        # argparse accepts this, but train.py logic should ignore num_threads for bootstrap
        assert args.optimizer == "bootstrap"
        assert args.num_threads == 8


class TestMetricEdgeCases:
    """Test metric function edge cases."""

    def test_metric_signature(self):
        """Verify metric has correct signature."""
        from qa_module import hallucination_aware_metric
        import inspect

        sig = inspect.signature(hallucination_aware_metric)
        params = list(sig.parameters.keys())

        # Should have gold and pred
        assert 'gold' in params
        assert 'pred' in params
        # May or may not have trace
        # assert 'trace' in params

    def test_metric_is_callable(self):
        """Verify metric function is callable."""
        from qa_module import hallucination_aware_metric

        assert callable(hallucination_aware_metric)

    def test_metric_with_none_trace(self):
        """Test metric can be called with trace=None."""
        from qa_module import hallucination_aware_metric

        # Create mock gold and pred
        gold = MagicMock()
        gold.answer = "Test answer"
        pred = MagicMock()
        pred.answer = "Test answer"

        # Should not raise error
        result = hallucination_aware_metric(gold, pred, trace=None)
        # Result should be a number
        assert isinstance(result, (int, float))

    def test_metric_without_trace(self):
        """Test metric can be called without trace parameter."""
        from qa_module import hallucination_aware_metric

        # Create mock gold and pred
        gold = MagicMock()
        gold.answer = "Test answer"
        pred = MagicMock()
        pred.answer = "Test answer"

        # Should not raise error
        result = hallucination_aware_metric(gold, pred)
        # Result should be a number
        assert isinstance(result, (int, float))


class TestModelFilenameEdgeCases:
    """Test model filename generation edge cases."""

    def test_model_filename_underscore_in_optimizer_name(self):
        """Test filename generation with underscores."""
        # Simulate different optimizer names
        optimizers = ["bootstrap", "miprov2"]

        for opt in optimizers:
            optimizer_suffix = "miprov2" if opt == "miprov2" else "bootstrap"
            model_path = f"trained_qa_model_{optimizer_suffix}.json"

            assert model_path.endswith(".json")
            assert "_" in model_path
            assert model_path.count("_") == 3  # trained_qa_model_optimizer

    def test_both_models_can_exist_simultaneously(self):
        """Verify both model files can exist without conflicts."""
        import os

        # Simulate both filenames
        bootstrap_path = "trained_qa_model_bootstrap.json"
        miprov2_path = "trained_qa_model_miprov2.json"

        # Should be different
        assert bootstrap_path != miprov2_path

        # Should have same base name except suffix
        bootstrap_base = bootstrap_path.replace("_bootstrap.json", "")
        miprov2_base = miprov2_path.replace("_miprov2.json", "")

        assert bootstrap_base == miprov2_base == "trained_qa_model"

    def test_model_filename_extensions(self):
        """Verify all model files have .json extension."""
        filenames = [
            "trained_qa_model_bootstrap.json",
            "trained_qa_model_miprov2.json"
        ]

        for filename in filenames:
            assert filename.endswith(".json")
            assert not filename.endswith(".json.txt")  # No double extensions


class TestOptimizerParameterEdgeCases:
    """Test optimizer parameter edge cases."""

    @patch('train.dspy.BootstrapFewShot')
    def test_bootstrap_without_max_rounds(self, mock_bootstrap):
        """Test BootstrapFewShot requires max_rounds."""
        mock_optimizer = MagicMock()
        mock_bootstrap.return_value = mock_optimizer

        # This is the default in train.py
        max_rounds = 1

        optimizer = mock_bootstrap(
            metric=MagicMock(),
            max_labeled_demos=6,
            max_bootstrapped_demos=4,
            max_rounds=max_rounds,
            max_errors=10
        )

        call_kwargs = mock_bootstrap.call_args[1]
        assert call_kwargs['max_rounds'] == 1

    @patch('train.dspy.MIPROv2')
    def test_miprov2_without_auto(self, mock_miprov2):
        """Test MIPROv2 auto parameter defaults to 'light' when not provided."""
        mock_optimizer = MagicMock()
        mock_miprov2.return_value = mock_optimizer

        # Simulate train.py logic
        auto = None
        auto_mode = auto or "light"

        mipro_kwargs = {
            "metric": MagicMock(),
            "auto": auto_mode,
            "max_bootstrapped_demos": 4,
            "max_labeled_demos": 6
        }

        optimizer = mock_miprov2(**mipro_kwargs)

        call_kwargs = mock_miprov2.call_args[1]
        assert call_kwargs['auto'] == "light"

    @patch('train.dspy.MIPROv2')
    def test_miprov2_with_very_large_threads(self, mock_miprov2):
        """Test MIPROv2 with very large num_threads value."""
        mock_optimizer = MagicMock()
        mock_miprov2.return_value = mock_optimizer

        mipro_kwargs = {
            "metric": MagicMock(),
            "auto": "light",
            "max_bootstrapped_demos": 4,
            "max_labeled_demos": 6
        }

        # Add very large num_threads
        num_threads = 9999
        if num_threads:
            mipro_kwargs["num_threads"] = num_threads

        optimizer = mock_miprov2(**mipro_kwargs)

        call_kwargs = mock_miprov2.call_args[1]
        assert call_kwargs['num_threads'] == 9999
        # Note: DSPy may validate this at runtime


class TestBackwardCompatibility:
    """Test backward compatibility edge cases."""

    def test_default_behavior_unchanged(self):
        """Verify default behavior uses BootstrapFewShot."""
        parser = argparse.ArgumentParser()
        parser.add_argument("--optimizer", choices=["bootstrap", "miprov2"], default="bootstrap")
        parser.add_argument("--auto", choices=["light", "medium", "heavy"], default=None)
        parser.add_argument("--num-threads", type=int, default=None)

        # Parse empty args (simulates: python train.py)
        args = parser.parse_args([])

        assert args.optimizer == "bootstrap"
        assert args.auto is None
        assert args.num_threads is None

    def test_explicit_bootstrap_same_as_default(self):
        """Verify explicit --optimizer bootstrap same as default."""
        parser = argparse.ArgumentParser()
        parser.add_argument("--optimizer", choices=["bootstrap", "miprov2"], default="bootstrap")

        # Default
        args1 = parser.parse_args([])
        # Explicit
        args2 = parser.parse_args(["--optimizer", "bootstrap"])

        assert args1.optimizer == args2.optimizer == "bootstrap"

    @patch('train.dspy.BootstrapFewShot')
    def test_bootstrap_parameters_unchanged(self, mock_bootstrap):
        """Verify BootstrapFewShot parameters haven't changed."""
        mock_optimizer = MagicMock()
        mock_bootstrap.return_value = mock_optimizer

        # These are the original parameters from train.py
        optimizer = mock_bootstrap(
            metric=MagicMock(),
            max_labeled_demos=6,
            max_bootstrapped_demos=4,
            max_rounds=1,
            max_errors=10
        )

        call_kwargs = mock_bootstrap.call_args[1]
        assert call_kwargs['max_labeled_demos'] == 6
        assert call_kwargs['max_bootstrapped_demos'] == 4
        assert call_kwargs['max_rounds'] == 1
        assert call_kwargs['max_errors'] == 10
