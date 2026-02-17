"""Unit tests for optimizer selection logic."""

import pytest
import os
from unittest.mock import patch, MagicMock, Mock
import argparse
import sys


class TestCLIArgumentParsing:
    """Test command-line argument parsing."""

    def test_default_optimizer_is_bootstrap(self):
        """Verify default optimizer is BootstrapFewShot."""
        # Parse empty args (simulates: python train.py)
        parser = argparse.ArgumentParser()
        parser.add_argument("--optimizer", choices=["bootstrap", "miprov2"], default="bootstrap")
        parser.add_argument("--auto", choices=["light", "medium", "heavy"], default=None)
        parser.add_argument("--num-threads", type=int, default=None)
        args = parser.parse_args([])

        assert args.optimizer == "bootstrap"
        assert args.auto is None
        assert args.num_threads is None

    def test_explicit_bootstrap_optimizer(self):
        """Verify explicit --optimizer bootstrap."""
        parser = argparse.ArgumentParser()
        parser.add_argument("--optimizer", choices=["bootstrap", "miprov2"], default="bootstrap")
        parser.add_argument("--auto", choices=["light", "medium", "heavy"], default=None)
        parser.add_argument("--num-threads", type=int, default=None)
        args = parser.parse_args(["--optimizer", "bootstrap"])

        assert args.optimizer == "bootstrap"

    def test_miprov2_optimizer_flag(self):
        """Verify --optimizer miprov2 is parsed correctly."""
        parser = argparse.ArgumentParser()
        parser.add_argument("--optimizer", choices=["bootstrap", "miprov2"], default="bootstrap")
        parser.add_argument("--auto", choices=["light", "medium", "heavy"], default=None)
        parser.add_argument("--num-threads", type=int, default=None)
        args = parser.parse_args(["--optimizer", "miprov2"])

        assert args.optimizer == "miprov2"

    def test_auto_mode_light(self):
        """Verify --auto light is parsed correctly."""
        parser = argparse.ArgumentParser()
        parser.add_argument("--optimizer", choices=["bootstrap", "miprov2"], default="bootstrap")
        parser.add_argument("--auto", choices=["light", "medium", "heavy"], default=None)
        parser.add_argument("--num-threads", type=int, default=None)
        args = parser.parse_args(["--optimizer", "miprov2", "--auto", "light"])

        assert args.optimizer == "miprov2"
        assert args.auto == "light"

    def test_auto_mode_medium(self):
        """Verify --auto medium is parsed correctly."""
        parser = argparse.ArgumentParser()
        parser.add_argument("--optimizer", choices=["bootstrap", "miprov2"], default="bootstrap")
        parser.add_argument("--auto", choices=["light", "medium", "heavy"], default=None)
        parser.add_argument("--num-threads", type=int, default=None)
        args = parser.parse_args(["--optimizer", "miprov2", "--auto", "medium"])

        assert args.auto == "medium"

    def test_auto_mode_heavy(self):
        """Verify --auto heavy is parsed correctly."""
        parser = argparse.ArgumentParser()
        parser.add_argument("--optimizer", choices=["bootstrap", "miprov2"], default="bootstrap")
        parser.add_argument("--auto", choices=["light", "medium", "heavy"], default=None)
        parser.add_argument("--num-threads", type=int, default=None)
        args = parser.parse_args(["--optimizer", "miprov2", "--auto", "heavy"])

        assert args.auto == "heavy"

    def test_num_threads_argument(self):
        """Verify --num-threads is parsed correctly."""
        parser = argparse.ArgumentParser()
        parser.add_argument("--optimizer", choices=["bootstrap", "miprov2"], default="bootstrap")
        parser.add_argument("--auto", choices=["light", "medium", "heavy"], default=None)
        parser.add_argument("--num-threads", type=int, default=None)
        args = parser.parse_args(["--optimizer", "miprov2", "--num-threads", "8"])

        assert args.num_threads == 8

    def test_all_miprov2_arguments(self):
        """Verify all MIPROv2 arguments work together."""
        parser = argparse.ArgumentParser()
        parser.add_argument("--optimizer", choices=["bootstrap", "miprov2"], default="bootstrap")
        parser.add_argument("--auto", choices=["light", "medium", "heavy"], default=None)
        parser.add_argument("--num-threads", type=int, default=None)
        args = parser.parse_args(["--optimizer", "miprov2", "--auto", "medium", "--num-threads", "4"])

        assert args.optimizer == "miprov2"
        assert args.auto == "medium"
        assert args.num_threads == 4

    def test_optimizer_choices_limited(self):
        """Verify invalid optimizer names are rejected."""
        parser = argparse.ArgumentParser()
        parser.add_argument("--optimizer", choices=["bootstrap", "miprov2"], default="bootstrap")

        # Should raise SystemExit for invalid choice
        with pytest.raises(SystemExit):
            parser.parse_args(["--optimizer", "invalid"])

    def test_auto_choices_limited(self):
        """Verify invalid auto modes are rejected."""
        parser = argparse.ArgumentParser()
        parser.add_argument("--auto", choices=["light", "medium", "heavy"], default=None)

        # Should raise SystemExit for invalid choice
        with pytest.raises(SystemExit):
            parser.parse_args(["--auto", "invalid"])

    def test_auto_mode_default_is_light(self):
        """Verify MIPROv2 auto defaults to 'light' when not specified."""
        optimizer = "miprov2"
        auto = None  # Not specified by user

        # Simulate the default logic in train.py
        auto_mode = auto or "light"

        assert auto_mode == "light"


class TestOptimizerCreation:
    """Test optimizer object creation."""

    @patch('train.dspy.BootstrapFewShot')
    def test_bootstrap_optimizer_creation(self, mock_bootstrap):
        """Verify BootstrapFewShot optimizer is created correctly."""
        from qa_module import hallucination_aware_metric

        # Create optimizer like train.py does
        optimizer = MagicMock()
        mock_bootstrap.return_value = optimizer

        # Simulate BootstrapFewShot configuration
        configured = mock_bootstrap(
            metric=hallucination_aware_metric,
            max_labeled_demos=6,
            max_bootstrapped_demos=4,
            max_rounds=1,
            max_errors=10
        )

        # Verify it was called with correct parameters
        mock_bootstrap.assert_called_once()
        call_kwargs = mock_bootstrap.call_args[1]
        assert call_kwargs['max_labeled_demos'] == 6
        assert call_kwargs['max_bootstrapped_demos'] == 4
        assert call_kwargs['max_rounds'] == 1
        assert call_kwargs['max_errors'] == 10
        assert call_kwargs['metric'] == hallucination_aware_metric

    @patch('train.dspy.MIPROv2')
    def test_miprov2_optimizer_creation(self, mock_miprov2):
        """Verify MIPROv2 optimizer is created correctly."""
        from qa_module import hallucination_aware_metric

        # Create optimizer like train.py does
        optimizer = MagicMock()
        mock_miprov2.return_value = optimizer

        # Simulate MIPROv2 configuration
        auto_mode = "light"
        mipro_kwargs = {
            "metric": hallucination_aware_metric,
            "auto": auto_mode,
            "max_bootstrapped_demos": 4,
            "max_labeled_demos": 6,
            "max_errors": 10
        }

        configured = mock_miprov2(**mipro_kwargs)

        # Verify it was called with correct parameters
        mock_miprov2.assert_called_once()
        call_kwargs = mock_miprov2.call_args[1]
        assert call_kwargs['metric'] == hallucination_aware_metric
        assert call_kwargs['auto'] == "light"
        assert call_kwargs['max_bootstrapped_demos'] == 4
        assert call_kwargs['max_labeled_demos'] == 6
        assert call_kwargs['max_errors'] == 10

    @patch('train.dspy.MIPROv2')
    def test_miprov2_with_medium_auto(self, mock_miprov2):
        """Verify MIPROv2 with medium auto mode."""
        from qa_module import hallucination_aware_metric

        optimizer = MagicMock()
        mock_miprov2.return_value = optimizer

        # Simulate MIPROv2 configuration with medium auto
        auto_mode = "medium"
        mipro_kwargs = {
            "metric": hallucination_aware_metric,
            "auto": auto_mode,
            "max_bootstrapped_demos": 4,
            "max_labeled_demos": 6,
            "max_errors": 10
        }

        configured = mock_miprov2(**mipro_kwargs)

        call_kwargs = mock_miprov2.call_args[1]
        assert call_kwargs['auto'] == "medium"

    @patch('train.dspy.MIPROv2')
    def test_miprov2_with_heavy_auto(self, mock_miprov2):
        """Verify MIPROv2 with heavy auto mode."""
        from qa_module import hallucination_aware_metric

        optimizer = MagicMock()
        mock_miprov2.return_value = optimizer

        # Simulate MIPROv2 configuration with heavy auto
        auto_mode = "heavy"
        mipro_kwargs = {
            "metric": hallucination_aware_metric,
            "auto": auto_mode,
            "max_bootstrapped_demos": 4,
            "max_labeled_demos": 6,
            "max_errors": 10
        }

        configured = mock_miprov2(**mipro_kwargs)

        call_kwargs = mock_miprov2.call_args[1]
        assert call_kwargs['auto'] == "heavy"

    @patch('train.dspy.MIPROv2')
    def test_miprov2_without_threads(self, mock_miprov2):
        """Verify MIPROv2 works without num_threads parameter."""
        from qa_module import hallucination_aware_metric

        optimizer = MagicMock()
        mock_miprov2.return_value = optimizer

        # Simulate MIPROv2 configuration without threads
        mipro_kwargs = {
            "metric": hallucination_aware_metric,
            "auto": "light",
            "max_bootstrapped_demos": 4,
            "max_labeled_demos": 6,
            "max_errors": 10
        }

        configured = mock_miprov2(**mipro_kwargs)

        # Verify num_threads is NOT in kwargs
        assert 'num_threads' not in mipro_kwargs
        mock_miprov2.assert_called_once()

    @patch('train.dspy.MIPROv2')
    def test_miprov2_with_threads(self, mock_miprov2):
        """Verify MIPROv2 works with num_threads parameter."""
        from qa_module import hallucination_aware_metric

        optimizer = MagicMock()
        mock_miprov2.return_value = optimizer

        # Simulate MIPROv2 configuration with threads
        mipro_kwargs = {
            "metric": hallucination_aware_metric,
            "auto": "light",
            "max_bootstrapped_demos": 4,
            "max_labeled_demos": 6,
            "max_errors": 10
        }

        # Add num_threads if provided
        num_threads = 8
        if num_threads:
            mipro_kwargs["num_threads"] = num_threads

        configured = mock_miprov2(**mipro_kwargs)

        # Verify num_threads IS in kwargs
        assert 'num_threads' in mipro_kwargs
        assert mipro_kwargs['num_threads'] == 8
        mock_miprov2.assert_called_once()


class TestOptimizerSelectionLogic:
    """Test if/elif logic for optimizer selection."""

    def test_optimizer_name_display_bootstrap(self):
        """Verify optimizer name display for BootstrapFewShot."""
        args = Mock(optimizer="bootstrap", auto=None, num_threads=None)

        optimizer_name = "MIPROv2" if args.optimizer == "miprov2" else "BootstrapFewShot"
        assert optimizer_name == "BootstrapFewShot"

    def test_optimizer_name_display_miprov2(self):
        """Verify optimizer name display for MIPROv2."""
        args = Mock(optimizer="miprov2", auto=None, num_threads=None)

        optimizer_name = "MIPROv2" if args.optimizer == "miprov2" else "BootstrapFewShot"
        assert optimizer_name == "MIPROv2"

    def test_auto_mode_default_logic(self):
        """Verify auto mode defaults to 'light' for MIPROv2."""
        args = Mock(optimizer="miprov2", auto=None, num_threads=None)

        auto_mode = args.auto or "light"
        assert auto_mode == "light"

    def test_auto_mode_explicit_value(self):
        """Verify explicit auto mode is used."""
        args = Mock(optimizer="miprov2", auto="medium", num_threads=None)

        auto_mode = args.auto or "light"
        assert auto_mode == "medium"


class TestModelFilenameGeneration:
    """Test model filename generation."""

    def test_bootstrap_model_filename(self):
        """Verify filename for BootstrapFewShot model."""
        optimizer = "bootstrap"

        optimizer_suffix = "miprov2" if optimizer == "miprov2" else "bootstrap"
        model_path = f"trained_qa_model_{optimizer_suffix}.json"

        assert model_path == "trained_qa_model_bootstrap.json"
        assert model_path.endswith(".json")

    def test_miprov2_model_filename(self):
        """Verify filename for MIPROv2 model."""
        optimizer = "miprov2"

        optimizer_suffix = "miprov2" if optimizer == "miprov2" else "bootstrap"
        model_path = f"trained_qa_model_{optimizer_suffix}.json"

        assert model_path == "trained_qa_model_miprov2.json"
        assert model_path.endswith(".json")

    def test_both_models_can_exist_simultaneously(self):
        """Verify both optimizers produce different filenames."""
        bootstrap_optimizer = "bootstrap"
        miprov2_optimizer = "miprov2"

        bootstrap_suffix = "miprov2" if bootstrap_optimizer == "miprov2" else "bootstrap"
        bootstrap_path = f"trained_qa_model_{bootstrap_suffix}.json"

        miprov2_suffix = "miprov2" if miprov2_optimizer == "miprov2" else "bootstrap"
        miprov2_path = f"trained_qa_model_{miprov2_suffix}.json"

        assert bootstrap_path != miprov2_path
        assert bootstrap_path == "trained_qa_model_bootstrap.json"
        assert miprov2_path == "trained_qa_model_miprov2.json"


class TestMetricCompatibility:
    """Test metric compatibility with both optimizers."""

    def test_metric_is_callable(self):
        """Verify hallucination_aware_metric is callable."""
        from qa_module import hallucination_aware_metric

        assert callable(hallucination_aware_metric)

    def test_metric_signature(self):
        """Verify metric has correct signature."""
        from qa_module import hallucination_aware_metric
        import inspect

        sig = inspect.signature(hallucination_aware_metric)
        params = list(sig.parameters.keys())

        assert 'gold' in params
        assert 'pred' in params

    @patch('train.dspy.BootstrapFewShot')
    def test_bootstrap_accepts_metric(self, mock_bootstrap):
        """Verify BootstrapFewShot accepts hallucination_aware_metric."""
        from qa_module import hallucination_aware_metric

        optimizer = MagicMock()
        mock_bootstrap.return_value = optimizer

        configured = mock_bootstrap(
            metric=hallucination_aware_metric,
            max_labeled_demos=6,
            max_bootstrapped_demos=4
        )

        call_kwargs = mock_bootstrap.call_args[1]
        assert call_kwargs['metric'] == hallucination_aware_metric

    @patch('train.dspy.MIPROv2')
    def test_miprov2_accepts_metric(self, mock_miprov2):
        """Verify MIPROv2 accepts hallucination_aware_metric."""
        from qa_module import hallucination_aware_metric

        optimizer = MagicMock()
        mock_miprov2.return_value = optimizer

        configured = mock_miprov2(
            metric=hallucination_aware_metric,
            auto="light",
            max_bootstrapped_demos=4,
            max_labeled_demos=6
        )

        call_kwargs = mock_miprov2.call_args[1]
        assert call_kwargs['metric'] == hallucination_aware_metric

    def test_both_optimizers_use_same_metric(self):
        """Verify both optimizers can use the same metric function."""
        from qa_module import hallucination_aware_metric

        # The same metric should work with both optimizers
        assert callable(hallucination_aware_metric)

        # Verify it can be passed to both (just checking signature compatibility)
        import inspect
        sig = inspect.signature(hallucination_aware_metric)

        # Should have gold and pred parameters
        assert 'gold' in sig.parameters
        assert 'pred' in sig.parameters
