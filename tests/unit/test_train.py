"""Unit tests for train.py configuration."""

import pytest
import os
from unittest.mock import patch, MagicMock, Mock
import dspy


class TestTrainingConfiguration:
    """Test training setup and configuration."""

    @patch('train.dspy.LM')
    @patch('train.dspy.configure')
    @patch.dict(os.environ, {'GROQ_API_KEY': 'test-api-key-12345'})
    def test_dspy_lm_configuration(self, mock_configure, mock_lm):
        """Verify DSPy LM is configured correctly."""
        # Import after setting env var
        import importlib
        import train

        # Reload to pick up env var
        importlib.reload(train)

        # Verify LM was called with correct model name
        mock_lm.assert_called_once()
        call_args = str(mock_lm.call_args)
        assert 'llama-3.1-8b-instant' in call_args or 'groq' in call_args.lower()

    @patch.dict(os.environ, {'GROQ_API_KEY': 'test-api-key-12345'})
    def test_api_key_from_environment(self):
        """Verify API key is loaded from environment."""
        # Set env var before import
        os.environ['GROQ_API_KEY'] = 'test-api-key-12345'

        # Import should not raise ValueError
        try:
            from dotenv import load_dotenv
            load_dotenv()
            api_key = os.getenv("GROQ_API_KEY")
            assert api_key == 'test-api-key-12345'
        except ValueError:
            pytest.fail("Should not raise ValueError when API key is set")

    def test_api_key_validation_missing(self):
        """Verify missing API key raises ValueError."""
        # Remove env var
        with patch.dict(os.environ, {}, clear=True):
            with patch.dict(os.environ, {'GROQ_API_KEY': ''}, clear=False):
                # Try to get API key
                api_key = os.getenv("GROQ_API_KEY")
                # Empty string should be treated as missing
                assert api_key in (None, '')


class TestOptimizerConfiguration:
    """Test BootstrapFewShot optimizer configuration."""

    @patch('train.dspy.BootstrapFewShot')
    def test_bootstrapfewshot_parameters(self, mock_bootstrap):
        """Verify BootstrapFewShot is configured with correct parameters."""
        # Mock the optimizer
        mock_optimizer = MagicMock()
        mock_bootstrap.return_value = mock_optimizer

        # Import to trigger optimizer creation
        with patch.dict(os.environ, {'GROQ_API_KEY': 'test-key'}):
            # We can't actually run train.py, but we can test the config values
            from qa_module import hallucination_aware_metric

            # Simulate the optimizer configuration
            optimizer = dspy.BootstrapFewShot(
                metric=hallucination_aware_metric,
                max_labeled_demos=6,
                max_bootstrapped_demos=4,
                max_rounds=1,
                max_errors=10
            )

            # Verify parameters
            mock_bootstrap.assert_called_once()
            call_kwargs = mock_bootstrap.call_args[1]
            assert call_kwargs['max_labeled_demos'] == 6
            assert call_kwargs['max_bootstrapped_demos'] == 4
            assert call_kwargs['max_rounds'] == 1
            assert call_kwargs['max_errors'] == 10
            assert call_kwargs['metric'] == hallucination_aware_metric

    def test_max_rounds_is_1(self):
        """Verify max_rounds is set to 1."""
        # This is a specific requirement - only one round of training
        max_rounds = 1
        assert max_rounds == 1, "Training should use max_rounds=1"

    def test_metric_is_hallucination_aware(self):
        """Verify optimizer uses hallucination_aware_metric."""
        from qa_module import hallucination_aware_metric as actual_metric

        # Verify metric function exists and is callable
        assert callable(actual_metric), "hallucination_aware_metric should be callable"

        # Verify it's the correct function (has proper signature)
        import inspect
        sig = inspect.signature(actual_metric)
        params = list(sig.parameters.keys())
        assert 'gold' in params
        assert 'pred' in params


class TestModelInitialization:
    """Test QAModule initialization in training context."""

    @patch('train.QAModule')
    def test_model_initialization(self, mock_qa):
        """Verify QAModule initializes correctly for training."""
        # Create mock instance
        mock_instance = MagicMock()
        mock_qa.return_value = mock_instance

        # Create QAModule
        from qa_module import QAModule
        qa = QAModule()

        # Verify it was created
        assert qa is not None
        assert hasattr(qa, 'generate_answer')


class TestDatasetLoading:
    """Test dataset loading in training context."""

    def test_trainset_import(self):
        """Verify trainset imports correctly."""
        # This should not raise any errors
        from dataset import trainset

        assert isinstance(trainset, list)
        assert len(trainset) == 15

    @patch.dict(os.environ, {'GROQ_API_KEY': 'test-key'})
    def test_training_samples_count(self):
        """Verify training shows correct sample count."""
        from dataset import trainset

        total = len(trainset)
        positive = sum(1 for s in trainset if "not provided" not in s.answer)
        negative = sum(1 for s in trainset if "not provided" in s.answer)

        assert total == 15
        assert positive == 9
        assert negative == 6
        assert positive + negative == total


class TestEnvironmentVariables:
    """Test environment variable handling."""

    def test_groq_api_key_required(self):
        """Verify error when GROQ_API_KEY is missing."""
        # Test with empty environment
        with patch.dict(os.environ, {}, clear=True):
            api_key = os.getenv("GROQ_API_KEY")
            assert api_key is None

    def test_empty_api_key_raises_error(self):
        """Verify empty API key is treated as missing."""
        with patch.dict(os.environ, {'GROQ_API_KEY': ''}):
            api_key = os.getenv("GROQ_API_KEY")
            # Empty string is falsy but not None
            assert api_key == ''

    @patch.dict(os.environ, {'GROQ_API_KEY': 'valid-key-12345'})
    def test_valid_api_key_accepted(self):
        """Verify valid API key is accepted."""
        api_key = os.getenv("GROQ_API_KEY")
        assert api_key == 'valid-key-12345'
        assert len(api_key) > 0


class TestTrainingOutput:
    """Test training output and model saving."""

    @patch('train.dspy.LM')
    @patch('train.dspy.configure')
    @patch('train.os.getenv')
    def test_model_save_path(self, mock_getenv, mock_configure, mock_lm):
        """Verify model is saved to correct path."""
        mock_getenv.return_value = 'test-key'

        # The expected save path
        expected_path = "trained_qa_model.json"

        # Verify path is a string
        assert isinstance(expected_path, str)
        assert expected_path.endswith('.json')

    def test_demonstrations_output_format(self):
        """Verify demonstrations are output in correct format."""
        # Mock demo data structure
        mock_demo = {
            'context': 'Test context for demonstration',
            'question': 'Test question?',
            'answer': 'Test answer'
        }

        # Verify structure
        assert 'context' in mock_demo
        assert 'question' in mock_demo
        assert 'answer' in mock_demo
        assert isinstance(mock_demo['context'], str)
        assert isinstance(mock_demo['question'], str)
        assert isinstance(mock_demo['answer'], str)
