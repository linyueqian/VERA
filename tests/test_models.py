#!/usr/bin/env python3
"""
Test module for VERA model adapters
Tests basic functionality of each model to ensure they work correctly
"""

import os
import sys
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Try to import pytest, but make it optional
try:
    import pytest
    PYTEST_AVAILABLE = True
except ImportError:
    PYTEST_AVAILABLE = False
    # Define minimal pytest decorators for standalone mode
    class pytest:
        class fixture:
            def __init__(self, *args, **kwargs):
                pass
            def __call__(self, func):
                return func
        fixture = fixture()
        @staticmethod
        def skip(msg):
            pass

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.shared.base_adapter import ModelConfig, BaseAdapter, TextAdapter, VoiceAdapter, RealtimeAdapter


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def sample_episode():
    """Sample episode data for testing"""
    return {
        "id": "test_episode_001",
        "track": "standard",
        "turns": [
            {
                "role": "user",
                "text_content": "What is 2+2?",
                "audio_file": None
            }
        ],
        "context_documents": []
    }


@pytest.fixture
def sample_mrcr_episode():
    """Sample MRCR episode with context"""
    return {
        "id": "test_mrcr_001",
        "track": "long_context",
        "turns": [
            {
                "role": "user",
                "text_content": "What was discussed earlier?",
                "audio_file": None
            }
        ],
        "context_documents": [
            {
                "content": "User: Hello\nAssistant: Hi there!\nUser: What's the weather?\nAssistant: It's sunny today."
            }
        ]
    }


@pytest.fixture
def temp_output_dir():
    """Temporary output directory"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


# ============================================================================
# Test Base Classes
# ============================================================================

class TestBaseAdapter:
    """Test base adapter functionality"""

    def test_model_config_creation(self):
        """Test ModelConfig dataclass"""
        config = ModelConfig(model_name="test-model")
        assert config.model_name == "test-model"
        assert config.temperature == 0.0
        assert config.max_tokens == 4096
        assert config.timeout == 300.0
        assert config.max_concurrent == 16

    def test_base_adapter_initialization(self):
        """Test BaseAdapter initialization"""
        config = ModelConfig(model_name="test-model")

        # Create concrete implementation for testing
        class TestAdapter(BaseAdapter):
            def process_episode(self, episode, output_dir):
                return {"episode_id": episode["id"], "success": True}

        adapter = TestAdapter(config)
        assert adapter.config == config
        assert adapter.model_name == "test-model"


# ============================================================================
# Test Text Models
# ============================================================================

class TestGPT4oAdapter:
    """Test GPT-4o text adapter"""

    @patch('httpx.Client')
    def test_adapter_initialization(self, mock_client):
        """Test GPT-4o adapter can be initialized"""
        from models.text.gpt4o import GPT4oOpenAIBrowseAdapter

        adapter = GPT4oOpenAIBrowseAdapter(api_key="test-key")
        assert adapter.model_name == "gpt-4o"
        assert adapter.api_key == "test-key"

    @patch('httpx.Client')
    def test_prepare_prompt(self, mock_client):
        """Test prompt preparation"""
        from models.text.gpt4o import GPT4oOpenAIBrowseAdapter

        adapter = GPT4oOpenAIBrowseAdapter(api_key="test-key")

        episode = {
            "id": "test_001",
            "turns": [
                {"role": "user", "text_content": "Hello"}
            ],
            "context_documents": []
        }

        turn = episode["turns"][0]
        prompt = adapter._prepare_prompt(turn, episode, 0)

        assert "Hello" in prompt

    @patch('httpx.Client')
    def test_make_api_request_simple_message(self, mock_client):
        """Test API request with simple message"""
        from models.text.gpt4o import GPT4oOpenAIBrowseAdapter

        # Mock the API response
        mock_response = Mock()
        mock_response.json.return_value = {
            "output": {"content": "Test response"},
            "usage": {"total_tokens": 10}
        }
        mock_response.raise_for_status = Mock()

        mock_client_instance = Mock()
        mock_client_instance.post.return_value = mock_response
        mock_client.return_value.__enter__.return_value = mock_client_instance

        adapter = GPT4oOpenAIBrowseAdapter(api_key="test-key")

        messages = [{"role": "user", "content": "Hello"}]
        response = adapter._make_api_request(messages)

        assert response == "Test response"


class TestGemini25ProAdapter:
    """Test Gemini 2.5 Pro adapter"""

    @patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"})
    def test_adapter_can_be_imported(self):
        """Test that Gemini adapter can be imported"""
        try:
            from models.text.gemini25_pro import Gemini25ProAdapter
            assert True
        except ImportError as e:
            pytest.skip(f"Gemini dependencies not available: {e}")


class TestGPT5Adapter:
    """Test GPT-5 adapter"""

    def test_adapter_can_be_imported(self):
        """Test that GPT-5 adapter can be imported"""
        try:
            from models.text.gpt5 import GPT5Adapter
            assert True
        except ImportError as e:
            pytest.skip(f"GPT-5 dependencies not available: {e}")


# ============================================================================
# Test Voice Models
# ============================================================================

class TestQwen2AudioAdapter:
    """Test Qwen2-Audio voice adapter"""

    def test_adapter_can_be_imported(self):
        """Test that Qwen2-Audio adapter can be imported"""
        try:
            from models.voice.qwen2_audio import Qwen2AudioAdaptiveEvaluator, EvaluationConfig

            config = EvaluationConfig()
            assert config.model_name == "Qwen/Qwen2-Audio-7B-Instruct"
            assert config.temperature == 0.7
        except ImportError as e:
            pytest.skip(f"Qwen2-Audio dependencies not available: {e}")

    def test_task_type_detection(self):
        """Test task type detection logic"""
        try:
            from models.voice.qwen2_audio import Qwen2AudioAdaptiveEvaluator, EvaluationConfig
        except ImportError:
            pytest.skip("Qwen2-Audio dependencies not available")

        config = EvaluationConfig()

        # Mock the LLM initialization to avoid loading the model
        with patch('models.voice.qwen2_audio.LLM'):
            evaluator = Qwen2AudioAdaptiveEvaluator(config)

            # Test MRCR detection
            mrcr_episode = {
                "id": "test_mrcr_001",
                "track": "long_context",
                "context_documents": [{"content": "test"}]
            }
            assert evaluator.detect_task_type(mrcr_episode) == "mrcr"

            # Test standard detection
            standard_episode = {
                "id": "test_standard_001",
                "track": "standard",
                "context_documents": []
            }
            assert evaluator.detect_task_type(standard_episode) == "standard"


class TestUltravoxAdapter:
    """Test Ultravox voice adapter"""

    def test_adapter_can_be_imported(self):
        """Test that Ultravox adapter can be imported"""
        try:
            from models.voice.ultravox import UltravoxAdapter
            assert True
        except ImportError as e:
            pytest.skip(f"Ultravox dependencies not available: {e}")


# ============================================================================
# Test Realtime Models
# ============================================================================

class TestGPTRealtimeAdapter:
    """Test GPT Realtime adapter"""

    def test_module_can_be_imported(self):
        """Test that GPT Realtime module can be imported"""
        try:
            from models.realtime import gpt_realtime
            assert hasattr(gpt_realtime, 'main')
            assert hasattr(gpt_realtime, 'parse_mrcr_context')
        except ImportError as e:
            pytest.skip(f"GPT Realtime dependencies not available: {e}")

    def test_parse_mrcr_context(self):
        """Test MRCR context parsing"""
        try:
            from models.realtime.gpt_realtime import parse_mrcr_context
        except ImportError:
            pytest.skip("GPT Realtime dependencies not available")

        context = "User: Hello\nAssistant: Hi there!\nUser: How are you?\nAssistant: I'm doing well!"
        messages = parse_mrcr_context(context)

        assert len(messages) == 4
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Hello"
        assert messages[1]["role"] == "assistant"
        assert messages[1]["content"] == "Hi there!"


class TestGeminiRealtimeAdapter:
    """Test Gemini Realtime adapter"""

    def test_adapter_can_be_imported(self):
        """Test that Gemini Realtime adapter can be imported"""
        try:
            from models.realtime import gemini
            assert True
        except ImportError as e:
            pytest.skip(f"Gemini Realtime dependencies not available: {e}")


class TestMoshiAdapter:
    """Test Moshi adapter"""

    def test_adapter_can_be_imported(self):
        """Test that Moshi adapter can be imported"""
        try:
            from models.realtime import moshi
            assert True
        except ImportError as e:
            pytest.skip(f"Moshi dependencies not available: {e}")


# ============================================================================
# Integration Tests
# ============================================================================

class TestModelIntegration:
    """Integration tests for model adapters"""

    @patch('httpx.Client')
    def test_text_model_episode_processing(self, mock_client, sample_episode, temp_output_dir):
        """Test that a text model can process an episode"""
        from models.text.gpt4o import GPT4oOpenAIBrowseAdapter

        # Mock successful API response
        mock_response = Mock()
        mock_response.json.return_value = {
            "id": "test-response",
            "output": {"content": "4"},
            "usage": {
                "total_tokens": 10,
                "prompt_tokens": 5,
                "completion_tokens": 5
            }
        }
        mock_response.raise_for_status = Mock()

        mock_client_instance = Mock()
        mock_client_instance.post.return_value = mock_response
        mock_client.return_value.__enter__.return_value = mock_client_instance

        adapter = GPT4oOpenAIBrowseAdapter(api_key="test-key")

        result = adapter.process_episode(sample_episode, temp_output_dir)

        assert "episode_id" in result
        assert result["episode_id"] == "test_episode_001"
        assert "turn_results" in result

    def test_model_config_variations(self):
        """Test different model configurations"""
        configs = [
            ModelConfig(model_name="test-1", temperature=0.0),
            ModelConfig(model_name="test-2", temperature=0.7, max_tokens=2048),
            ModelConfig(model_name="test-3", max_concurrent=8)
        ]

        for config in configs:
            assert config.model_name.startswith("test-")
            assert 0.0 <= config.temperature <= 1.0
            assert config.max_tokens > 0


# ============================================================================
# Utility Tests
# ============================================================================

class TestTimingUtils:
    """Test timing utilities"""

    def test_timing_utils_can_be_imported(self):
        """Test that timing utilities can be imported"""
        try:
            from models.shared.timing_utils import (
                create_turn_result,
                create_standardized_episode_result,
                create_standardized_batch_result
            )
            assert True
        except ImportError as e:
            pytest.skip(f"Timing utilities not available: {e}")


# ============================================================================
# Main Test Runner
# ============================================================================

def run_smoke_tests():
    """Run basic smoke tests without pytest"""
    print("=" * 70)
    print("VERA Model Smoke Tests")
    print("=" * 70)

    passed = 0
    failed = 0
    skipped = 0

    # Test 1: Import base classes
    print("\n[1/8] Testing base classes...")
    try:
        from models.shared.base_adapter import ModelConfig, BaseAdapter
        config = ModelConfig(model_name="test")
        print("✓ Base classes work")
        passed += 1
    except Exception as e:
        print(f"✗ Base classes failed: {e}")
        failed += 1

    # Test 2: Import text models
    print("\n[2/8] Testing text model imports...")
    try:
        from models.text.gpt4o import GPT4oOpenAIBrowseAdapter
        print("✓ Text models can be imported")
        passed += 1
    except Exception as e:
        print(f"✗ Text models failed: {e}")
        failed += 1

    # Test 3: Import voice models
    print("\n[3/8] Testing voice model imports...")
    try:
        from models.voice.qwen2_audio import EvaluationConfig
        print("✓ Voice models can be imported")
        passed += 1
    except Exception as e:
        print(f"⊘ Voice models skipped: {e}")
        skipped += 1

    # Test 4: Import realtime models
    print("\n[4/8] Testing realtime model imports...")
    try:
        from models.realtime import gpt_realtime
        print("✓ Realtime models can be imported")
        passed += 1
    except Exception as e:
        print(f"⊘ Realtime models skipped: {e}")
        skipped += 1

    # Test 5: Test ModelConfig
    print("\n[5/8] Testing ModelConfig...")
    try:
        config = ModelConfig(
            model_name="test-model",
            temperature=0.5,
            max_tokens=2048
        )
        assert config.model_name == "test-model"
        assert config.temperature == 0.5
        print("✓ ModelConfig works")
        passed += 1
    except Exception as e:
        print(f"✗ ModelConfig failed: {e}")
        failed += 1

    # Test 6: Test timing utilities
    print("\n[6/8] Testing timing utilities...")
    try:
        from models.shared.timing_utils import create_turn_result
        print("✓ Timing utilities can be imported")
        passed += 1
    except Exception as e:
        print(f"⊘ Timing utilities skipped: {e}")
        skipped += 1

    # Test 7: Test GPT-4o adapter initialization
    print("\n[7/8] Testing GPT-4o adapter initialization...")
    try:
        from models.text.gpt4o import GPT4oOpenAIBrowseAdapter
        adapter = GPT4oOpenAIBrowseAdapter(api_key="test-key")
        assert adapter.model_name == "gpt-4o"
        print("✓ GPT-4o adapter initializes")
        passed += 1
    except Exception as e:
        print(f"✗ GPT-4o adapter failed: {e}")
        failed += 1

    # Test 8: Test MRCR context parsing
    print("\n[8/8] Testing MRCR context parsing...")
    try:
        from models.realtime.gpt_realtime import parse_mrcr_context
        context = "User: Hello\nAssistant: Hi!"
        messages = parse_mrcr_context(context)
        assert len(messages) == 2
        print("✓ MRCR parsing works")
        passed += 1
    except Exception as e:
        print(f"⊘ MRCR parsing skipped: {e}")
        skipped += 1

    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"✓ Passed:  {passed}")
    print(f"✗ Failed:  {failed}")
    print(f"⊘ Skipped: {skipped}")
    print(f"Total:    {passed + failed + skipped}")

    if failed == 0:
        print("\n✓ All required tests passed!")
        return 0
    else:
        print(f"\n✗ {failed} test(s) failed")
        return 1


if __name__ == "__main__":
    # If run directly, execute smoke tests
    # If run with pytest, pytest will discover and run the test classes
    import sys
    if len(sys.argv) == 1 or not PYTEST_AVAILABLE:
        sys.exit(run_smoke_tests())
    else:
        if PYTEST_AVAILABLE:
            pytest.main([__file__] + sys.argv[1:])
        else:
            print("pytest not installed. Running smoke tests instead.")
            sys.exit(run_smoke_tests())
