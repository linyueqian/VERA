# VERA Model Tests

This directory contains tests for VERA model adapters.

## Running Tests

### Quick Smoke Tests (No Dependencies)

Run the basic smoke tests without installing pytest:

```bash
python tests/test_models.py
```

This will test:
- Base adapter classes
- Text model imports (GPT-4o, Gemini, etc.)
- Voice model imports (Qwen2-Audio, Ultravox, etc.)
- Realtime model imports (GPT Realtime, Gemini Realtime, Moshi)
- Configuration utilities
- MRCR context parsing

### Full Test Suite (with pytest)

If you have pytest installed, run comprehensive tests:

```bash
# Install pytest if needed
pip install pytest

# Run all tests
pytest tests/test_models.py

# Run with verbose output
pytest tests/test_models.py -v

# Run specific test class
pytest tests/test_models.py::TestGPT4oAdapter -v
```

## Test Coverage

### Base Classes (`TestBaseAdapter`)
- ✓ ModelConfig creation
- ✓ BaseAdapter initialization

### Text Models
- ✓ GPT-4o adapter (`TestGPT4oAdapter`)
- ✓ Gemini 2.5 Pro adapter (`TestGemini25ProAdapter`)
- ✓ GPT-5 adapter (`TestGPT5Adapter`)

### Voice Models
- ✓ Qwen2-Audio adapter (`TestQwen2AudioAdapter`)
- ✓ Ultravox adapter (`TestUltravoxAdapter`)

### Realtime Models
- ✓ GPT Realtime adapter (`TestGPTRealtimeAdapter`)
- ✓ Gemini Realtime adapter (`TestGeminiRealtimeAdapter`)
- ✓ Moshi adapter (`TestMoshiAdapter`)

### Utilities
- ✓ Timing utilities (`TestTimingUtils`)

## Test Output

### Success
```
✓ All required tests passed!
```

### Skipped Tests
Some tests may be skipped if optional dependencies aren't installed:
```
⊘ Voice models skipped: No module named 'librosa'
```

This is expected and won't affect the core functionality tests.

## Adding New Tests

To add tests for a new model:

1. Import the model adapter
2. Create a test class (e.g., `TestMyNewAdapter`)
3. Add test methods starting with `test_`
4. Update the smoke tests in `run_smoke_tests()` if needed

Example:

```python
class TestMyNewAdapter:
    """Test my new adapter"""

    def test_adapter_initialization(self):
        """Test adapter can be initialized"""
        from models.mytype.mynew import MyNewAdapter

        adapter = MyNewAdapter(api_key="test-key")
        assert adapter.model_name == "my-new-model"
```

## Notes

- Tests use mocking to avoid requiring API keys or making real API calls
- Voice model tests may require additional dependencies (librosa, torch, vllm)
- Realtime model tests check module imports and basic functionality
- The test suite is designed to run quickly and not require model downloads
