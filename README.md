# VERA: Voice Evaluation of Reasoning Ability

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

A benchmark for evaluating reasoning capabilities in voice-interactive AI systems.

## Installation

```bash
# Clone the repository
git clone https://github.com/anonymous/vera.git
cd vera

# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies (handles virtual environment automatically)
uv sync
```

## Dataset

### Sample Data Structure

Check `test_voice_episodes/` for examples of the data format and audio files:

```bash
# View sample episode structure
cat test_voice_episodes/test.json
```

**Sample Audio:** Listen to what VERA episodes sound like:

<audio controls>
  <source src="test_voice_episodes/audio/vera_aime_0a923d23.wav" type="audio/wav">
  Your browser does not support the audio element.
</audio>

<audio controls>
  <source src="test_voice_episodes/audio/vera_browsecomp_9c79d2a8.wav" type="audio/wav">
  Your browser does not support the audio element.
</audio>

### Download Complete Dataset

```bash
# Download the complete VERA dataset
# Check data/download.txt for the download URL
cat data/download.txt
# Then use the URL from that file to download the dataset
```

## Quick Start

### 1. Set up API keys

```bash
cp .env.template .env
# Edit .env with your API keys
```

### 2. Run evaluation

```bash
# Evaluate voice models
uv run python evaluation/voice/batch_evaluate.py

# Evaluate text models (for comparison)
uv run python evaluation/text/batch_evaluate.py

# Evaluate realtime models
uv run python evaluation/realtime/batch_evaluate.py
```

### 3. View results

Results will be saved in the specified output directory with performance metrics and analysis.

## Citation

```bibtex
@misc{vera2025,
  title={Voice Evaluation of Reasoning Ability},
  author={Anonymous},
  year={2025},
  url={https://github.com/anonymous/vera}
}
```

## License

MIT License for code, CC-BY-4.0 for data.