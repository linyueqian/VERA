# VERA Dataset

This directory contains the VERA (Voice Evaluation of Reasoning Ability) dataset.

## Download

The complete dataset is hosted on OSF (Open Science Framework):

**Download URL:** https://osf.io/4k2t7?view_only=3fa7b16f71234b7c97f98a59c4f213e7

## Dataset Overview

The VERA dataset contains **2,931 voice-native episodes** organized into five tracks:

| Track | Episodes | Source | Description |
|-------|----------|--------|-------------|
| **Math** | 115 | AIME 2025 | Competition mathematics problems |
| **Web** | 1,107 | BrowseComp | Web browsing and research tasks |
| **Science** | 161 | GPQA Diamond | Graduate-level science questions |
| **Long-Context** | 548 | MRCR | Multi-turn reading comprehension |
| **Factual** | 1,000 | SimpleQA | Factual recall questions |

## Dataset Structure

Each episode contains:

- **`id`**: Unique identifier (e.g., `vera_aime_58789fd1`)
- **`track`**: Category (`mathematical_reasoning`, `web`, `science`, `long_context`, `factual`)
- **`turns`**: Array of conversation turns with:
  - `role`: Speaker role (`user`)
  - `text_content`: Encrypted question text (base64-encoded)
  - `audio_file`: Path to corresponding audio file
  - `prefix_text`: Optional prefix (usually null)
  - `postfix_text`: Optional postfix (usually null)
- **`context_documents`**: Additional context materials (if any)
- **`interruptions`**: Interruption events (if any)
- **`metadata`**: Contains encrypted `expected_answer`
- **`canary`**: Unique decryption key for this episode

### Example Episode Structure

```json
{
  "id": "vera_aime_58789fd1",
  "track": "mathematical_reasoning",
  "turns": [
    {
      "role": "user",
      "text_content": "ayDyHIziBKCtUXnstgrT...",
      "audio_file": "aime_voice_episodes_audio/vera_aime_58789fd1.wav",
      "prefix_text": null,
      "postfix_text": null
    }
  ],
  "context_documents": [],
  "interruptions": [],
  "metadata": {
    "expected_answer": "EnS9"
  },
  "canary": "04a8d78a8fe43328c0a9936731ed47fd"
}
```

## Encryption

To prevent LLM memorization and ensure evaluation integrity, all questions (`text_content`) and answers (`expected_answer`) are encrypted using XOR cipher with SHA256-derived keys, following the methodology used in OpenAI's BrowseComp benchmark.

### Decryption

To decrypt the questions and answers, use the following Python code:

```python
import base64
import hashlib

def derive_key(password: str, length: int) -> bytes:
    """Derive a fixed-length key from the password using SHA256."""
    hasher = hashlib.sha256()
    hasher.update(password.encode())
    key = hasher.digest()
    return key * (length // len(key)) + key[: length % len(key)]

def decrypt(ciphertext_b64: str, password: str) -> str:
    """Decrypt base64-encoded ciphertext with XOR."""
    encrypted = base64.b64decode(ciphertext_b64)
    key = derive_key(password, len(encrypted))
    decrypted = bytes(a ^ b for a, b in zip(encrypted, key))
    return decrypted.decode()

# Example usage:
import json

with open('voice_episodes.json', 'r') as f:
    data = json.load(f)

# Decrypt the first episode
episode = data['episodes'][0]
canary = episode['canary']

# Decrypt question
question = decrypt(episode['turns'][0]['text_content'], canary)
print(f"Question: {question}")

# Decrypt answer
answer = decrypt(episode['metadata']['expected_answer'], canary)
print(f"Expected Answer: {answer}")
```

## Audio Files

Audio files are organized in the following directories:
- `aime_voice_episodes_audio/` - Math problems (115 files)
- `browsecomp_voice_episodes_audio/` - Web tasks (1,107 files)
- `gpqa_diamond_voice_episodes_audio/` - Science questions (161 files)
- `mrcr_voice_episodes_audio/` - Long-context tasks (548 files)
- `simpleqa_voice_episodes_audio/` - Factual questions (1,000 files)

Each `audio_file` field in the dataset references the relative path to the corresponding audio file.

All audio is synthesized using **Boson Higgs Audio 2** for consistent, high-quality speech generation.

## Sample Data

A small sample of the dataset (with unencrypted text for easier inspection) is available in the `test_voice_episodes/` directory at the repository root:

```bash
# View sample episodes
cat test_voice_episodes/test.json

# Listen to sample audio
ls test_voice_episodes/audio/
```

## License and Attribution

The dataset follows upstream licenses:

- **SimpleQA, BrowseComp, MRCR**: MIT License
- **GPQA Diamond**: CC BY 4.0
- **Audio**: Boson Higgs Audio 2 Community License (with usage restrictions)

**Important restriction**: Do not use the audio outputs to improve any other large language model.

See [ATTRIBUTIONS.md](../ATTRIBUTIONS.md) and [NOTICE.txt](../NOTICE.txt) in the repository root for complete attribution and licensing details.

## Citation

If you use this dataset, please cite:

```bibtex
@misc{lin2025vera,
  title={Voice Evaluation of Reasoning Ability: Diagnosing the Modality-Induced Performance Gap},
  author={Lin, Yueqian and Hu, Zhengmian and Wang, Qinsi and Liu, Yudong and Zhang, Hengfan and Subramanian, Jayakumar and Vlassis, Nikos and Li, Hai Helen and Chen, Yiran},
  year={2025},
  eprint={2509.26542},
  archivePrefix={arXiv},
  primaryClass={eess.AS},
  url={https://arxiv.org/abs/2509.26542}
}
```
