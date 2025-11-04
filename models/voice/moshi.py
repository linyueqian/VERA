#!/usr/bin/env python3
"""
Moshi Local Direct Inference evaluator
Uses the local Moshi model for direct audio-to-audio inference without server setup
Based on moshi/run_inference.py for direct model loading and inference
"""

import os
import json
import time
import argparse
import tempfile
import uuid
import asyncio
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
import numpy as np
import soundfile as sf
import torch
import random
from collections import deque
from tqdm import tqdm

# Import utilities
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from utils.web_search import is_browsecomp_episode

def convert_paths_to_strings(obj):
    """Recursively convert all Path objects to strings in a data structure."""
    if isinstance(obj, Path):
        return str(obj)
    elif isinstance(obj, dict):
        return {key: convert_paths_to_strings(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_paths_to_strings(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_paths_to_strings(item) for item in obj)
    else:
        return obj

# Add Moshi package root to path (so imports like moshi.* work)
moshi_pkg_root = Path(__file__).parent / "moshi" / "moshi"
sys.path.insert(0, str(moshi_pkg_root))

try:
    import sentencepiece
    import sphn

    # Import Moshi modules using package path so relative imports resolve
    from moshi.client_utils import AnyPrinter, Printer, RawPrinter, log
    from moshi.conditioners import ConditionAttributes, ConditionTensors
    from moshi.models import LMGen, LMModel, MimiModel
    from moshi.models import loaders

except ImportError as e:
    print(f"Error importing Moshi modules: {e}")
    print("Make sure the Moshi model is properly installed and the path is correct")
    print(f"Expected moshi pkg root: {moshi_pkg_root}")
    print(f"Path exists: {moshi_pkg_root.exists()}")
    sys.exit(1)

def seed_all(seed):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False

def get_condition_tensors(
    model_type: str, lm: LMModel, batch_size: int, cfg_coef: float
) -> ConditionTensors:
    """Get condition tensors for the model"""
    condition_tensors = {}
    if lm.condition_provider is not None and lm.condition_provider.conditioners:
        conditions: list[ConditionAttributes] | None = None
        if model_type == "hibiki":
            conditions = [
                ConditionAttributes(text={"description": "very_good"}, tensor={})
                for _ in range(batch_size)
            ]
            if cfg_coef != 1.0:
                conditions += [
                    ConditionAttributes(text={"description": "very_bad"}, tensor={})
                    for _ in range(batch_size)
                ]
        else:
            raise RuntimeError(
                f"Model expects conditioning but model type {model_type} is not supported."
            )
        assert conditions is not None
        return lm.condition_provider.prepare_and_provide(conditions)
    return condition_tensors

@dataclass
class EvaluationConfig:
    """Configuration for Moshi local evaluation"""
    hf_repo: str = None  # Use default repo
    moshi_weight: str = None
    mimi_weight: str = None
    tokenizer: str = None
    config: str = None
    device: str = "cuda"
    dtype: str = "bfloat16"  # "bfloat16" or "float16"
    batch_size: int = 1
    cfg_coef: float = 1.0
    audio_sample_rate: int = 24000
    output_sample_rate: int = 24000
    seed: int = 4242
    continue_until_text_eos: bool = True
    max_steps: Optional[int] = None

class MoshiInferenceState:
    """Moshi inference state wrapper"""

    def __init__(
        self,
        checkpoint_info,
        mimi: MimiModel,
        text_tokenizer: sentencepiece.SentencePieceProcessor,
        lm: LMModel,
        batch_size: int,
        cfg_coef: float,
        device: str | torch.device,
        continue_until_text_eos: bool = True,
        max_steps: Optional[int] = None,
        **kwargs,
    ):
        self.checkpoint_info = checkpoint_info
        model_type = checkpoint_info.model_type
        self.model_type = model_type
        self.mimi = mimi
        self.text_tokenizer = text_tokenizer
        condition_tensors = get_condition_tensors(model_type, lm, batch_size, cfg_coef)
        self.lm_gen = LMGen(
            lm, cfg_coef=cfg_coef, condition_tensors=condition_tensors, **kwargs
        )
        self.device = device
        self.frame_size = int(self.mimi.sample_rate / self.mimi.frame_rate)
        self.batch_size = batch_size
        self.continue_until_text_eos = continue_until_text_eos
        self.max_steps = max_steps
        self.mimi.streaming_forever(batch_size)
        self.lm_gen.streaming_forever(batch_size)
        self.printer: AnyPrinter
        if sys.stdout.isatty():
            self.printer = Printer()
        else:
            self.printer = RawPrinter()

    # Allow use as a context manager so we always stop streaming
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            if hasattr(self.mimi, '_stop_streaming'):
                self.mimi._stop_streaming()
            if hasattr(self.lm_gen, '_stop_streaming'):
                self.lm_gen._stop_streaming()
        except Exception:
            pass
        return False

    def run(self, in_pcms: torch.Tensor) -> tuple[str, Optional[torch.Tensor], dict]:
        """Run inference and return (text_response, audio_output, metrics)."""
        out_pcms_per_item: list[list[torch.Tensor]] = [
            [] for _ in range(self.batch_size)
        ]
        out_text_tokens_per_item: list[list[torch.Tensor]] = [
            [] for _ in range(self.batch_size)
        ]
        text_response_parts = []

        eos_reached: list[bool] = [False] * self.batch_size
        need_eos_input: bool = True

        device = self.lm_gen.lm_model.device
        start_time = time.time()
        ntokens = 0
        first_frame = True
        ttft_audio = None
        ttft_text = None

        if self.model_type == "stt":
            stt_config = self.checkpoint_info.stt_config
            pad_right = stt_config.get("audio_delay_seconds", 0.0)
            pad_left = stt_config.get("audio_silence_prefix_seconds", 0.0)
            pad_left = int(pad_left * 24000)
            pad_right = int((pad_right + 1.0) * 24000)
            in_pcms = torch.nn.functional.pad(in_pcms, (pad_left, pad_right), mode="constant")

        chunks = deque(
            [
                chunk
                for chunk in in_pcms.split(self.frame_size, dim=2)
                if chunk.shape[-1] == self.frame_size
            ]
        )

        while not all(eos_reached):
            if self.max_steps is not None and ntokens >= self.max_steps:
                break
            if chunks:
                chunk = chunks.popleft()
                codes = self.mimi.encode(chunk)
            else:
                if self.model_type == "hibiki" or self.continue_until_text_eos:
                    if need_eos_input:
                        need_eos_input = False
                        eos_value = self.mimi.cardinality
                        codes = torch.full(
                            (self.batch_size, self.mimi.num_codebooks, 1),
                            eos_value,
                            device=device,
                            dtype=torch.long,
                        )
                    else:
                        silence = torch.zeros(
                            (self.batch_size, self.mimi.channels, self.frame_size),
                            device=device,
                        )
                        codes = self.mimi.encode(silence)
                else:
                    break

            if first_frame:
                tokens = self.lm_gen.step(codes)
                if max(self.lm_gen.lm_model.delays) > 0:
                    assert tokens is None
                first_frame = False

            tokens = self.lm_gen.step(codes)
            if tokens is None:
                continue

            assert tokens.shape[1] == self.lm_gen.lm_model.dep_q + 1

            if self.lm_gen.lm_model.dep_q > 0:
                out_pcm = self.mimi.decode(tokens[:, 1:]).cpu()
                for b, (one_text, one_pcm) in enumerate(
                    zip(tokens[:, 0].cpu(), out_pcm)
                ):
                    if eos_reached[b]:
                        continue
                    elif one_text.item() == self.text_tokenizer.eos_id():
                        if need_eos_input:
                            self.printer.log("warning", "EOS sampled too early.")
                        else:
                            eos_reached[b] = True

                    out_text_tokens_per_item[b].append(one_text)
                    out_pcms_per_item[b].append(one_pcm)

                    if b == 0:
                        if one_text.item() not in [0, 3]:
                            text = self.text_tokenizer.id_to_piece(one_text.item())
                            text = text.replace("▁", " ")
                            text_response_parts.append(text)
                            if ttft_text is None:
                                ttft_text = time.time() - start_time
                        if ttft_audio is None:
                            ttft_audio = time.time() - start_time
            else:
                one_text = tokens[0, 0].cpu()
                if one_text.item() not in [0, 3]:
                    text = self.text_tokenizer.id_to_piece(one_text.item())
                    text = text.replace("▁", " ")
                    text_response_parts.append(text)
                    if ttft_text is None:
                        ttft_text = time.time() - start_time
            ntokens += 1

        # Combine text response
        full_text_response = "".join(text_response_parts).strip()

        # Combine audio output
        if self.lm_gen.lm_model.dep_q > 0 and out_pcms_per_item[0]:
            full_audio = torch.cat(out_pcms_per_item[0], dim=1)
            metrics = {"ttft_audio": ttft_audio, "ttft_text": ttft_text, "steps": ntokens}
            return full_text_response, full_audio, metrics
        else:
            metrics = {"ttft_audio": None, "ttft_text": ttft_text, "steps": ntokens}
            return full_text_response, None, metrics

class MoshiLocalEvaluator:
    """Moshi local direct inference evaluator"""

    def __init__(self, config: EvaluationConfig, output_dir: str = None):
        self.config = config
        self.output_dir = output_dir or "voice_output"

        print(f"Initializing Moshi local evaluator...")
        print(f"  Device: {config.device}")
        print(f"  Dtype: {config.dtype}")
        print(f"  Batch size: {config.batch_size}")
        print(f"  Output directory: {self.output_dir}")

        # Set random seed
        seed_all(config.seed)

        # Get dtype
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "auto": torch.bfloat16
        }
        self.dtype = dtype_map.get(config.dtype, torch.bfloat16)

        # Load model components
        print("Loading checkpoint info...")
        self.checkpoint_info = loaders.CheckpointInfo.from_hf_repo(
            config.hf_repo or loaders.DEFAULT_REPO, config.moshi_weight, config.mimi_weight,
            config.tokenizer, config.config
        )

        print("Loading Mimi model...")
        self.mimi = self.checkpoint_info.get_mimi(device=config.device)

        print("Loading text tokenizer...")
        self.text_tokenizer = self.checkpoint_info.get_text_tokenizer()

        print("Loading Moshi model...")
        self.lm = self.checkpoint_info.get_moshi(device=config.device, dtype=self.dtype)

        if self.lm.dep_q == 0:
            self.config.batch_size = 1

        print("✓ Moshi models loaded successfully")

        # Initialize a single streaming inference state to reuse across episodes
        print("Initializing streaming state once for all episodes...")
        self.state = MoshiInferenceState(
            self.checkpoint_info,
            self.mimi,
            self.text_tokenizer,
            self.lm,
            self.config.batch_size,
            self.config.cfg_coef,
            self.config.device,
            continue_until_text_eos=self.config.continue_until_text_eos,
            max_steps=self.config.max_steps,
            **self.checkpoint_info.lm_gen_config,
        )
        print("✓ Streaming state ready")

    def load_audio(self, audio_path: str) -> Optional[torch.Tensor]:
        """Load and preprocess audio file"""
        try:
            if not Path(audio_path).exists():
                print(f"Warning: Audio file not found: {audio_path}")
                return None

            # Load audio with sphn at the model's sample rate
            in_pcms, _ = sphn.read(audio_path, sample_rate=self.mimi.sample_rate)
            in_pcms = torch.from_numpy(in_pcms).to(device=self.config.device)
            # Expand for batch processing
            in_pcms = in_pcms[None, 0:1].expand(self.config.batch_size, -1, -1)
            return in_pcms
        except Exception as e:
            print(f"Warning: Could not load audio file {audio_path}: {e}")
            return None

    def detect_task_type(self, episode: Dict[str, Any]) -> str:
        """Detect task type based on episode characteristics"""
        episode_id = episode.get('id', '').lower()
        track = episode.get('track', '').lower()
        context_docs = episode.get('context_documents', [])

        if 'mrcr' in episode_id or track == 'long_context' or context_docs:
            return 'mrcr'
        elif is_browsecomp_episode(episode):
            return 'browsecomp'
        else:
            return 'standard'

    def process_episode(self, episode: Dict[str, Any], task_type: str) -> Dict[str, Any]:
        """Process a single episode using direct Moshi inference"""
        episode_id = episode.get("id", "unknown")
        turns = episode.get("turns", [])

        if not turns:
            return None

        turn = turns[0]
        text_content = turn.get('text_content', '')
        audio_file = turn.get('audio_file', '')

        # Load audio if available
        audio_data = None
        if audio_file:
            audio_data = self.load_audio(audio_file)

        if audio_data is None:
            print(f"Warning: No valid audio found for episode {episode_id}")
            return {
                "episode_id": episode_id,
                "task_type": task_type,
                "response": "",
                "audio_response_path": None,
                "expected_answer": turn.get("metadata", {}).get("expected_answer", ""),
                "contains_answer": False,
                "error": "No valid audio input",
                "has_audio_input": False,
                "has_audio_output": False,
                "token_usage": {"completion_tokens": 0, "total_tokens": 0}
            }

        try:
            # Create episode-specific directory
            episode_dir = Path(self.output_dir) / f"vera_{task_type}_{episode_id}"
            episode_dir.mkdir(parents=True, exist_ok=True)

            # Track timing
            request_start_time = time.time()

            # Reset the persistent streaming state for this episode
            try:
                self.state.mimi.reset_streaming()
            except Exception:
                pass
            try:
                self.state.lm_gen.reset_streaming()
            except Exception:
                pass
            # Apply a safety cap when continuing past audio end on non-hibiki models
            local_max_steps = self.config.max_steps
            if self.config.continue_until_text_eos and local_max_steps is None and self.checkpoint_info.model_type != "hibiki":
                local_max_steps = 400
            self.state.max_steps = local_max_steps

            # Run inference
            with torch.no_grad():
                text_response, audio_output, metrics = self.state.run(audio_data)

            request_end_time = time.time()
            total_time = request_end_time - request_start_time
            # Prefer TTFT as time to first audio chunk; fallback to first text token
            first_token_time = metrics.get("ttft_audio") or metrics.get("ttft_text")

            # Check if expected answer is in response
            expected = turn.get("metadata", {}).get("expected_answer", "")
            contains_answer = expected.lower() in text_response.lower() if expected and text_response else None

            # Save conversation.txt
            conversation_path = episode_dir / "conversation.txt"
            with open(conversation_path, 'w') as f:
                f.write(f"User: {text_content}\n")
                f.write(f"Assistant: {text_response}\n")

            # Save input audio if available
            if audio_file and Path(audio_file).exists():
                input_audio_path = episode_dir / "input.wav"
                import shutil
                shutil.copy2(audio_file, input_audio_path)

            # Save audio output if generated
            audio_output_path = None
            if audio_output is not None:
                audio_output_path = episode_dir / "output.wav"
                audio_numpy = audio_output[0].detach().cpu().numpy()  # (channels, time)
                # soundfile expects (time,) or (time, channels)
                if audio_numpy.ndim == 2:
                    audio_numpy = audio_numpy.T  # (time, channels)
                sf.write(str(audio_output_path), audio_numpy, samplerate=self.mimi.sample_rate)
                print(f"✓ Audio response saved: {audio_output_path}")

            # Save response.json
            response_json_path = episode_dir / "response.json"
            response_data = {
                "episode_id": episode_id,
                "task_type": task_type,
                "text_response": text_response,
                "has_audio_output": audio_output_path is not None,
                "audio_output_path": str(audio_output_path) if audio_output_path else None,
                "expected_answer": expected,
                "contains_answer": contains_answer,
                "timing": {
                    "total_time": total_time,
                    "first_token_time": first_token_time,
                    "ttft_audio": metrics.get("ttft_audio"),
                    "ttft_text": metrics.get("ttft_text"),
                    "steps": metrics.get("steps"),
                },
                "model": "moshi-local"
            }
            with open(response_json_path, 'w') as f:
                json.dump(response_data, f, indent=2)

            # Save execution.log
            execution_log_path = episode_dir / "execution.log"
            with open(execution_log_path, 'w') as f:
                f.write(f"Episode: {episode_id}\n")
                f.write(f"Task Type: {task_type}\n")
                f.write(f"Processing Time: {total_time:.2f}s\n")
                f.write(f"First Token Time: {first_token_time:.2f}s\n")
                f.write(f"Has Audio Input: {audio_data is not None}\n")
                f.write(f"Has Audio Output: {audio_output_path is not None}\n")
                f.write(f"Model: moshi-local\n")

            print(f"✓ Processed episode {episode_id}")

            return {
                "episode_id": episode_id,
                "task_type": task_type,
                "response": text_response,
                "audio_response_path": audio_output_path,
                "expected_answer": expected,
                "contains_answer": contains_answer,
                "audio_file": audio_file,
                "original_text": text_content,
                "has_audio_input": audio_data is not None,
                "has_audio_output": audio_output_path is not None,
                "timing": {
                    "request_start_time": request_start_time,
                    "request_end_time": request_end_time,
                    "first_token_time": first_token_time,
                    "total_time": total_time,
                    "ttft_audio": metrics.get("ttft_audio"),
                    "ttft_text": metrics.get("ttft_text"),
                    "steps": metrics.get("steps"),
                },
                "token_usage": {
                    "completion_tokens": len(text_response.split()) if text_response else 0,
                    "total_tokens": len(text_response.split()) if text_response else 0
                }
            }

        except Exception as e:
            print(f"Error processing episode {episode_id}: {e}")
            import traceback
            traceback.print_exc()
            return {
                "episode_id": episode_id,
                "task_type": task_type,
                "response": "",
                "audio_response_path": None,
                "expected_answer": turn.get("metadata", {}).get("expected_answer", ""),
                "contains_answer": False,
                "error": str(e),
                "has_audio_input": audio_data is not None,
                "has_audio_output": False,
                "token_usage": {"completion_tokens": 0, "total_tokens": 0}
            }

    def evaluate_dataset(self, dataset_path: str, output_base_dir: str, max_episodes: Optional[int] = None):
        """Evaluate voice dataset with local Moshi inference"""
        # Load dataset
        print(f"\nLoading dataset: {dataset_path}")
        with open(dataset_path, 'r') as f:
            data = json.load(f)

        episodes = data.get("episodes", [])
        track = data.get("track", "unknown")
        source = data.get("source", "unknown")

        # Limit episodes if specified
        if max_episodes:
            episodes = episodes[:max_episodes]

        print(f"Processing {len(episodes)} episodes from {source} ({track})")

        # Analyze task types
        task_counts = {"standard": 0, "mrcr": 0, "browsecomp": 0}
        for episode in episodes:
            task_type = self.detect_task_type(episode)
            task_counts[task_type] += 1

        print(f"Task distribution: {task_counts}")

        # Create output directory structure
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = Path(output_base_dir) / "moshi_local" / source / timestamp
        output_path.mkdir(parents=True, exist_ok=True)

        # Update output directory
        self.output_dir = str(output_path)

        # Process episodes
        all_results = []
        start_time = time.time()

        for episode in tqdm(episodes, desc="Processing episodes"):
            task_type = self.detect_task_type(episode)
            result = self.process_single_episode(episode, task_type)
            if result:
                all_results.append(result)

        elapsed_time = time.time() - start_time

        # Calculate statistics
        successful_episodes = len([r for r in all_results if r.get("error") is None])
        failed_episodes = len(all_results) - successful_episodes
        with_audio_input = len([r for r in all_results if r.get("has_audio_input", False)])
        with_audio_output = len([r for r in all_results if r.get("has_audio_output", False)])

        # Create evaluation summary
        evaluation_summary = {
            "model": "moshi-local",
            "dataset": source,
            "dataset_path": dataset_path,
            "output_directory": str(output_path),
            "total_episodes": len(all_results),
            "processed": len(all_results),
            "successful": successful_episodes,
            "failed": failed_episodes,
            "with_audio_input": with_audio_input,
            "with_audio_output": with_audio_output,
            "task_distribution": task_counts,
            "duration_seconds": elapsed_time,
            "episodes_per_second": len(all_results) / elapsed_time if elapsed_time > 0 else 0,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S.%f"),
            "config": {
                "device": self.config.device,
                "dtype": self.config.dtype,
                "batch_size": self.config.batch_size,
                "cfg_coef": self.config.cfg_coef,
                "hf_repo": self.config.hf_repo
            }
        }

        # Save results
        summary_file = output_path / "benchmark_report.json"
        results_file = output_path / f"moshi_local_results_{int(time.time())}.json"

        with open(summary_file, 'w') as f:
            json.dump(evaluation_summary, f, indent=2)

        with open(results_file, 'w') as f:
            all_results_serializable = convert_paths_to_strings(all_results)
            json.dump({
                "model_name": "moshi-local",
                "results": all_results_serializable
            }, f, indent=2)

        # Print summary
        print(f"\n{'='*60}")
        print(f"Moshi Local Evaluation Complete: {source}")
        print(f"{'='*60}")
        print(f"Output directory: {output_path}")
        print(f"Total episodes: {len(all_results)}")
        print(f"Successful: {successful_episodes}")
        print(f"Failed: {failed_episodes}")
        print(f"With audio input: {with_audio_input}")
        print(f"With audio output: {with_audio_output}")
        print(f"Task distribution: {task_counts}")
        print(f"Time taken: {elapsed_time:.1f}s")
        print(f"Speed: {len(all_results) / elapsed_time:.1f} episodes/sec")
        print(f"Files created:")
        print(f"  - {summary_file}")
        print(f"  - {results_file}")

        return evaluation_summary

def main():
    parser = argparse.ArgumentParser(description="Moshi local direct inference evaluator")
    parser.add_argument("--hf-repo", type=str, default=None,
                       help="HuggingFace repository (uses default if not specified)")
    parser.add_argument("--moshi-weight", type=str, default=None,
                       help="Path to local Moshi checkpoint")
    parser.add_argument("--mimi-weight", type=str, default=None,
                       help="Path to local Mimi checkpoint")
    parser.add_argument("--tokenizer", type=str, default=None,
                       help="Path to local tokenizer")
    parser.add_argument("--config", type=str, default=None,
                       help="Path to config file")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use (cuda/cpu)")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                       choices=["float16", "bfloat16", "auto"],
                       help="Model dtype")
    parser.add_argument("--batch-size", type=int, default=1,
                       help="Batch size")
    parser.add_argument("--cfg-coef", type=float, default=1.0,
                       help="CFG coefficient")
    parser.add_argument("--continue-until-text-eos", dest="continue_until_text_eos",
                       action="store_true", default=True,
                       help="Continue generation after audio ends until text EOS (default: on)")
    parser.add_argument("--no-continue-until-text-eos", dest="continue_until_text_eos",
                       action="store_false",
                       help="Disable continuing until text EOS when audio ends")
    parser.add_argument("--max-steps", type=int, default=None,
                       help="Optional max generation steps when continuing past audio end")
    parser.add_argument("--dataset-dir", type=str, default="data/final_dataset/voice",
                       help="Directory containing voice datasets")
    parser.add_argument("--output-dir", type=str, default="voice_output",
                       help="Directory to save results")
    parser.add_argument("--specific-dataset", type=str, default=None,
                       help="Specific dataset to evaluate")
    parser.add_argument("--max-episodes", type=int, default=None,
                       help="Maximum episodes to evaluate (for testing)")
    parser.add_argument("--seed", type=int, default=4242,
                       help="Random seed")

    args = parser.parse_args()

    # Create configuration
    config = EvaluationConfig(
        hf_repo=args.hf_repo,
        moshi_weight=args.moshi_weight,
        mimi_weight=args.mimi_weight,
        tokenizer=args.tokenizer,
        config=args.config,
        device=args.device,
        dtype=args.dtype,
        batch_size=args.batch_size,
        cfg_coef=args.cfg_coef,
        seed=args.seed
        , continue_until_text_eos=args.continue_until_text_eos
        , max_steps=args.max_steps
    )

    # Initialize evaluator
    print("Initializing Moshi local evaluator...")
    evaluator = MoshiLocalEvaluator(config, args.output_dir)

    # Get dataset files
    dataset_dir = Path(args.dataset_dir)

    if args.specific_dataset:
        if args.specific_dataset.endswith('.json'):
            dataset_files = [dataset_dir / args.specific_dataset]
        else:
            dataset_files = [dataset_dir / f"{args.specific_dataset}_voice_episodes.json"]
    else:
        dataset_files = sorted(dataset_dir.glob("*_voice_episodes.json"))

    print(f"\nFound {len(dataset_files)} dataset(s) to evaluate")

    # Process each dataset
    all_results = {}
    for dataset_file in dataset_files:
        if not dataset_file.exists():
            print(f"Warning: {dataset_file} not found, skipping")
            continue

        print(f"\n{'='*60}")
        print(f"Evaluating: {dataset_file.name}")
        print(f"{'='*60}")

        try:
            results = evaluator.evaluate_dataset(
                dataset_path=str(dataset_file),
                output_base_dir=args.output_dir,
                max_episodes=args.max_episodes
            )
            all_results[dataset_file.stem] = results
        except Exception as e:
            print(f"Error evaluating {dataset_file.name}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"All Moshi local evaluations complete!")
    print(f"Results saved to: {args.output_dir}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
