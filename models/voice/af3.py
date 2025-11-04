#!/usr/bin/env python3
"""
Audio Flamingo 3 (AF3) evaluator for voice-enabled reasoning tasks
Supports both thinking and non-thinking modes using direct model loading
Loads AF3 model once and reuses for all episodes for efficiency
"""

import os
import json
import time
import argparse
import tempfile
import uuid
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
import shutil
from tqdm import tqdm
import torch

# Import utilities
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Add AF3 path to imports
af3_path = Path(__file__).parent / "af3"
sys.path.insert(0, str(af3_path))

from utils.web_search import is_browsecomp_episode

# Import AF3 components
try:
    import llava
    from llava import conversation as clib
    from llava.media import Sound
    from peft import PeftModel
    from huggingface_hub import snapshot_download
except ImportError as e:
    print(f"Warning: AF3 dependencies not available: {e}")
    print("Please ensure AF3 is properly installed in models/voice/af3/")
    llava = None

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

@dataclass
class AF3EvaluationConfig:
    """Configuration for Audio Flamingo 3 evaluation"""
    model_base: str = "nvidia/audio-flamingo-3"
    conv_mode: str = "auto"
    use_thinking_mode: bool = False  # Enable thinking mode by default
    thinking_suffix: str = "Please think and reason about the input audio before you respond."
    device_map: str = "auto"  # Device mapping for model
    torch_dtype: str = "float16"  # PyTorch dtype for model
    max_retries: int = 3  # Maximum retries for failed calls
    output_sample_rate: int = 24000  # For consistency with other models

class AF3AudioFlamingoEvaluator:
    """Audio Flamingo 3 evaluator using direct model loading"""

    def __init__(self, config: AF3EvaluationConfig, output_dir: str = None):
        self.config = config
        self.output_dir = output_dir or "voice_output"
        self.model = None

        print(f"Initializing Audio Flamingo 3 (AF3) evaluator...")
        print(f"  Model: {config.model_base}")
        print(f"  Conv mode: {config.conv_mode}")
        print(f"  Thinking mode: {config.use_thinking_mode}")
        print(f"  Device map: {config.device_map}")
        print(f"  Output directory: {self.output_dir}")

        if llava is None:
            raise ImportError("AF3 dependencies not available. Please install AF3 properly.")

        self._load_model()

    def _load_model(self):
        """Load AF3 model once for reuse"""
        print("Loading AF3 model...")
        try:
            # Download model
            model_path = snapshot_download(self.config.model_base)
            model_think = os.path.join(model_path, 'stage35')

            # Load base model
            self.model = llava.load(model_path)

            # Load thinking adapter if needed
            if self.config.use_thinking_mode:
                print("Loading thinking mode adapter...")
                torch_dtype = getattr(torch, self.config.torch_dtype) if hasattr(torch, self.config.torch_dtype) else torch.float16
                self.model = PeftModel.from_pretrained(
                    self.model,
                    model_think,
                    device_map=self.config.device_map,
                    torch_dtype=torch_dtype,
                )

            # Move to GPU
            self.model = self.model.to("cuda")

            # Set conversation mode
            clib.default_conversation = clib.conv_templates[self.config.conv_mode].copy()

            print("✓ AF3 model loaded successfully")

        except Exception as e:
            print(f"Error loading AF3 model: {e}")
            raise

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

    def prepare_prompt(self, episode: Dict[str, Any], turn: Dict[str, Any],
                      task_type: str, thinking_mode: bool = None) -> str:
        """Prepare prompt for AF3 with context length management"""

        text_content = turn.get('text_content', '')

        # Use instance config if not explicitly specified
        if thinking_mode is None:
            thinking_mode = self.config.use_thinking_mode

        # Handle task-specific context
        if task_type == 'mrcr':
            # MRCR: Add context documents as text (same as qwen approach)
            context_docs = episode.get('context_documents', [])
            if context_docs:
                context_text = "\n\n".join([doc.get('content', '') for doc in context_docs])
                if context_text:
                    text_content = f"Context:\n{context_text}\n\nQuestion: {text_content}"

        elif task_type == 'browsecomp':
            # BrowseComp: web search disabled; leave prompt unchanged.
            print(f"BrowseComp detected for {episode.get('id', 'unknown')}, but web search is disabled.")

        # Add thinking mode suffix if enabled
        if thinking_mode:
            text_content = f"{text_content} {self.config.thinking_suffix}"

        return text_content

    def generate_response(self, text_prompt: str, audio_file: str) -> Dict[str, Any]:
        """Generate response using loaded AF3 model"""

        if not Path(audio_file).exists():
            return {
                "success": False,
                "response": "",
                "error": f"Audio file not found: {audio_file}",
                "execution_time": 0.0
            }

        start_time = time.time()

        try:
            # Prepare multimedia prompt
            mm_prompt = []

            # Add audio
            audio_media = Sound(audio_file)
            mm_prompt.append(audio_media)

            # Add text prompt
            if not text_prompt.strip():
                text_prompt = "Please analyze this audio."

            mm_prompt.append(text_prompt)

            # Generate response using model
            response = self.model.generate_content(mm_prompt)

            execution_time = time.time() - start_time

            return {
                "success": True,
                "response": response,
                "error": None,
                "execution_time": execution_time
            }

        except Exception as e:
            execution_time = time.time() - start_time
            return {
                "success": False,
                "response": "",
                "error": f"Error generating AF3 response: {str(e)}",
                "execution_time": execution_time
            }

    def process_episode(self, episode: Dict[str, Any], task_type: str) -> Dict[str, Any]:
        """Process a single episode using AF3"""
        episode_id = episode.get("id", "unknown")
        turns = episode.get("turns", [])

        if not turns:
            return None

        turn = turns[0]
        audio_file = turn.get('audio_file', '')

        if not audio_file or not Path(audio_file).exists():
            return {
                "episode_id": episode_id,
                "task_type": task_type,
                "response": "",
                "audio_response_path": None,
                "expected_answer": turn.get("metadata", {}).get("expected_answer", ""),
                "contains_answer": False,
                "error": f"Audio file not found or missing: {audio_file}",
                "has_audio_input": False,
                "has_audio_output": False,
                "token_usage": {"completion_tokens": 0, "total_tokens": 0}
            }

        try:
            # Prepare prompt (simplified approach)
            text_prompt = self.prepare_prompt(episode, turn, task_type, self.config.use_thinking_mode)

            # Track timing
            request_start_time = time.time()

            # Generate response with retries
            af3_result = None
            last_error = None

            for attempt in range(self.config.max_retries):
                af3_result = self.generate_response(text_prompt, audio_file)
                if af3_result["success"]:
                    break
                last_error = af3_result["error"]
                if attempt < self.config.max_retries - 1:
                    print(f"Retry {attempt + 1}/{self.config.max_retries} for episode {episode_id}")
                    time.sleep(1)  # Brief delay before retry

            if not af3_result or not af3_result["success"]:
                raise Exception(last_error or "AF3 processing failed")

            text_response = af3_result["response"]
            first_token_time = af3_result["execution_time"]
            request_end_time = time.time()
            total_time = request_end_time - request_start_time

            # Check if expected answer is in response
            expected = turn.get("metadata", {}).get("expected_answer", "")
            contains_answer = expected.lower() in text_response.lower() if expected else None

            # Create episode-specific directory (matching azure_gpt_realtime structure)
            episode_dir = Path(self.output_dir) / f"vera_{task_type}_{episode_id}"
            episode_dir.mkdir(parents=True, exist_ok=True)

            # Save conversation.txt
            conversation_path = episode_dir / "conversation.txt"
            with open(conversation_path, 'w') as f:
                f.write(f"User: {turn.get('text_content', '')}\n")
                f.write(f"Assistant: {text_response}\n")

            # Save input audio if available
            if Path(audio_file).exists():
                input_audio_path = episode_dir / "input.wav"
                shutil.copy2(audio_file, input_audio_path)

            # AF3 doesn't generate audio output, only text
            audio_output_path = None

            # Save response.json
            response_json_path = episode_dir / "response.json"
            response_data = {
                "episode_id": episode_id,
                "task_type": task_type,
                "text_response": text_response,
                "has_audio_output": False,
                "audio_output_path": None,
                "expected_answer": expected,
                "contains_answer": contains_answer,
                "thinking_mode": self.config.use_thinking_mode,
                "timing": {
                    "total_time": total_time,
                    "first_token_time": first_token_time,
                    "af3_execution_time": af3_result["execution_time"]
                },
                "model": "audio-flamingo-3"
            }
            with open(response_json_path, 'w') as f:
                json.dump(response_data, f, indent=2)

            # Save execution.log
            execution_log_path = episode_dir / "execution.log"
            with open(execution_log_path, 'w') as f:
                f.write(f"Episode: {episode_id}\n")
                f.write(f"Task Type: {task_type}\n")
                f.write(f"Processing Time: {total_time:.2f}s\n")
                f.write(f"AF3 Execution Time: {af3_result['execution_time']:.2f}s\n")
                f.write(f"Thinking Mode: {self.config.use_thinking_mode}\n")
                f.write(f"Has Audio Input: True\n")
                f.write(f"Has Audio Output: False\n")
                f.write(f"Model: audio-flamingo-3\n")

            return {
                "episode_id": episode_id,
                "task_type": task_type,
                "response": text_response,
                "audio_response_path": None,
                "expected_answer": expected,
                "contains_answer": contains_answer,
                "audio_file": audio_file,
                "original_text": turn.get("text_content", ""),
                "has_audio_input": True,
                "has_audio_output": False,
                "thinking_mode": self.config.use_thinking_mode,
                "timing": {
                    "request_start_time": request_start_time,
                    "request_end_time": request_end_time,
                    "first_token_time": first_token_time,
                    "total_time": total_time,
                    "af3_execution_time": af3_result["execution_time"]
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
                "has_audio_input": bool(audio_file and Path(audio_file).exists()),
                "has_audio_output": False,
                "thinking_mode": self.config.use_thinking_mode,
                "token_usage": {
                    "completion_tokens": 0,
                    "total_tokens": 0
                }
            }

    def evaluate_dataset(self, dataset_path: str, output_base_dir: str, max_episodes: Optional[int] = None):
        """Evaluate voice dataset with Audio Flamingo 3"""
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

        # Create output directory structure with thinking mode separation
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        # voice_output/af3_audio_flamingo[_thinking]/[dataset]/[timestamp]/
        model_dir = "af3_audio_flamingo_thinking" if self.config.use_thinking_mode else "af3_audio_flamingo"
        output_path = Path(output_base_dir) / model_dir / source / timestamp
        output_path.mkdir(parents=True, exist_ok=True)

        # Update output directory for audio files
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
        with_thinking_mode = len([r for r in all_results if r.get("thinking_mode", False)])

        # Create evaluation summary
        evaluation_summary = {
            "model": "audio-flamingo-3",
            "dataset": source,
            "dataset_path": dataset_path,
            "output_directory": str(output_path),
            "total_episodes": len(all_results),
            "processed": len(all_results),
            "successful": successful_episodes,
            "failed": failed_episodes,
            "with_audio_input": with_audio_input,
            "with_audio_output": 0,  # AF3 doesn't generate audio
            "with_thinking_mode": with_thinking_mode,
            "task_distribution": task_counts,
            "duration_seconds": elapsed_time,
            "episodes_per_second": len(all_results) / elapsed_time if elapsed_time > 0 else 0,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S.%f"),
            "config": {
                "model_base": self.config.model_base,
                "conv_mode": self.config.conv_mode,
                "device_map": self.config.device_map,
                "torch_dtype": self.config.torch_dtype,
                "use_thinking_mode": self.config.use_thinking_mode,
                "max_retries": self.config.max_retries
            }
        }

        # Save results (matching azure_gpt_realtime format)
        summary_file = output_path / "benchmark_report.json"
        results_file = output_path / f"af3_audio_flamingo_results_{int(time.time())}.json"

        with open(summary_file, 'w') as f:
            json.dump(evaluation_summary, f, indent=2)

        with open(results_file, 'w') as f:
            # Convert Path objects to strings for JSON serialization
            all_results_serializable = convert_paths_to_strings(all_results)
            json.dump({
                "model_name": "audio-flamingo-3",
                "results": all_results_serializable
            }, f, indent=2)

        # Print summary
        print(f"\n{'='*60}")
        print(f"AF3 Evaluation Complete: {source}")
        print(f"{'='*60}")
        print(f"Output directory: {output_path}")
        print(f"Total episodes: {len(all_results)}")
        print(f"Successful: {successful_episodes}")
        print(f"Failed: {failed_episodes}")
        print(f"With audio input: {with_audio_input}")
        print(f"With thinking mode: {with_thinking_mode}")
        print(f"Task distribution: {task_counts}")
        print(f"Time taken: {elapsed_time:.1f}s")
        print(f"Speed: {len(all_results) / elapsed_time:.1f} episodes/sec")
        print(f"Files created:")
        print(f"  - {summary_file}")
        print(f"  - {results_file}")

        return evaluation_summary

def main():
    parser = argparse.ArgumentParser(description="Audio Flamingo 3 (AF3) evaluator")
    parser.add_argument("--model-base", type=str, default="nvidia/audio-flamingo-3",
                       help="AF3 model base name")
    parser.add_argument("--conv-mode", type=str, default="auto",
                       help="Conversation mode for AF3")
    parser.add_argument("--device-map", type=str, default="auto",
                       help="Device mapping strategy for AF3 model")
    parser.add_argument("--torch-dtype", type=str, default="float16",
                       help="PyTorch dtype for AF3 model")
    parser.add_argument("--dataset-dir", type=str, default="data/final_dataset/voice",
                       help="Directory containing voice datasets")
    parser.add_argument("--output-dir", type=str, default="voice_output",
                       help="Directory to save results")
    parser.add_argument("--specific-dataset", type=str, default=None,
                       help="Specific dataset to evaluate")
    parser.add_argument("--max-episodes", type=int, default=None,
                       help="Maximum episodes to evaluate (for testing)")
    parser.add_argument("--thinking-mode", action="store_true",
                       help="Enable AF3 thinking mode (adds thinking suffix to prompts)")
    parser.add_argument("--max-retries", type=int, default=3,
                       help="Maximum retries for failed AF3 calls")

    args = parser.parse_args()

    # Create configuration
    config = AF3EvaluationConfig(
        model_base=args.model_base,
        conv_mode=args.conv_mode,
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
        use_thinking_mode=args.thinking_mode,
        max_retries=args.max_retries
    )

    # Initialize evaluator
    print("Initializing AF3 Audio Flamingo evaluator...")
    evaluator = AF3AudioFlamingoEvaluator(config, args.output_dir)

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
    print(f"All AF3 evaluations complete!")
    print(f"Results saved to: {args.output_dir}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
