#!/usr/bin/env python3
"""
Adaptive Ultravox voice model evaluator for VERA datasets.
Supports different task types with vLLM backend:
- Standard tasks (AIME, GPQA, SimpleQA)
- MRCR (Multi-turn conversations with context documents)
- BrowseComp (Web search tasks - requires web search integration)
"""

import os
import sys
import json
import time
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

import librosa
import soundfile as sf
from transformers import AutoTokenizer

# Add project root to Python path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from vllm import LLM, SamplingParams
from utils.web_search import is_browsecomp_episode

@dataclass
class EvaluationConfig:
    """Configuration for Ultravox evaluation"""
    model_name: str = "fixie-ai/ultravox-v0_3"
    tensor_parallel_size: int = 4
    gpu_memory_utilization: float = 0.9
    enforce_eager: bool = False
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 4096
    batch_size: int = 4

class UltravoxAdaptiveEvaluator:
    """Adaptive evaluator for Ultravox with task-specific capabilities"""
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        
        print(f"Initializing Adaptive Ultravox with vLLM...")
        print(f"  Model: {config.model_name}")
        print(f"  Tensor parallel: {config.tensor_parallel_size} GPUs")
        print(f"  Max model length: auto (model default)")
        print(f"  Batch size: {config.batch_size}")
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        
        # Initialize vLLM with multimodal support
        # Note: limit_mm_per_prompt might not be supported in all vLLM versions
        self.llm = LLM(
            model=config.model_name,
            tensor_parallel_size=config.tensor_parallel_size,
            gpu_memory_utilization=config.gpu_memory_utilization,
            trust_remote_code=True,
            dtype="auto",
            enforce_eager=config.enforce_eager,
            disable_log_stats=True
        )
        
        # Sampling parameters
        self.sampling_params = SamplingParams(
            temperature=config.temperature,
            top_p=config.top_p,
            max_tokens=config.max_tokens,
        )
        
        print("✓ Model loaded successfully with multimodal support")
    
    def load_audio(self, audio_path: str) -> Optional[tuple]:
        """Load and preprocess audio file for Ultravox (returns tuple with sample rate)"""
        try:
            if not os.path.exists(audio_path):
                print(f"Warning: Audio file not found: {audio_path}")
                return None
            
            # Load audio with librosa - Ultravox expects original sample rate
            audio_data, sample_rate = librosa.load(audio_path, sr=None)  # Keep original sample rate
            print(f"Loaded audio: {audio_path} (duration: {len(audio_data)/sample_rate:.2f}s, sr: {sample_rate})")
            
            # Return tuple as expected by vLLM
            return (audio_data, sample_rate)
            
        except Exception as e:
            print(f"Error loading audio {audio_path}: {e}")
            return None
    
    def parse_mrcr_context(self, context: str) -> List[Dict[str, str]]:
        """Parse MRCR context document into conversation messages (User:/Assistant:)."""
        messages: List[Dict[str, str]] = []
        role: Optional[str] = None
        buf: List[str] = []
        for line in context.splitlines():
            if line.startswith('User:'):
                if role and buf:
                    messages.append({'role': role, 'content': "\n".join(buf).strip()})
                role = 'user'
                buf = [line[len('User:'):].strip()]
            elif line.startswith('Assistant:'):
                if role and buf:
                    messages.append({'role': role, 'content': "\n".join(buf).strip()})
                role = 'assistant'
                buf = [line[len('Assistant:'):].strip()]
            else:
                buf.append(line)
        if role and buf:
            messages.append({'role': role, 'content': "\n".join(buf).strip()})
        return messages

    def create_prompt(self, episode: Dict[str, Any], task_type: str):
        """Create prompt context for Ultravox.
        - MRCR: return prior chat messages parsed from context documents
        - Others: return empty string (audio-only)
        """
        if task_type == "mrcr":
            context_docs = episode.get("context_documents", [])
            history: List[Dict[str, str]] = []
            for doc in context_docs or []:
                content = doc.get("content", "")
                if content:
                    history.extend(self.parse_mrcr_context(content))
            if history:
                return history
        return ""
    
    def detect_task_type(self, episode: Dict[str, Any]) -> str:
        """Detect task type from episode data"""
        # Check for MRCR indicators
        if episode.get("context_documents"):
            return "mrcr"
        
        # Check for BrowseComp indicators
        if is_browsecomp_episode(episode):
            return "browsecomp"
            
        # Default to standard
        return "standard"
    
    def handle_browsecomp_search(self, episode: Dict[str, Any], prompt: str) -> str:
        """No-op placeholder while BrowseComp web search is disabled."""
        _ = episode  # Kept for signature compatibility
        return f"{prompt}\n\nNote: Web search is disabled for this release."
    
    def format_prompt_for_ultravox(self, content: str = "", has_audio: bool = True, history: Optional[List[Dict[str, str]]] = None) -> str:
        """Format prompt using Ultravox chat template with optional prior chat history.
        - history: list of {'role': 'user'|'assistant', 'content': str}
        - final user message: only <|audio|> when audio is present; else content
        """
        messages: List[Dict[str, Any]] = []
        if history:
            for m in history:
                r = m.get('role', 'user')
                c = m.get('content', '')
                messages.append({'role': r, 'content': c})
        final_content = "<|audio|>" if has_audio else (content or "")
        messages.append({'role': 'user', 'content': final_content})
        
        # Use tokenizer to format properly
        try:
            formatted = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            return formatted
        except Exception as e:
            print(f"Warning: Tokenizer formatting failed: {e}")
            # Fallback to manual formatting
            # Fallback to minimal manual formatting with prior messages
            parts = ["<|begin_of_text|>"]
            for m in messages:
                parts.append(f"<|start_header_id|>{m['role']}<|end_header_id|>\n\n{m['content']}<|eot_id|>")
            parts.append("<|start_header_id|>assistant<|end_header_id|>\n\n")
            return "".join(parts)
    
    def process_batch(self, batch: List[Dict[str, Any]], dataset_source: Optional[str] = None) -> List[Dict[str, Any]]:
        """Process a batch of episodes"""
        prompts = []
        audio_data_list = []
        metadata = []
        
        # Prepare batch data
        for episode in batch:
            # Prefer common id key, with fallbacks
            episode_id = (
                episode.get("id")
                or episode.get("episode_id")
                or f"episode_{len(prompts)}"
            )
            
            try:
                # Detect task type and create appropriate prompt
                task_type = self.detect_task_type(episode)
                base_prompt = self.create_prompt(episode, task_type)
                
                # Skip BrowseComp web search; treat as standard audio-only
                
                # Load audio first to determine if we have audio
                # Many datasets store audio path inside the first turn
                audio_file = episode.get("audio_file", "")
                if not audio_file:
                    turns = episode.get("turns", [])
                    if turns and isinstance(turns, list):
                        audio_file = turns[0].get("audio_file", "") or ""
                audio_data = None
                
                if audio_file and os.path.exists(audio_file):
                    audio_data = self.load_audio(audio_file)
                else:
                    # Try alternative audio file paths using dataset source when available
                    episode_id_clean = episode_id.replace("vera_", "")
                    source_folder = dataset_source or ""

                    # Build a list of plausible filenames
                    candidate_ids = [episode_id]
                    # If id contains historical variant like 'vera_aimehistorical_XXXX', map to 'vera_aime_XXXX'
                    if episode_id.startswith("vera_aimehistorical_"):
                        candidate_ids.append("vera_aime_" + episode_id.split("vera_aimehistorical_", 1)[1])
                    # Also try without the 'vera_' prefix
                    candidate_ids.append(episode_id_clean)

                    # Build possible paths (prefer dataset source folder if provided)
                    possible_paths = []
                    if source_folder:
                        for cid in candidate_ids:
                            possible_paths.append(
                                f"data/final_dataset/voice/{source_folder}_voice_episodes_audio/{cid}.wav"
                            )
                    # Fallback to known task-based folders (mrcr/browsecomp) and common sources
                    for cid in candidate_ids:
                        possible_paths.extend([
                            f"data/final_dataset/voice/mrcr_voice_episodes_audio/{cid}.wav",
                            f"data/final_dataset/voice/browsecomp_voice_episodes_audio/{cid}.wav",
                            f"data/final_dataset/voice/aime_voice_episodes_audio/{cid}.wav",
                            f"data/final_dataset/voice/gpqa_diamond_voice_episodes_audio/{cid}.wav",
                            f"data/final_dataset/voice/simpleqa_voice_episodes_audio/{cid}.wav",
                        ])
                    
                    for path in possible_paths:
                        if os.path.exists(path):
                            audio_data = self.load_audio(path)
                            audio_file = path
                            break
                
                # Format for Ultravox with optional MRCR history and audio info
                has_audio = audio_data is not None
                history = base_prompt if isinstance(base_prompt, list) else None
                content_str = base_prompt if isinstance(base_prompt, str) else ""
                formatted_prompt = self.format_prompt_for_ultravox(content_str, has_audio, history=history)
                prompts.append(formatted_prompt)
                
                audio_data_list.append(audio_data)
                
                # Store metadata
                metadata.append({
                    "episode_id": episode_id,
                    "task_type": task_type,
                    "audio_file": audio_file,
                    "expected_answer": episode.get("expected_answer", ""),
                    "original_text": episode.get("question", episode.get("transcript", ""))
                })
                
            except Exception as e:
                print(f"Error processing episode {episode_id}: {e}")
                continue
        
        # Generate responses
        if prompts:
            try:
                # Prepare inputs for vLLM
                inputs = []
                for i, prompt in enumerate(prompts):
                    input_data = {"prompt": prompt}
                    
                    if audio_data_list[i] is not None:
                        # vLLM expects audio as list of tuples (audio_data, sample_rate)
                        input_data["multi_modal_data"] = {
                            "audio": [audio_data_list[i]]  # Wrap in list as per vLLM example
                        }
                        audio_tuple = audio_data_list[i]
                        if isinstance(audio_tuple, tuple) and len(audio_tuple) == 2:
                            audio_array, sample_rate = audio_tuple
                            print(f"Audio data: shape={audio_array.shape}, sr={sample_rate}")
                        else:
                            print(f"Audio data type: {type(audio_tuple)}")
                    
                    inputs.append(input_data)
                
                # Process each input individually to track TTFT
                results = []
                
                for i, input_data in enumerate(inputs):
                    meta = metadata[i]
                    
                    # Track timing for TTFT
                    request_start_time = time.time()
                    first_token_time = None
                    full_response = ""
                    
                    try:
                        # Debug: Print input info
                        has_audio = "multi_modal_data" in input_data
                        print(f"Processing episode {meta['episode_id']} with audio: {has_audio}")
                        if has_audio:
                            audio_list = input_data["multi_modal_data"].get("audio", [])
                            if isinstance(audio_list, list) and len(audio_list) > 0:
                                audio_tuple = audio_list[0]
                                if isinstance(audio_tuple, tuple) and len(audio_tuple) == 2:
                                    audio_array, sample_rate = audio_tuple
                                    try:
                                        shape_info = getattr(audio_array, "shape", (len(audio_array),))
                                    except Exception:
                                        shape_info = ("unknown",)
                                    print(f"  Audio: shape={shape_info}, sr={sample_rate}")
                            else:
                                print("  Audio list present but empty or invalid")
                        
                        # Use standard generation (streaming not supported)
                        outputs = self.llm.generate([input_data], self.sampling_params, use_tqdm=False)
                        output = outputs[0]
                        
                        request_end_time = time.time()
                        total_time = request_end_time - request_start_time
                        
                        # Get the complete response
                        full_response = output.outputs[0].text
                        token_count = len(output.outputs[0].token_ids)
                        
                        # TTFT not available for non-streaming generation, set to null
                        first_token_time = None
                        
                        # Check if expected answer is in response
                        expected = meta["expected_answer"]
                        contains_answer = expected.lower() in full_response.lower() if expected else None
                        
                        result = {
                            "episode_id": meta["episode_id"],
                            "task_type": meta["task_type"],
                            "response": full_response,
                            "expected_answer": expected,
                            "contains_answer": contains_answer,
                            "audio_file": meta["audio_file"],
                            "original_text": meta["original_text"],
                            "has_audio": audio_data_list[i] is not None,
                            "timing": {
                                "request_start_time": request_start_time,
                                "request_end_time": request_end_time,
                                "first_token_time": first_token_time,
                                "total_time": total_time
                            },
                            "token_usage": {
                                "prompt_tokens": len(output.prompt_token_ids) if 'output' in locals() else 0,
                                "completion_tokens": token_count,
                                "total_tokens": (len(output.prompt_token_ids) if 'output' in locals() else 0) + token_count
                            }
                        }
                        
                    except Exception as e:
                        # Handle individual input errors; normalize likely context/capacity limits
                        err_msg = str(e)
                        lowered = err_msg.lower()
                        if any(k in lowered for k in [
                            'context', 'too long', 'exceeds', 'kv', 'cache', 'oom', 'out of memory', 'max seq'
                        ]):
                            err_msg = 'context_or_capacity_limit'
                        result = {
                            "episode_id": meta["episode_id"],
                            "task_type": meta["task_type"],
                            "response": "",
                            "expected_answer": meta["expected_answer"],
                            "contains_answer": False,
                            "audio_file": meta["audio_file"],
                            "original_text": meta["original_text"],
                            "has_audio": audio_data_list[i] is not None,
                            "error": err_msg,
                            "timing": {
                                "request_start_time": request_start_time,
                                "request_end_time": time.time(),
                                "first_token_time": None,  # Not available for errors
                                "total_time": time.time() - request_start_time
                            },
                            "token_usage": {
                                "prompt_tokens": 0,
                                "completion_tokens": 0,
                                "total_tokens": 0
                            }
                        }
                    
                    results.append(result)
                
                return results
                
            except Exception as e:
                # Handle context length exceeded or other errors
                print(f"Error during batch generation: {e}")
                
                # Create zero-score results for all episodes in this batch
                results = []
                for i, meta in enumerate(metadata):
                    result = {
                        "episode_id": meta["episode_id"],
                        "task_type": meta["task_type"],
                        "response": "",
                        "expected_answer": meta["expected_answer"],
                        "contains_answer": False,
                        "audio_file": meta["audio_file"],
                        "original_text": meta["original_text"],
                        "has_audio": audio_data_list[i] is not None,
                        "error": str(e),
                        "token_usage": {
                            "prompt_tokens": 0,
                            "completion_tokens": 0,
                            "total_tokens": 0
                        }
                    }
                    results.append(result)
                
                return results
        
        return []
    
    def convert_to_text_output_format(self, result: Dict[str, Any], start_time: float) -> Dict[str, Any]:
        """Convert result to match text_output format"""
        
        response = result.get("response", "")
        episode_id = result.get("episode_id", "unknown")
        expected_answer = result.get("expected_answer", "")
        token_usage = result.get("token_usage", {})
        audio_file = result.get("audio_file", "")
        task_type = result.get("task_type", "standard")
        
        # Extract timing information from result
        timing_info = result.get("timing", {})
        session_duration = timing_info.get("total_time", time.time() - start_time)
        first_token_time = timing_info.get("first_token_time", session_duration * 0.1)  # Default fallback
        request_start = timing_info.get("request_start_time", start_time)
        request_end = timing_info.get("request_end_time", start_time + session_duration)
        
        # Create turn result in text_output format
        turn_result = {
            "turn_index": 0,
            "user_input": audio_file if audio_file else "Audio input processed",
            "model_response": response,
            "total_turn_latency_ms": session_duration * 1000,
            "api_latency_ms": session_duration * 1000,
            "processing_overhead_ms": 0.0,
            "token_usage": {
                "completion_tokens": token_usage.get("completion_tokens", 0),
                "completion_tokens_details": {
                    "accepted_prediction_tokens": 0,
                    "audio_tokens": token_usage.get("prompt_tokens", 0) if result.get("has_audio") else 0,
                    "reasoning_tokens": 0,
                    "rejected_prediction_tokens": 0
                },
                "prompt_tokens": token_usage.get("prompt_tokens", 0),
                "prompt_tokens_details": {
                    "audio_tokens": token_usage.get("prompt_tokens", 0) if result.get("has_audio") else 0,
                    "cached_tokens": 0
                },
                "total_tokens": token_usage.get("total_tokens", 0)
            },
            "timing_metadata": {
                "api_latency_ms": session_duration * 1000,  # Total processing time
                "time_to_first_token_ms": first_token_time * 1000 if first_token_time is not None else None,  # Time to first token (null if not supported)
                "request_start_timestamp": request_start,
                "request_end_timestamp": request_end,
                "failed": bool(result.get("error")),
                "timed_out": False,
                "error": bool(result.get("error"))
            },
            "model_metadata": {
                "model": "ultravox-v0_3",
                "deployment": f"ultravox-v0_3-{task_type}-vllm",
                "response_id": f"ultravox_{episode_id}_{int(time.time())}",
                "finish_reason": "stop"
            },
            "error": result.get("error")
        }
        
        # Create episode result in text_output format
        episode_result = {
            "episode_id": episode_id,
            "model_name": f"ultravox-v0_3-{task_type}-vllm",
            "task_type": task_type,
            "session_duration": session_duration,
            "total_turns_processed": 1,
            "total_tokens_used": token_usage.get("total_tokens", 0),
            "turn_results": [turn_result],
            "timing_data": {
                "total_session_time_ms": session_duration * 1000,
                "average_total_turn_latency_ms": session_duration * 1000,
                "average_api_latency_ms": session_duration * 1000,
                "average_processing_overhead_ms": 0.0,
                "total_api_time_ms": session_duration * 1000,
                "total_processing_overhead_ms": 0.0
            },
            "reasoning_data": {
                "responses": [response],
                "average_response_length": len(response)
            },
            "success": not bool(result.get("error")),
            "expected_answer": expected_answer,
            "contains_answer": result.get("contains_answer", False),
            "has_audio": result.get("has_audio", False)
        }
        
        return episode_result
    
    def evaluate_dataset(self, dataset_file: str, output_dir: str, max_episodes: Optional[int] = None) -> List[Dict[str, Any]]:
        """Evaluate a complete dataset"""
        # Load dataset
        with open(dataset_file, 'r') as f:
            data = json.load(f)
        
        episodes = data.get("episodes", [])
        if max_episodes:
            episodes = episodes[:max_episodes]
        print(f"Loaded {len(episodes)} episodes from {dataset_file}")
        
        # Determine dataset source and create output directory
        dataset_path = Path(dataset_file)
        # Prefer source from file name if it matches our folder naming, otherwise fallback to JSON 'source'
        source = dataset_path.stem.replace("_voice_episodes", "")
        if not source:
            source = data.get("source", "").replace("_converted", "")
        
        # Create output directory with timestamp
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_dir_name = f"ultravox_adaptive_{source}_converted_{timestamp}"
        output_path = Path(output_dir) / output_dir_name
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Process in batches
        all_episode_results = []
        batch_size = self.config.batch_size
        
        for i in range(0, len(episodes), batch_size):
            batch = episodes[i:i+batch_size]
            batch_start_time = time.time()
            
            print(f"Processing batch {i//batch_size + 1}/{(len(episodes)-1)//batch_size + 1} ({len(batch)} episodes)")
            
            batch_results = self.process_batch(batch, dataset_source=source)
            
            # Convert to text_output format
            for result in batch_results:
                episode_result = self.convert_to_text_output_format(result, batch_start_time)
                all_episode_results.append(episode_result)
        
        # Calculate statistics
        successful_episodes = [r for r in all_episode_results if r["success"]]
        failed_episodes = [r for r in all_episode_results if not r["success"]]
        episodes_with_audio = [r for r in all_episode_results if r["has_audio"]]
        
        # Task distribution
        task_counts = {}
        for result in all_episode_results:
            task = result["task_type"]
            task_counts[task] = task_counts.get(task, 0) + 1
        
        # Create batch results for saving
        batch_id = int(time.time() * 1000)
        batch_results = {
            "model_name": "ultravox-v0.3-adaptive-vllm",
            "processed": len(all_episode_results),
            "successful": len(successful_episodes),
            "failed": len(failed_episodes),
            "batch_duration": time.time() - time.time(),  # Will be updated
            "episodes_per_second": 0,  # Will be calculated
            "max_concurrent": 1,
            "task_distribution": task_counts,
            "results": all_episode_results
        }
        
        # Create evaluation summary
        evaluation_summary = {
            "model": "ultravox-adaptive",
            "dataset": source + "_converted",
            "dataset_path": dataset_file,
            "output_directory": str(output_path),
            "total_episodes": len(all_episode_results),
            "processed": len(all_episode_results),
            "successful": len(successful_episodes),
            "failed": len(failed_episodes),
            "with_audio": len(episodes_with_audio),
            "task_distribution": task_counts,
            "duration_seconds": time.time() - time.time(),  # Will be updated
            "episodes_per_second": 0,  # Will be calculated
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S.%f"),
            "max_concurrent": 1,
            "async_processing": False
        }
        
        # Save files
        summary_file = output_path / "evaluation_summary.json"
        batch_file = output_path / f"ultravox_adaptive_batch_{batch_id}.json"
        
        # Ensure directory still exists (in case of race conditions)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"Saving results to: {output_path}")
        print(f"Directory exists: {output_path.exists()}")
        
        with open(summary_file, 'w') as f:
            json.dump(evaluation_summary, f, indent=2)
            
        with open(batch_file, 'w') as f:
            json.dump(batch_results, f, indent=2)
        
        print(f"✓ Evaluation completed!")
        print(f"  Total episodes: {len(all_episode_results)}")
        print(f"  Successful: {len(successful_episodes)}")
        print(f"  Failed: {len(failed_episodes)}")
        print(f"  With audio: {len(episodes_with_audio)}")
        print(f"  Results saved to: {output_path}")
        
        return all_episode_results

def main():
    parser = argparse.ArgumentParser(description="Adaptive Ultravox evaluator for VERA voice datasets")
    parser.add_argument("--dataset-dir", type=Path, required=True,
                       help="Directory containing voice datasets")
    parser.add_argument("--output-dir", type=Path, required=True,
                       help="Output directory for results")
    parser.add_argument("--specific-dataset", type=str, default=None,
                       help="Specific dataset file to evaluate")
    parser.add_argument("--max-episodes", type=int, default=None,
                       help="Maximum episodes to evaluate (for testing)")
    parser.add_argument("--batch-size", type=int, default=4,
                       help="Batch size for processing")
    parser.add_argument("--tensor-parallel", type=int, default=4,
                       help="Number of GPUs for tensor parallelism")
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="Temperature for generation")
    parser.add_argument("--max-tokens", type=int, default=None,
                       help="Maximum tokens to generate (uses model default if not specified)")
    
    args = parser.parse_args()
    
    # Create configuration
    config = EvaluationConfig(
        tensor_parallel_size=args.tensor_parallel,
        temperature=args.temperature,
        batch_size=args.batch_size
    )
    
    if args.max_tokens is not None:
        config.max_tokens = args.max_tokens
    
    # Initialize evaluator
    evaluator = UltravoxAdaptiveEvaluator(config)
    
    # Determine dataset file
    if args.specific_dataset:
        if Path(args.specific_dataset).is_absolute():
            dataset_file = args.specific_dataset
        else:
            dataset_file = args.dataset_dir / args.specific_dataset
    else:
        # Find first voice dataset file
        voice_files = list(args.dataset_dir.glob("*_voice_episodes.json"))
        if not voice_files:
            print("Error: No voice dataset files found")
            return
        dataset_file = voice_files[0]
        print(f"No specific dataset provided, using: {dataset_file}")
    
    if not Path(dataset_file).exists():
        print(f"Error: Dataset file not found: {dataset_file}")
        return
    
    print(f"Evaluating dataset: {dataset_file}")
    
    # Run evaluation
    try:
        results = evaluator.evaluate_dataset(
            dataset_file=str(dataset_file),
            output_dir=str(args.output_dir),
            max_episodes=args.max_episodes,
        )
        
        if args.max_episodes and len(results) > args.max_episodes:
            print(f"Limited to first {args.max_episodes} episodes as requested")
            
    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user")
    except Exception as e:
        print(f"Evaluation failed: {e}")
        raise

if __name__ == "__main__":
    main()
