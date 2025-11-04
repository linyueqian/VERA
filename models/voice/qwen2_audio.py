#!/usr/bin/env python3
"""
Adaptive Qwen2-Audio voice model evaluator
Supports different capabilities based on task requirements:
- MRCR: Context documents + audio input
- BrowseComp: Web search + audio input  
- Other tasks: Audio input only
"""

import os
import json
import time
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import torch
import librosa
import numpy as np
from vllm import LLM, SamplingParams
from vllm.multimodal import MultiModalDataDict
from tqdm import tqdm

# Import utilities
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from utils.web_search import is_browsecomp_episode

@dataclass
class EvaluationConfig:
    """Configuration for adaptive Qwen2-Audio evaluation"""
    model_name: str = "Qwen/Qwen2-Audio-7B-Instruct"
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.9
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 4096
    batch_size: int = 4  # Reduced for multimodal processing
    enforce_eager: bool = False
    audio_sample_rate: int = 16000
    audio_max_seconds: int = 30  # Qwen2-Audio requires 3000 mel frames (~30s at 16kHz)

class Qwen2AudioAdaptiveEvaluator:
    """Adaptive Qwen2-Audio evaluator with task-specific capabilities"""
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        
        print(f"Initializing Adaptive Qwen2-Audio with vLLM...")
        print(f"  Model: {config.model_name}")
        print(f"  Tensor parallel: {config.tensor_parallel_size} GPUs")
        print(f"  Max model length: auto (model default)")
        print(f"  Batch size: {config.batch_size}")
        
        # Force non-Ray backend unless explicitly enabled
        os.environ.setdefault("VLLM_USE_RAY", "0")

        # Initialize vLLM with multimodal support (use model's maximum length)
        # Include limit_mm_per_prompt for audio per vLLM Qwen2-Audio examples
        # Note: Some older vLLM versions might not support this kwarg; add fallback
        try:
            self.llm = LLM(
                model=config.model_name,
                tensor_parallel_size=config.tensor_parallel_size,
                gpu_memory_utilization=config.gpu_memory_utilization,
                trust_remote_code=True,
                dtype="auto",
                enforce_eager=config.enforce_eager,
                disable_log_stats=True,
                limit_mm_per_prompt={"audio": 1},
                distributed_executor_backend="mp",
            )
        except Exception as e:
            print(f"Primary LLM init failed ({e}); retrying without limit_mm_per_prompt...")
            self.llm = LLM(
                model=config.model_name,
                tensor_parallel_size=config.tensor_parallel_size,
                gpu_memory_utilization=config.gpu_memory_utilization,
                trust_remote_code=True,
                dtype="auto",
                enforce_eager=config.enforce_eager,
                disable_log_stats=True,
            )
        
        # Sampling parameters
        self.sampling_params = SamplingParams(
            temperature=config.temperature,
            top_p=config.top_p,
            max_tokens=config.max_tokens,
        )
        
        print("✓ Model loaded successfully with multimodal support")
    
    def load_audio(self, audio_path: str) -> Optional[tuple]:
        """Load and preprocess audio file for Qwen2-Audio (returns tuple with sample rate)"""
        try:
            if not Path(audio_path).exists():
                print(f"Warning: Audio file not found: {audio_path}")
                return None
            
            # Load audio with librosa - resample to configured sample rate (16k default)
            sr = self.config.audio_sample_rate
            audio_data, sample_rate = librosa.load(audio_path, sr=sr)

            # Enforce exact 30s window: trim or pad to avoid mel length errors
            target_len = int(self.config.audio_max_seconds * sr)
            n = len(audio_data)
            if n > target_len:
                audio_data = audio_data[:target_len]
                print(f"Trimmed audio to {self.config.audio_max_seconds}s ({target_len} samples) from {n}")
            elif n < target_len:
                pad = target_len - n
                audio_data = np.pad(audio_data, (0, pad), mode='constant')
                print(f"Padded audio to {self.config.audio_max_seconds}s ({target_len} samples) from {n}")

            print(f"Loaded audio: {audio_path} (duration: {len(audio_data)/sr:.2f}s, sr: {sr})")
            
            # Return tuple as expected by vLLM
            return (audio_data, sample_rate)
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
    
    def parse_mrcr_context(self, context: str) -> List[Dict[str, str]]:
        """Parse MRCR context document into conversation messages"""
        messages = []
        
        # Split by User: and Assistant: markers
        lines = context.split('\n')
        current_role = None
        current_content = []
        
        for line in lines:
            if line.startswith('User:'):
                if current_role and current_content:
                    messages.append({"role": current_role, "content": '\n'.join(current_content).strip()})
                current_role = "user"
                current_content = [line[5:].strip()]  # Remove 'User:' prefix
            elif line.startswith('Assistant:'):
                if current_role and current_content:
                    messages.append({"role": current_role, "content": '\n'.join(current_content).strip()})
                current_role = "assistant"
                current_content = [line[10:].strip()]  # Remove 'Assistant:' prefix
            else:
                if current_content is not None:
                    current_content.append(line)
        
        # Add the last message
        if current_role and current_content:
            messages.append({"role": current_role, "content": '\n'.join(current_content).strip()})
        
        return messages
    
    def prepare_messages_for_task(self, episode: Dict[str, Any], turn: Dict[str, Any], 
                                 task_type: str) -> tuple[List[Dict[str, str]], Optional[np.ndarray]]:
        """Prepare messages and audio based on task type"""
        
        messages = []
        audio_data = None
        text_content = turn.get('text_content', '')
        audio_file = turn.get('audio_file', '')
        
        # Load audio if available
        if audio_file:
            audio_data = self.load_audio(audio_file)
        
        # Handle task-specific context
        if task_type == 'mrcr':
            # MRCR: Parse context documents as conversation history
            context_docs = episode.get('context_documents', [])
            if context_docs:
                for doc in context_docs:
                    doc_content = doc.get('content', '')
                    parsed_messages = self.parse_mrcr_context(doc_content)
                    messages.extend(parsed_messages)
            
            # Add current user input
            if audio_data is not None:
                # With audio input - use multimodal conversation format
                messages.append({
                    "role": "user", 
                    "content": [
                        {"type": "audio", "audio": audio_data},
                        {"type": "text", "text": text_content}
                    ]
                })
            else:
                # Text-only fallback
                messages.append({"role": "user", "content": text_content})
                
        elif task_type == 'browsecomp':
            # Treat BrowseComp like standard (no web search)
            messages.append({"role": "system", "content": "You are a helpful AI assistant."})
            if audio_data is not None:
                messages.append({
                    "role": "user",
                    "content": [
                        {"type": "audio", "audio": audio_data},
                        {"type": "text", "text": text_content}
                    ]
                })
            else:
                messages.append({"role": "user", "content": text_content})
                
        else:
            # Standard: Just audio + text
            messages.append({"role": "system", "content": "You are a helpful AI assistant."})
            
            if audio_data is not None:
                messages.append({
                    "role": "user",
                    "content": [
                        {"type": "audio", "audio": audio_data},
                        {"type": "text", "text": text_content}
                    ]
                })
            else:
                messages.append({"role": "user", "content": text_content})
        
        return messages, audio_data
    
    def format_conversation_for_qwen(self, messages: List[Dict[str, Any]], has_audio: bool = False) -> str:
        """Render full message history; inject audio token in any user block that carries audio."""
        parts: List[str] = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                parts.append(f"<|im_start|>system\n{content}<|im_end|>")
            elif role == "assistant":
                parts.append(f"<|im_start|>assistant\n{content}<|im_end|>")
            else:
                audio_line = ""
                text_part = ""
                if isinstance(content, list):
                    for c in content:
                        if c.get("type") == "audio":
                            audio_line = "Audio 1: <|audio_bos|><|AUDIO|><|audio_eos|>\n"
                        elif c.get("type") == "text":
                            text_part = c.get("text", "")
                else:
                    text_part = str(content)
                parts.append(f"<|im_start|>user\n{audio_line}{text_part}<|im_end|>")
        parts.append("<|im_start|>assistant\n")
        return "\n".join(parts)
    
    def process_batch(self, episodes: List[Dict[str, Any]], dataset_source: Optional[str] = None) -> List[Dict[str, Any]]:
        """Process a batch of episodes with adaptive capabilities"""
        prompts = []
        audio_data_list = []
        metadata = []
        
        # Prepare prompts based on task types
        for episode in episodes:
            episode_id = episode.get("id") or episode.get("episode_id") or "unknown"
            turns = episode.get("turns", [])
            
            if not turns:
                continue
                
            turn = turns[0]
            task_type = self.detect_task_type(episode)
            
            try:
                # Ensure we have audio data; if not provided in turn, try to infer path
                audio_data = None
                if turn.get("audio_file"):
                    audio_data = self.load_audio(turn.get("audio_file"))
                else:
                    # Build candidate paths using dataset source and known folders
                    candidate_ids = [episode_id]
                    # Strip 'vera_' prefix fallback
                    if episode_id.startswith("vera_"):
                        candidate_ids.append(episode_id.replace("vera_", ""))
                    # Build path list
                    possible_paths: List[str] = []
                    if dataset_source:
                        for cid in candidate_ids:
                            possible_paths.append(
                                f"data/final_dataset/voice/{dataset_source}_voice_episodes_audio/{cid}.wav"
                            )
                    for cid in candidate_ids:
                        possible_paths.extend([
                            f"data/final_dataset/voice/gpqa_diamond_voice_episodes_audio/{cid}.wav",
                            f"data/final_dataset/voice/aime_voice_episodes_audio/{cid}.wav",
                            f"data/final_dataset/voice/mrcr_voice_episodes_audio/{cid}.wav",
                            f"data/final_dataset/voice/browsecomp_voice_episodes_audio/{cid}.wav",
                            f"data/final_dataset/voice/simpleqa_voice_episodes_audio/{cid}.wav",
                        ])
                    for path in possible_paths:
                        if Path(path).exists():
                            audio_data = self.load_audio(path)
                            # Update turn so downstream metadata carries the audio path
                            turn["audio_file"] = path
                            break

                messages, _ = self.prepare_messages_for_task(episode, turn, task_type)
                has_audio = audio_data is not None
                has_audio = audio_data is not None
                prompt = self.format_conversation_for_qwen(messages, has_audio)
                
                prompts.append(prompt)
                audio_data_list.append(audio_data)
                metadata.append({
                    "episode_id": episode_id,
                    "task_type": task_type,
                    "expected_answer": turn.get("metadata", {}).get("expected_answer", ""),
                    "audio_file": turn.get("audio_file", ""),
                    "original_text": turn.get("text_content", "")
                })
                
            except Exception as e:
                print(f"Error processing episode {episode_id}: {e}")
                continue
        
        # Generate responses
        if prompts:
            try:
                # Process each prompt individually to track TTFT
                results = []
                
                for i, prompt in enumerate(prompts):
                    meta = metadata[i]
                    
                    # Prepare individual input with audio data
                    input_data = {"prompt": prompt}
                    if audio_data_list[i] is not None:
                        # vLLM expects audio as list of tuples (audio_data, sample_rate)
                        input_data["multi_modal_data"] = {"audio": [audio_data_list[i]]}  # Wrap in list
                        audio_tuple = audio_data_list[i]
                        if isinstance(audio_tuple, tuple) and len(audio_tuple) == 2:
                            audio_array, sample_rate = audio_tuple
                            shape_info = getattr(audio_array, "shape", (len(audio_array),))
                            print(f"Processing {meta['episode_id']} with audio: shape={shape_info}, sr={sample_rate}")
                        else:
                            print(f"Processing {meta['episode_id']} with audio type: {type(audio_tuple)}")
                    else:
                        print(f"Processing {meta['episode_id']} without audio")
                    
                    # Track timing for TTFT
                    request_start_time = time.time()
                    first_token_time = None
                    full_response = ""
                    
                    try:
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
                        # Handle individual prompt errors (e.g., context/capacity limits)
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
            "api_latency_ms": 0.0,
            "processing_overhead_ms": session_duration * 1000,
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
                "model": "qwen2-audio-7b-instruct",
                "deployment": f"qwen2-audio-7b-{task_type}-vllm",
                "response_id": f"qwen_{episode_id}_{int(time.time())}",
                "finish_reason": "stop"
            },
            "error": result.get("error")
        }
        
        # Create episode result in text_output format
        episode_result = {
            "episode_id": episode_id,
            "model_name": f"qwen2-audio-7b-{task_type}-vllm",
            "task_type": task_type,
            "session_duration": session_duration,
            "total_turns_processed": 1,
            "total_tokens_used": token_usage.get("total_tokens", 0),
            "turn_results": [turn_result],
            "timing_data": {
                "total_session_time_ms": session_duration * 1000,
                "average_total_turn_latency_ms": session_duration * 1000,
                "average_api_latency_ms": 0.0,
                "average_processing_overhead_ms": session_duration * 1000,
                "total_api_time_ms": 0.0,
                "total_processing_overhead_ms": session_duration * 1000
            },
            "reasoning_data": {
                "responses": [response],
                "average_response_length": len(response)
            },
            "success": result.get("error") is None,
            "expected_answer": expected_answer,
            "contains_answer": result.get("contains_answer", False),
            "has_audio": result.get("has_audio", False)
        }
        
        return episode_result
    
    def evaluate_dataset(self, dataset_path: str, output_base_dir: str, max_episodes: Optional[int] = None):
        """Evaluate voice dataset with adaptive task-specific processing"""
        # Load dataset
        print(f"\nLoading dataset: {dataset_path}")
        with open(dataset_path, 'r') as f:
            data = json.load(f)
        
        episodes = data.get("episodes", [])
        track = data.get("track", "unknown")
        source = data.get("source", "unknown")
        # Normalize source to match folder naming (drop trailing _converted)
        if isinstance(source, str) and source.endswith("_converted"):
            source = source[:-len("_converted")]
        
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
        
        # Create output directory with timestamp
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_dir_name = f"qwen2audio_adaptive_{source}_{timestamp}"
        output_path = Path(output_base_dir) / output_dir_name
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Process in batches
        all_episode_results = []
        batch_size = self.config.batch_size
        
        start_time = time.time()
        
        for i in tqdm(range(0, len(episodes), batch_size), desc="Processing adaptive batches"):
            batch = episodes[i:i+batch_size]
            batch_results = self.process_batch(batch, dataset_source=source)
            
            # Convert to text_output format
            for result in batch_results:
                episode_result = self.convert_to_text_output_format(result, start_time)
                all_episode_results.append(episode_result)
        
        elapsed_time = time.time() - start_time
        
        # Calculate statistics  
        successful_episodes = len([r for r in all_episode_results if r.get("success", False)])
        failed_episodes = len(all_episode_results) - successful_episodes
        with_audio = len([r for r in all_episode_results if r.get("has_audio", False)])
        
        # Create evaluation summary
        evaluation_summary = {
            "model": "qwen2audio-adaptive",
            "dataset": source,
            "dataset_path": dataset_path,
            "output_directory": str(output_path),
            "total_episodes": len(all_episode_results),
            "processed": len(all_episode_results),
            "successful": successful_episodes,
            "failed": failed_episodes,
            "with_audio": with_audio,
            "task_distribution": task_counts,
            "duration_seconds": elapsed_time,
            "episodes_per_second": len(all_episode_results) / elapsed_time if elapsed_time > 0 else 0,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S.%f"),
            "max_concurrent": self.config.batch_size,
            "async_processing": False
        }
        
        # Create main batch results file
        batch_id = int(time.time() * 1000)
        batch_results = {
            "model_name": "qwen2-audio-7b-adaptive-vllm",
            "processed": len(all_episode_results),
            "successful": successful_episodes,
            "failed": failed_episodes,
            "batch_duration": elapsed_time,
            "episodes_per_second": len(all_episode_results) / elapsed_time if elapsed_time > 0 else 0,
            "max_concurrent": self.config.batch_size,
            "task_distribution": task_counts,
            "results": all_episode_results
        }
        
        # Save files
        summary_file = output_path / "evaluation_summary.json"
        batch_file = output_path / f"qwen2audio_adaptive_batch_{batch_id}.json"
        
        # Ensure directory still exists (in case of race conditions)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"Saving results to: {output_path}")
        print(f"Directory exists: {output_path.exists()}")
        
        with open(summary_file, 'w') as f:
            json.dump(evaluation_summary, f, indent=2)
            
        with open(batch_file, 'w') as f:
            json.dump(batch_results, f, indent=2)
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"Adaptive Evaluation Complete: {source}")
        print(f"{'='*60}")
        print(f"Output directory: {output_path}")
        print(f"Total episodes: {len(all_episode_results)}")
        print(f"Successful: {successful_episodes}")
        print(f"Failed: {failed_episodes}")
        print(f"With audio: {with_audio}")
        print(f"Task distribution: {task_counts}")
        print(f"Time taken: {elapsed_time:.1f}s")
        print(f"Speed: {len(all_episode_results) / elapsed_time:.1f} episodes/sec")
        print(f"Files created:")
        print(f"  - {summary_file}")
        print(f"  - {batch_file}")
        
        return evaluation_summary

def main():
    parser = argparse.ArgumentParser(description="Adaptive Qwen2-Audio evaluator for VERA voice datasets")
    parser.add_argument("--dataset-dir", type=str, default="data/final_dataset/voice",
                       help="Directory containing voice datasets")
    parser.add_argument("--output-dir", type=str, default="voice_output",
                       help="Directory to save results")
    parser.add_argument("--specific-dataset", type=str, default=None,
                       help="Specific dataset to evaluate")
    parser.add_argument("--max-episodes", type=int, default=None,
                       help="Maximum episodes to evaluate (for testing)")
    parser.add_argument("--batch-size", type=int, default=4,
                       help="Batch size for processing (reduced for multimodal)")
    parser.add_argument("--tensor-parallel", type=int, default=4,
                       help="Number of GPUs for tensor parallelism")
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="Temperature for generation")
    parser.add_argument("--max-tokens", type=int, default=512,
                       help="Maximum tokens to generate")
    
    args = parser.parse_args()
    
    # Create configuration
    config = EvaluationConfig(
        batch_size=args.batch_size,
        tensor_parallel_size=args.tensor_parallel,
        temperature=args.temperature,
        max_tokens=args.max_tokens
    )
    
    # Initialize evaluator
    print("Initializing Adaptive Qwen2-Audio evaluator...")
    evaluator = Qwen2AudioAdaptiveEvaluator(config)
    
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
    print(f"All adaptive evaluations complete!")
    print(f"Results saved to: {args.output_dir}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
