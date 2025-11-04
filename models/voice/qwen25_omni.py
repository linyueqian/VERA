#!/usr/bin/env python3
"""
Qwen2.5-Omni HuggingFace Transformers-based evaluator
Simplified version using transformers instead of vLLM for easier setup
Supports audio+text input/output for voice evaluation tasks
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
import torch
import librosa
import numpy as np
import soundfile as sf
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
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

# Import qwen_omni_utils if available, otherwise define basic version
try:
    from qwen_omni_utils import process_mm_info
except ImportError:
    def process_mm_info(conversation, use_audio_in_video=False):
        """Basic implementation of process_mm_info if not available"""
        audios, images, videos = [], [], []
        
        for message in conversation:
            if not isinstance(message['content'], list):
                continue
                
            for content in message['content']:
                if content.get('type') == 'audio':
                    if 'audio' in content:
                        # Load audio file
                        audio_path = content['audio']
                        if isinstance(audio_path, str) and os.path.exists(audio_path):
                            audio, _ = librosa.load(audio_path, sr=16000)
                            audios.append(audio)
                        elif isinstance(audio_path, np.ndarray):
                            audios.append(audio_path)
        
        return audios, images, videos

@dataclass
class EvaluationConfig:
    """Configuration for HuggingFace Qwen2.5-Omni evaluation"""
    model_path: str = "Qwen/Qwen2.5-Omni-7B"
    device_map: Union[str, dict] = "cuda:0"  # Use single GPU instead of auto
    torch_dtype: str = "auto"
    attn_implementation: Optional[str] = None  # "flash_attention_2" for better performance
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 4096
    batch_size: int = 1  # HF transformers processes one at a time
    audio_sample_rate: int = 16000
    output_sample_rate: int = 24000
    use_audio_in_video: bool = False  # Set to True only if processing video content
    text_only: bool = False

class Qwen25OmniHFEvaluator:
    """HuggingFace Transformers-based Qwen2.5-Omni evaluator"""
    
    def __init__(self, config: EvaluationConfig, output_dir: str = None):
        self.config = config
        self.output_dir = output_dir or "voice_output"
        
        print(f"Initializing Qwen2.5-Omni with HuggingFace Transformers...")
        print(f"  Model: {config.model_path}")
        print(f"  Device map: {config.device_map}")
        print(f"  Attention: {config.attn_implementation or 'default'}")
        print(f"  Output directory: {self.output_dir}")
        print(f"  Batch size: {config.batch_size}")
        
        # Initialize model - disable tensor parallelism for single-node inference
        model_kwargs = {
            "torch_dtype": config.torch_dtype,
            "device_map": config.device_map,
            "trust_remote_code": True,
            # Disable automatic tensor parallelism
            "tp_plan": None,
        }
        
        if config.attn_implementation:
            model_kwargs["attn_implementation"] = config.attn_implementation
        
        print("Loading model...")
        self.model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            config.model_path,
            **model_kwargs
        )
        
        print("Loading processor...")
        self.processor = Qwen2_5OmniProcessor.from_pretrained(
            config.model_path,
            trust_remote_code=True
        )
        
        print("✓ Model and processor loaded successfully")
    
    def load_audio(self, audio_path: str) -> Optional[np.ndarray]:
        """Load and preprocess audio file to 16kHz"""
        try:
            if not Path(audio_path).exists():
                print(f"Warning: Audio file not found: {audio_path}")
                return None
            
            # Load audio with librosa at 16kHz
            audio, sr = librosa.load(audio_path, sr=16000)
            return audio
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
    
    def prepare_conversation(self, episode: Dict[str, Any], turn: Dict[str, Any], 
                           task_type: str) -> List[Dict[str, Any]]:
        """Prepare conversation for Qwen2.5-Omni"""
        
        text_content = turn.get('text_content', '')
        audio_file = turn.get('audio_file', '')
        
        # Start with system message
        if self.config.text_only:
            system_content = "You are a helpful assistant."
        else:
            system_content = "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."
        
        conversation = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_content}]
            }
        ]
        
        # Prepare user content
        user_content = []
        
        # Handle task-specific context
        if task_type == 'mrcr':
            # MRCR: Add context documents if available
            context_docs = episode.get('context_documents', [])
            if context_docs:
                context_text = "\n\n".join([doc.get('content', '') for doc in context_docs])
                if context_text:
                    text_content = f"Context:\n{context_text}\n\nQuestion: {text_content}"
        
        # Add audio if available
        if audio_file and not self.config.text_only:
            audio_data = self.load_audio(audio_file)
            if audio_data is not None:
                user_content.append({"type": "audio", "audio": audio_data})
        
        # Add text content
        user_content.append({"type": "text", "text": text_content})
        
        conversation.append({
            "role": "user", 
            "content": user_content
        })
        
        return conversation
    
    def process_episode(self, episode: Dict[str, Any], task_type: str) -> Dict[str, Any]:
        """Process a single episode using HuggingFace Transformers"""
        episode_id = episode.get("id", "unknown")
        turns = episode.get("turns", [])
        
        if not turns:
            return None
        
        turn = turns[0]
        
        try:
            # Prepare conversation
            conversation = self.prepare_conversation(episode, turn, task_type)
            
            # Track timing
            request_start_time = time.time()
            
            # Prepare inputs
            text = self.processor.apply_chat_template(
                conversation, 
                add_generation_prompt=True, 
                tokenize=False
            )
            
            # Process multimedia content
            audios, images, videos = process_mm_info(
                conversation, 
                use_audio_in_video=self.config.use_audio_in_video
            )
            
            # Prepare model inputs
            inputs = self.processor(
                text=text, 
                audio=audios if audios else None, 
                images=images if images else None, 
                videos=videos if videos else None, 
                return_tensors="pt", 
                padding=True, 
                use_audio_in_video=self.config.use_audio_in_video
            )
            
            # Move to model device
            inputs = inputs.to(self.model.device)
            if hasattr(self.model, 'dtype'):
                inputs = inputs.to(self.model.dtype)
            
            first_token_time = time.time() - request_start_time
            
            # Generate response
            with torch.no_grad():
                if self.config.text_only:
                    # Text-only generation
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=self.config.max_tokens,
                        temperature=self.config.temperature,
                        top_p=self.config.top_p,
                        do_sample=True,
                        pad_token_id=self.processor.tokenizer.eos_token_id
                    )
                    
                    # Decode text response
                    response_ids = outputs[0][len(inputs.input_ids[0]):]
                    text_response = self.processor.tokenizer.decode(
                        response_ids, 
                        skip_special_tokens=True, 
                        clean_up_tokenization_spaces=False
                    )
                    audio_output = None
                    
                else:
                    # Audio+text generation
                    text_ids, audio_output = self.model.generate(
                        **inputs,
                        max_new_tokens=self.config.max_tokens,
                        temperature=self.config.temperature,
                        top_p=self.config.top_p,
                        do_sample=True,
                        use_audio_in_video=self.config.use_audio_in_video,
                        pad_token_id=self.processor.tokenizer.eos_token_id
                    )
                    
                    # Decode text response
                    response_ids = text_ids[0][len(inputs.input_ids[0]):]
                    text_response = self.processor.tokenizer.decode(
                        response_ids, 
                        skip_special_tokens=True, 
                        clean_up_tokenization_spaces=False
                    )
            
            request_end_time = time.time()
            total_time = request_end_time - request_start_time
            
            # Check if expected answer is in response (do this early)
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
            if turn.get('audio_file') and Path(turn['audio_file']).exists():
                input_audio_path = episode_dir / "input.wav"
                # Copy or link the input audio
                import shutil
                shutil.copy2(turn['audio_file'], input_audio_path)
            
            # Save audio output if generated
            audio_output_path = None
            if audio_output is not None and not self.config.text_only:
                audio_output_path = episode_dir / "output.wav"
                
                # Convert audio tensor to numpy and save
                audio_numpy = audio_output.reshape(-1).detach().cpu().numpy()
                sf.write(str(audio_output_path), audio_numpy, samplerate=self.config.output_sample_rate)
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
                    "first_token_time": first_token_time
                },
                "model": "qwen2.5-omni-hf"
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
                f.write(f"Has Audio Input: {len(audios) > 0 if audios else False}\n")
                f.write(f"Has Audio Output: {audio_output_path is not None}\n")
                f.write(f"Model: qwen2.5-omni-hf\n")
            
            return {
                "episode_id": episode_id,
                "task_type": task_type,
                "response": text_response,
                "audio_response_path": audio_output_path,
                "expected_answer": expected,
                "contains_answer": contains_answer,
                "audio_file": turn.get("audio_file", ""),
                "original_text": turn.get("text_content", ""),
                "has_audio_input": len(audios) > 0 if audios else False,
                "has_audio_output": audio_output_path is not None,
                "timing": {
                    "request_start_time": request_start_time,
                    "request_end_time": request_end_time,
                    "first_token_time": first_token_time,
                    "total_time": total_time
                },
                "token_usage": {
                    "completion_tokens": len(response_ids) if 'response_ids' in locals() else 0,
                    "total_tokens": len(response_ids) if 'response_ids' in locals() else 0
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
                "has_audio_input": False,
                "has_audio_output": False,
                "token_usage": {
                    "completion_tokens": 0,
                    "total_tokens": 0
                }
            }
    
    def evaluate_dataset(self, dataset_path: str, output_base_dir: str, max_episodes: Optional[int] = None):
        """Evaluate voice dataset with HuggingFace Transformers"""
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
        
        # Create output directory structure matching azure_gpt_realtime format
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        # voice_output/qwen25omni_hf/[dataset]/[timestamp]/
        output_path = Path(output_base_dir) / "qwen25omni_hf" / source / timestamp
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
        with_audio_output = len([r for r in all_results if r.get("has_audio_output", False)])
        
        # Create evaluation summary
        evaluation_summary = {
            "model": "qwen2.5-omni-hf",
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
                "model_path": self.config.model_path,
                "device_map": self.config.device_map,
                "torch_dtype": self.config.torch_dtype,
                "attn_implementation": self.config.attn_implementation,
                "text_only": self.config.text_only
            }
        }
        
        # Save results (matching azure_gpt_realtime format)
        summary_file = output_path / "benchmark_report.json"
        results_file = output_path / f"qwen25omni_hf_results_{int(time.time())}.json"
        
        with open(summary_file, 'w') as f:
            json.dump(evaluation_summary, f, indent=2)
            
        with open(results_file, 'w') as f:
            # Convert Path objects to strings for JSON serialization
            all_results_serializable = convert_paths_to_strings(all_results)
            json.dump({
                "model_name": "qwen2.5-omni-hf",
                "results": all_results_serializable
            }, f, indent=2)
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"HF Evaluation Complete: {source}")
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
    parser = argparse.ArgumentParser(description="Qwen2.5-Omni HuggingFace Transformers evaluator")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-Omni-7B",
                       help="Model path or HuggingFace model ID")
    parser.add_argument("--dataset-dir", type=str, default="data/final_dataset/voice",
                       help="Directory containing voice datasets")
    parser.add_argument("--output-dir", type=str, default="voice_output",
                       help="Directory to save results")
    parser.add_argument("--specific-dataset", type=str, default=None,
                       help="Specific dataset to evaluate")
    parser.add_argument("--max-episodes", type=int, default=None,
                       help="Maximum episodes to evaluate (for testing)")
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="Temperature for generation")
    parser.add_argument("--max-tokens", type=int, default=4096,
                       help="Maximum tokens to generate")
    parser.add_argument("--text-only", action="store_true",
                       help="Text-only mode without audio generation")
    parser.add_argument("--device-map", type=str, default="cuda:0",
                       help="Device mapping strategy (use 'cuda:0' for single GPU, 'auto' for multi-GPU)")
    parser.add_argument("--torch-dtype", type=str, default="auto",
                       help="PyTorch dtype for model")
    parser.add_argument("--flash-attention", action="store_true",
                       help="Use flash attention 2 for better performance")
    
    args = parser.parse_args()
    
    # Create configuration
    config = EvaluationConfig(
        model_path=args.model,
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
        attn_implementation="flash_attention_2" if args.flash_attention else None,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        text_only=args.text_only
    )
    
    # Initialize evaluator
    print("Initializing Qwen2.5-Omni HF evaluator...")
    evaluator = Qwen25OmniHFEvaluator(config, args.output_dir)
    
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
    print(f"All HF evaluations complete!")
    print(f"Results saved to: {args.output_dir}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
