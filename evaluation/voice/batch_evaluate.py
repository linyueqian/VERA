#!/usr/bin/env python3
"""
Batch evaluation script for voice models on VERA datasets.

Supports multiple voice model types with adaptive capabilities:
- Qwen2-Audio with vLLM (adaptive: standard/mrcr/browsecomp)
- HuggingFace transformers models
- Other voice model implementations

Organized output structure by model and benchmark, similar to text evaluation structure.
"""

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path
from typing import List, Dict, Optional, Any
from datetime import datetime
import concurrent.futures
import time
import logging
import threading
from collections import deque

# Add parent directories to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))  # Add project root

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RateLimiter:
    """Rate limiter to handle API rate limits"""
    def __init__(self, max_requests_per_minute: int = 60):
        self.max_requests = max_requests_per_minute
        self.requests = deque()
        self.lock = threading.Lock()
    
    def wait_if_needed(self):
        """Wait if necessary to stay within rate limits"""
        with self.lock:
            now = time.time()
            
            # Remove requests older than 1 minute
            while self.requests and now - self.requests[0] > 60:
                self.requests.popleft()
            
            # If we're at the limit, wait
            if len(self.requests) >= self.max_requests:
                sleep_time = 60 - (now - self.requests[0]) + 1  # +1 for buffer
                logger.info(f"Rate limit reached, sleeping for {sleep_time:.1f}s")
                time.sleep(sleep_time)
                
                # Clean up old requests after sleep
                now = time.time()
                while self.requests and now - self.requests[0] > 60:
                    self.requests.popleft()
            
            # Record this request
            self.requests.append(now)

def detect_benchmark_from_dataset(dataset_file: Path) -> str:
    """Detect benchmark name from dataset file path."""
    dataset_name = dataset_file.name.lower()
    if "simpleqa" in dataset_name:
        return "simpleqa"
    elif "browsecomp" in dataset_name:
        return "browsecomp"
    elif "gpqa" in dataset_name:
        return "gpqa_diamond"
    elif "mrcr" in dataset_name:
        return "mrcr"
    elif "aime" in dataset_name:
        return "aime"
    else:
        # Extract from filename before _voice_episodes.json
        base_name = dataset_name.replace("_voice_episodes.json", "")
        return base_name

def create_benchmark_report(benchmark: str, results: List[Dict], output_dir: Path, start_time, args) -> Dict:
    """Create a report for a specific benchmark."""
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    successful = [r for r in results if r.get("success", False)]
    failed = [r for r in results if not r.get("success", False)]
    
    # Calculate accuracy statistics
    total_episodes = len(results)
    accuracy_count = 0
    total_with_answers = 0
    
    for result in results:
        if result.get("success", False):
            # Check if has expected answer
            if result.get("expected_answer"):
                total_with_answers += 1
                if result.get("contains_answer", False):
                    accuracy_count += 1
    
    accuracy = (accuracy_count / total_with_answers * 100) if total_with_answers > 0 else 0
    
    return {
        "benchmark": benchmark,
        "model": args.model_name,
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "duration_seconds": duration,
        "output_directory": str(output_dir),
        "evaluation_config": {
            "model_type": args.model_type,
            "batch_size": args.batch_size,
            "max_episodes": args.max_episodes,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens
        },
        "summary": {
            "total_episodes": total_episodes,
            "successful": len(successful),
            "failed": len(failed),
            "success_rate": len(successful) / total_episodes if total_episodes else 0,
            "accuracy_count": accuracy_count,
            "total_with_answers": total_with_answers,
            "accuracy_percentage": accuracy,
            "average_duration": sum(r.get("duration", 0) for r in successful) / len(successful) if successful else 0
        },
        "task_distribution": results[0].get("task_distribution", {}) if results else {},
        "results": results
    }

def run_voice_model_evaluation(dataset_file: Path, model_type: str, output_dir: Path, args) -> Dict[str, Any]:
    """Run voice model evaluation on a single dataset."""
    start_time = datetime.now()
    logger.info(f"Starting evaluation of {dataset_file.name} with {model_type}")
    
    # Prepare command based on model type
    if model_type == "qwen2_audio_adaptive":
        cmd = [
            "python", "models/voice/qwen2_audio_adaptive.py",
            "--dataset-dir", str(dataset_file.parent),
            "--specific-dataset", dataset_file.name,
            "--output-dir", str(output_dir),
            "--batch-size", str(args.batch_size),
            "--tensor-parallel", str(args.tensor_parallel),
            "--temperature", str(args.temperature)
        ]
        
        if args.max_tokens is not None:
            cmd.extend(["--max-tokens", str(args.max_tokens)])
        
        if args.max_episodes:
            cmd.extend(["--max-episodes", str(args.max_episodes)])
            
    elif model_type == "qwen2_audio_hf":
        cmd = [
            "python", "models/voice/qwen2_audio_hf.py",  # You'd need to create this
            "--dataset", str(dataset_file),
            "--output-dir", str(output_dir),
            "--max-episodes", str(args.max_episodes) if args.max_episodes else "None",
            "--device-map", "auto"
        ]
        
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Add max episodes limit if specified
    if args.max_episodes and model_type == "qwen2_audio_adaptive":
        cmd.extend(["--max-episodes", str(args.max_episodes)])
    
    try:
        # Run the evaluation
        logger.info(f"Running command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=args.timeout)
        
        if result.returncode != 0:
            logger.error(f"Evaluation failed for {dataset_file.name}")
            logger.error(f"STDERR: {result.stderr}")
            logger.error(f"STDOUT: {result.stdout}")
            return {
                "dataset": dataset_file.name,
                "success": False,
                "error": result.stderr,
                "duration": (datetime.now() - start_time).total_seconds()
            }
        
        # Parse results from output directory
        benchmark_name = detect_benchmark_from_dataset(dataset_file)
        result_dirs = list(output_dir.glob(f"*{benchmark_name}*"))
        
        if not result_dirs:
            logger.warning(f"No result directories found for {dataset_file.name}")
            return {
                "dataset": dataset_file.name,
                "success": False,
                "error": "No result directories found",
                "duration": (datetime.now() - start_time).total_seconds()
            }
        
        # Get the most recent result directory
        latest_result_dir = max(result_dirs, key=lambda x: x.stat().st_mtime)
        
        # Read evaluation summary
        summary_file = latest_result_dir / "evaluation_summary.json"
        if summary_file.exists():
            with open(summary_file, 'r') as f:
                summary_data = json.load(f)
            
            return {
                "dataset": dataset_file.name,
                "benchmark": benchmark_name,
                "success": True,
                "duration": (datetime.now() - start_time).total_seconds(),
                "result_directory": str(latest_result_dir),
                "summary": summary_data,
                "expected_answer": None,  # Will be populated from batch results
                "contains_answer": None,  # Will be populated from batch results
                "task_distribution": summary_data.get("task_distribution", {})
            }
        else:
            logger.warning(f"No evaluation summary found in {latest_result_dir}")
            return {
                "dataset": dataset_file.name,
                "success": False,
                "error": "No evaluation summary file found",
                "duration": (datetime.now() - start_time).total_seconds()
            }
            
    except subprocess.TimeoutExpired:
        logger.error(f"Evaluation timed out for {dataset_file.name}")
        return {
            "dataset": dataset_file.name,
            "success": False,
            "error": "Evaluation timed out",
            "duration": args.timeout
        }
    except Exception as e:
        logger.error(f"Evaluation failed for {dataset_file.name}: {e}")
        return {
            "dataset": dataset_file.name,
            "success": False,
            "error": str(e),
            "duration": (datetime.now() - start_time).total_seconds()
        }

def process_dataset_batch(dataset_files: List[Path], model_type: str, output_base_dir: Path, 
                         rate_limiter: RateLimiter, args) -> List[Dict[str, Any]]:
    """Process a batch of datasets with rate limiting."""
    results = []
    
    for dataset_file in dataset_files:
        # Apply rate limiting
        rate_limiter.wait_if_needed()
        
        # Create output directory for this model/benchmark
        benchmark_name = detect_benchmark_from_dataset(dataset_file)
        output_dir = output_base_dir / f"{model_type}_{benchmark_name}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Run evaluation
        result = run_voice_model_evaluation(dataset_file, model_type, output_dir, args)
        results.append(result)
        
        logger.info(f"Completed {dataset_file.name}: {'SUCCESS' if result['success'] else 'FAILED'}")
    
    return results

def find_voice_datasets(voice_dataset_dir: Path, pattern: str = "*_voice_episodes.json", specific_dataset: Optional[str] = None) -> List[Path]:
    """Find all voice dataset files."""
    if specific_dataset:
        # Handle specific dataset file
        specific_path = Path(specific_dataset)
        if specific_path.is_absolute():
            dataset_files = [specific_path] if specific_path.exists() else []
        else:
            dataset_files = [voice_dataset_dir / specific_path] if (voice_dataset_dir / specific_path).exists() else []
        
        if not dataset_files:
            logger.error(f"Specific dataset file not found: {specific_dataset}")
            return []
    else:
        # Use pattern matching
        dataset_files = list(voice_dataset_dir.glob(pattern))
    
    logger.info(f"Found {len(dataset_files)} voice dataset files")
    for f in dataset_files:
        logger.info(f"  - {f.name}")
    return dataset_files

def main():
    parser = argparse.ArgumentParser(description="Batch voice model evaluation on VERA datasets")
    parser.add_argument("--voice-dataset-dir", type=Path, 
                       default=Path("data/final_dataset/voice"),
                       help="Directory containing voice dataset JSON files")
    parser.add_argument("--output-dir", type=Path,
                       default=Path("voice_output"),
                       help="Output directory for results")
    parser.add_argument("--model-type", type=str,
                       default="qwen2_audio_adaptive",
                       choices=["qwen2_audio_adaptive", "qwen2_audio_hf"],
                       help="Type of voice model to evaluate")
    parser.add_argument("--model-name", type=str,
                       default=None,
                       help="Model name for output folder structure (auto-detected if not provided)")
    parser.add_argument("--max-workers", type=int, default=1,
                       help="Maximum parallel workers")
    parser.add_argument("--max-episodes", type=int, default=None,
                       help="Maximum episodes per dataset (for testing)")
    parser.add_argument("--limit", type=int, default=None,
                       help="Process only first N datasets (for testing)")
    parser.add_argument("--pattern", default="*_voice_episodes.json",
                       help="Dataset file pattern to match")
    parser.add_argument("--specific-dataset", type=str, default=None,
                       help="Specific dataset file path to evaluate (overrides pattern)")
    parser.add_argument("--rpm", type=int, default=60,
                       help="Rate limit: requests per minute")
    parser.add_argument("--timeout", type=int, default=3600,
                       help="Timeout per dataset evaluation in seconds")
    
    # Model-specific arguments
    parser.add_argument("--batch-size", type=int, default=4,
                       help="Batch size for model evaluation")
    parser.add_argument("--tensor-parallel", type=int, default=4,
                       help="Number of GPUs for tensor parallelism")
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="Generation temperature")
    parser.add_argument("--max-tokens", type=int, default=None,
                       help="Maximum tokens to generate (uses model default if not specified)")
    
    args = parser.parse_args()
    
    # Auto-detect model name if not provided
    if not args.model_name:
        if args.model_type == "qwen2_audio_adaptive":
            args.model_name = "qwen2_audio_adaptive"
        else:
            args.model_name = args.model_type
    
    # Paths
    voice_dataset_dir = args.voice_dataset_dir
    output_base_dir = args.output_dir
    
    # Validate paths
    if not voice_dataset_dir.exists():
        logger.error(f"Voice dataset directory not found: {voice_dataset_dir}")
        return
    
    # Create output directory
    output_base_dir.mkdir(parents=True, exist_ok=True)
    
    # Find dataset files
    dataset_files = find_voice_datasets(voice_dataset_dir, args.pattern, args.specific_dataset)
    
    if not dataset_files:
        logger.error("No voice dataset files found")
        return
    
    # Limit datasets if specified
    if args.limit:
        dataset_files = dataset_files[:args.limit]
        logger.info(f"Limited to first {args.limit} datasets")
    
    # Create rate limiter
    rate_limiter = RateLimiter(args.rpm)
    
    start_time = datetime.now()
    logger.info(f"Starting batch evaluation of {len(dataset_files)} datasets with {args.model_type}")
    
    # Process datasets
    if args.max_workers == 1:
        # Sequential processing
        all_results = process_dataset_batch(dataset_files, args.model_type, output_base_dir, 
                                          rate_limiter, args)
    else:
        # Parallel processing
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            # Split datasets into chunks for workers
            chunk_size = max(1, len(dataset_files) // args.max_workers)
            chunks = [dataset_files[i:i + chunk_size] for i in range(0, len(dataset_files), chunk_size)]
            
            # Submit jobs
            futures = []
            for chunk in chunks:
                future = executor.submit(process_dataset_batch, chunk, args.model_type, 
                                       output_base_dir, rate_limiter, args)
                futures.append(future)
            
            # Collect results
            all_results = []
            for future in concurrent.futures.as_completed(futures):
                chunk_results = future.result()
                all_results.extend(chunk_results)
    
    end_time = datetime.now()
    total_duration = (end_time - start_time).total_seconds()
    
    # Group results by benchmark
    benchmark_results = {}
    for result in all_results:
        benchmark = result.get("benchmark", "unknown")
        if benchmark not in benchmark_results:
            benchmark_results[benchmark] = []
        benchmark_results[benchmark].append(result)
    
    # Create reports for each benchmark
    benchmark_reports = {}
    for benchmark, results in benchmark_results.items():
        report = create_benchmark_report(benchmark, results, output_base_dir, start_time, args)
        benchmark_reports[benchmark] = report
        
        # Save individual benchmark report
        benchmark_output_dir = output_base_dir / f"{args.model_type}_{benchmark}"
        benchmark_output_dir.mkdir(parents=True, exist_ok=True)
        
        report_file = benchmark_output_dir / f"{args.model_name}_batch_evaluation_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Benchmark {benchmark}: {report['summary']['successful']}/{report['summary']['total_episodes']} successful")
        if report['summary']['total_with_answers'] > 0:
            logger.info(f"  Accuracy: {report['summary']['accuracy_percentage']:.1f}%")
    
    # Create overall summary report
    total_successful = sum(len([r for r in results if r.get("success", False)]) for results in benchmark_results.values())
    total_episodes = len(all_results)
    
    overall_report = {
        "evaluation_type": "voice_batch_evaluation",
        "model_type": args.model_type,
        "model_name": args.model_name,
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "total_duration_seconds": total_duration,
        "configuration": {
            "max_workers": args.max_workers,
            "rate_limit_rpm": args.rpm,
            "batch_size": args.batch_size,
            "tensor_parallel": args.tensor_parallel,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "max_episodes": args.max_episodes
        },
        "overall_summary": {
            "total_datasets": len(dataset_files),
            "total_episodes": total_episodes,
            "successful_episodes": total_successful,
            "failed_episodes": total_episodes - total_successful,
            "success_rate": total_successful / total_episodes if total_episodes else 0,
            "benchmarks_evaluated": list(benchmark_results.keys())
        },
        "benchmark_reports": benchmark_reports
    }
    
    # Save overall report
    timestamp = start_time.strftime("%Y%m%d_%H%M%S")
    overall_report_file = output_base_dir / f"{args.model_name}_batch_evaluation_summary_{timestamp}.json"
    with open(overall_report_file, 'w') as f:
        json.dump(overall_report, f, indent=2)
    
    # Print summary
    logger.info(f"\n{'='*60}")
    logger.info(f"Batch Voice Evaluation Complete!")
    logger.info(f"{'='*60}")
    logger.info(f"Model: {args.model_name}")
    logger.info(f"Total Duration: {total_duration:.1f}s")
    logger.info(f"Datasets Processed: {len(dataset_files)}")
    logger.info(f"Episodes Processed: {total_episodes}")
    logger.info(f"Success Rate: {total_successful}/{total_episodes} ({total_successful/total_episodes*100:.1f}%)")
    logger.info(f"\nBenchmark Results:")
    for benchmark, report in benchmark_reports.items():
        summary = report['summary']
        logger.info(f"  {benchmark}: {summary['successful']}/{summary['total_episodes']} success")
        if summary['total_with_answers'] > 0:
            logger.info(f"    Accuracy: {summary['accuracy_percentage']:.1f}%")
    
    logger.info(f"\nReports saved to:")
    logger.info(f"  Overall: {overall_report_file}")
    for benchmark in benchmark_results.keys():
        benchmark_dir = output_base_dir / f"{args.model_type}_{benchmark}"
        logger.info(f"  {benchmark}: {benchmark_dir}")

if __name__ == "__main__":
    main()