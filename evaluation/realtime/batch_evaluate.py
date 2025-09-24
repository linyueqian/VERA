#!/usr/bin/env python3
"""
Batch evaluation script for processing voice audio files through realtime models.

This script processes all audio files in data/final_dataset/voice using realtime voice models,
with organized output structure by model and benchmark, similar to evaluation/text structure.
"""

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path
from typing import List, Dict, Optional
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
    """Rate limiter to handle API rate limits (e.g., 30 RPM)"""
    def __init__(self, max_requests_per_minute: int = 30):
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

def detect_benchmark_from_audio(audio_file: Path) -> str:
    """Detect benchmark name from audio file path."""
    audio_name = audio_file.name.lower()
    if "simpleqa" in audio_name:
        return "simpleqa"
    elif "browsecomp" in audio_name:
        return "browsecomp"
    elif "gpqa" in audio_name or "gpqadiamond" in audio_name:
        return "gpqa_diamond"
    elif "mrcr" in audio_name:
        return "mrcr"
    elif "aime" in audio_name:
        return "aime"
    else:
        # Try to detect from parent directory
        parent_dir = audio_file.parent.name.lower()
        if "simpleqa" in parent_dir:
            return "simpleqa"
        elif "browsecomp" in parent_dir:
            return "browsecomp"
        elif "gpqa" in parent_dir or "gpqadiamond" in parent_dir:
            return "gpqa_diamond"
        elif "mrcr" in parent_dir:
            return "mrcr"
        elif "aime" in parent_dir:
            return "aime"
        else:
            return "unknown"

def create_benchmark_report(benchmark: str, results: List[Dict], output_dir: Path, start_time, args) -> Dict:
    """Create a report for a specific benchmark."""
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]
    
    return {
        "benchmark": benchmark,
        "model": args.model_name,
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "duration_seconds": duration,
        "output_directory": str(output_dir),
        "summary": {
            "total_files": len(results),
            "successful": len(successful),
            "failed": len(failed),
            "success_rate": len(successful) / len(results) if results else 0,
            "average_duration": sum(r["duration"] for r in successful) / len(successful) if successful else 0
        },
        "results": results
    }

def preload_episodes(dataset_dir: Path) -> Dict[str, Path]:
    """Preload all episodes into memory for fast lookup."""
    logger.info("Preloading episodes for fast lookup...")
    episode_map = {}

    start_time = time.time()
    json_files_processed = 0
    episodes_loaded = 0

    for json_file in dataset_dir.rglob("*.json"):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                episodes = data.get("episodes", [])
                json_files_processed += 1

                for ep in episodes:
                    episode_id = ep.get("id", "")
                    if episode_id:
                        # Store full episode ID for exact match (this is the primary mapping)
                        episode_map[episode_id] = json_file
                        episodes_loaded += 1

                        # Store short ID only if no conflict with existing full IDs
                        if "_" in episode_id:
                            short_id = episode_id.split("_")[-1]
                            # Only add short_id if it doesn't conflict with any full episode ID
                            if short_id not in episode_map:
                                episode_map[short_id] = json_file
        except Exception as e:
            logger.warning(f"Failed to parse {json_file}: {e}")
            continue

    elapsed = time.time() - start_time
    logger.info(f"Preloaded {episodes_loaded} episodes from {json_files_processed} JSON files in {elapsed:.2f}s")
    return episode_map

def find_episode_json_for_audio_fast(audio_file: Path, episode_map: Dict[str, Path]) -> Optional[Path]:
    """Find the corresponding episode JSON file for an audio file using preloaded map."""
    # Extract the ID from audio filename (e.g., vera_simpleqa_efbbb22c.wav -> efbbb22c)
    audio_name = audio_file.stem
    logger.debug(f"Looking for episode for audio file: {audio_name}")

    if audio_name.startswith("vera_"):
        # Try exact match first (full audio filename should match episode ID exactly)
        if audio_name in episode_map:
            logger.debug(f"Found exact match: {audio_name} -> {episode_map[audio_name]}")
            return episode_map[audio_name]

        parts = audio_name.split("_")
        if len(parts) >= 3:
            # Try short ID match (last part) only as fallback
            episode_id = parts[-1]
            if episode_id in episode_map:
                logger.debug(f"Found short ID match: {episode_id} -> {episode_map[episode_id]}")
                return episode_map[episode_id]

    logger.debug(f"No match found for {audio_name}")
    return None

def find_episode_json_for_audio(audio_file: Path, dataset_dir: Path) -> Optional[Path]:
    """Legacy function - kept for compatibility but not recommended for batch processing."""
    # Extract the ID from audio filename (e.g., vera_simpleqa_efbbb22c.wav -> efbbb22c)
    audio_name = audio_file.stem
    if audio_name.startswith("vera_"):
        parts = audio_name.split("_")
        if len(parts) >= 3:
            episode_id = parts[-1]  # Last part should be the ID

            # Look for JSON files with this ID in the dataset directory
            for json_file in dataset_dir.rglob("*.json"):
                if episode_id in json_file.name:
                    return json_file

                # Also check inside JSON files for matching episode ID
                try:
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                        episodes = data.get("episodes", [])
                        for ep in episodes:
                            if ep.get("id", "").endswith(episode_id):
                                return json_file
                except:
                    continue

    return None

def run_single_inference(audio_file: Path, episode_json: Path, output_dir: Path, 
                        azure_script: Path, rate_limiter: RateLimiter, benchmark: str = "unknown", max_retries: int = 3) -> Dict:
    """Run inference for a single audio file."""
    audio_id = audio_file.stem
    result_dir = output_dir / audio_id
    result_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Processing {audio_id}")
    logger.debug(f"Audio file: {audio_file}")
    logger.debug(f"Episode JSON: {episode_json}")
    logger.debug(f"Output directory: {result_dir}")

    # Prepare command
    cmd = [
        "/opt/venv/bin/python", str(azure_script),
        str(episode_json),
        str(result_dir),
        "--mode", "audio",
        "--target-audio", str(audio_file)
    ]
    logger.debug(f"Command: {' '.join(cmd)}")
    
    for attempt in range(max_retries):
        try:
            # Wait for rate limiting before making API call
            rate_limiter.wait_if_needed()
            
            start_time = time.time()
            
            # Run the inference
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout per audio file (increased for web search)
                cwd=Path.cwd()
            )
            
            duration = time.time() - start_time
            
            # Check if successful
            response_json = result_dir / "response.json"
            output_wav = result_dir / "output.wav"
            
            success = (
                result.returncode == 0 and 
                response_json.exists() and 
                output_wav.exists()
            )
            
            # Log execution details
            logger.debug(f"Return code: {result.returncode}")
            logger.debug(f"Duration: {duration:.2f}s")
            if result.stdout:
                logger.debug(f"STDOUT: {result.stdout[:500]}...")  # First 500 chars
            if result.stderr:
                logger.debug(f"STDERR: {result.stderr[:500]}...")  # First 500 chars

            # Save execution log
            log_file = result_dir / "execution.log"
            with open(log_file, 'w') as f:
                f.write(f"Command: {' '.join(cmd)}\n")
                f.write(f"Return code: {result.returncode}\n")
                f.write(f"Duration: {duration:.2f}s\n")
                f.write(f"Attempt: {attempt + 1}/{max_retries}\n\n")
                f.write("=== STDOUT ===\n")
                f.write(result.stdout)
                f.write("\n=== STDERR ===\n")
                f.write(result.stderr)
            
            if success:
                logger.info(f"✓ {audio_id} completed successfully in {duration:.1f}s")
                return {
                    "audio_file": str(audio_file),
                    "episode_json": str(episode_json),
                    "output_dir": str(result_dir),
                    "success": True,
                    "duration": duration,
                    "attempt": attempt + 1,
                    "error": None
                }
            else:
                error_msg = f"Failed: return_code={result.returncode}"
                if attempt < max_retries - 1:
                    logger.warning(f"⚠ {audio_id} failed (attempt {attempt+1}), retrying...")
                    time.sleep(2)  # Brief delay before retry
                    continue
                else:
                    logger.error(f"✗ {audio_id} failed after {max_retries} attempts")
                    return {
                        "audio_file": str(audio_file),
                        "episode_json": str(episode_json),
                        "output_dir": str(result_dir),
                        "success": False,
                        "duration": duration,
                        "attempt": attempt + 1,
                        "error": error_msg
                    }
                    
        except subprocess.TimeoutExpired:
            duration = time.time() - start_time
            error_msg = f"Timeout after {duration:.1f}s"
            logger.warning(f"⚠ {audio_id} timed out (attempt {attempt+1})")
            if attempt < max_retries - 1:
                time.sleep(2)
                continue
            else:
                logger.error(f"✗ {audio_id} timed out after {max_retries} attempts")
                return {
                    "audio_file": str(audio_file),
                    "episode_json": str(episode_json),
                    "output_dir": str(result_dir),
                    "success": False,
                    "duration": duration,
                    "attempt": attempt + 1,
                    "error": error_msg
                }
        except Exception as e:
            error_msg = f"Exception: {str(e)}"
            if attempt < max_retries - 1:
                logger.warning(f"⚠ {audio_id} error (attempt {attempt+1}): {e}")
                time.sleep(2)
                continue
            else:
                logger.error(f"✗ {audio_id} failed with exception: {e}")
                return {
                    "audio_file": str(audio_file),
                    "episode_json": str(episode_json),
                    "output_dir": str(result_dir),
                    "success": False,
                    "duration": time.time() - start_time,
                    "attempt": attempt + 1,
                    "error": error_msg
                }

def main():
    parser = argparse.ArgumentParser(description="Batch voice inference using Azure GPT Realtime")
    parser.add_argument("--voice-dir", type=Path, 
                       default=Path("data/final_dataset/voice"),
                       help="Directory containing voice audio files")
    parser.add_argument("--dataset-dir", type=Path,
                       default=Path("data/final_dataset"),
                       help="Directory containing episode JSON files")
    parser.add_argument("--output-dir", type=Path,
                       default=Path("voice_output"),
                       help="Output directory for results")
    parser.add_argument("--model-name", type=str,
                       default="azure_gpt_realtime",
                       help="Model name for output folder structure")
    parser.add_argument("--max-workers", type=int, default=1,
                       help="Maximum parallel workers (consider rate limits: 30 RPM)")
    parser.add_argument("--max-retries", type=int, default=3,
                       help="Maximum retries per audio file")
    parser.add_argument("--limit", type=int, default=None,
                       help="Process only first N audio files (for testing)")
    parser.add_argument("--pattern", default="*.wav",
                       help="Audio file pattern to match")
    parser.add_argument("--rpm", type=int, default=30,
                       help="Rate limit: requests per minute")
    # Resume options
    parser.add_argument("--resume-from", type=Path, default=None,
                       help="Path to an overall or benchmark report JSON, or a model/batch directory to resume from. If a directory of model outputs is provided, the latest overall report is used.")
    parser.add_argument("--only-failed", action="store_true",
                       help="(Legacy flag - now default behavior) When resuming, only re-run items that previously failed or are missing output.wav")
    parser.add_argument("--continue-all", action="store_true",
                       help="When resuming, process failed items AND continue with remaining unprocessed episodes")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose debug logging")

    args = parser.parse_args()

    # Set logging level based on verbose flag
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Verbose logging enabled")
    
    # Paths
    voice_dir = args.voice_dir
    dataset_dir = args.dataset_dir
    output_dir = Path(args.output_dir)  # Convert to Path object
    azure_script = Path("models/realtime/inference/azure_gpt_realtime/main.py")
    
    # Validate paths
    if not voice_dir.exists():
        logger.error(f"Voice directory not found: {voice_dir}")
        return 1
        
    if not dataset_dir.exists():
        logger.error(f"Dataset directory not found: {dataset_dir}")
        return 1
        
    if not azure_script.exists():
        logger.error(f"Azure script not found: {azure_script}")
        return 1
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    def _load_resume_report(path: Path) -> Dict:
        """Load a report JSON given a path that can be a file or directory.
        If no report JSON exists, scan the directory for individual response.json files.
        Returns a dict with keys: type ('overall'|'benchmark'|'scan'), data (JSON), and base_path.
        """
        chosen: Optional[Path] = None
        base_path = path

        if path.is_file() and path.suffix == ".json":
            chosen = path
        elif path.is_dir():
            # Prefer latest overall report in this directory tree
            candidates = sorted((p for p in path.glob("overall_batch_report_*.json")), reverse=True)
            if not candidates:
                # Maybe it's a model root like output_dir/model_name
                candidates = sorted((p for p in path.rglob("overall_batch_report_*.json")), reverse=True)
            if not candidates:
                # Maybe it's a specific benchmark batch directory with benchmark_report.json
                bench = path / "benchmark_report.json"
                if bench.exists():
                    chosen = bench
                else:
                    # No report JSON found - scan for individual response.json files
                    logger.info(f"No report JSON found, scanning {path} for individual response.json files...")
                    return _scan_directory_for_results(path)
            else:
                chosen = candidates[0]
        else:
            raise FileNotFoundError(f"Resume path not found: {path}")

        with open(chosen, 'r') as f:
            data = json.load(f)
        report_type = 'overall' if 'batch_id' in data else 'benchmark'
        return {"type": report_type, "data": data, "report_path": str(chosen), "base_path": str(base_path)}

    def _scan_directory_for_results(directory: Path) -> Dict:
        """Scan a directory for episode result folders and determine which have completed successfully."""
        logger.info(f"Scanning {directory} for completed and failed episodes...")

        results = []
        episode_dirs = [d for d in directory.iterdir() if d.is_dir() and d.name.startswith("vera_")]

        for episode_dir in episode_dirs:
            episode_id = episode_dir.name
            response_json = episode_dir / "response.json"
            output_wav = episode_dir / "output.wav"
            execution_log = episode_dir / "execution.log"

            if response_json.exists() and output_wav.exists():
                # Successfully completed
                try:
                    with open(response_json) as f:
                        response_data = json.load(f)

                    # Extract info from response.json and infer audio file path
                    input_file = response_data.get("audio_info", {}).get("input_file", "")

                    # If input_file is empty, try to infer from episode_id
                    if not input_file:
                        # Try common audio file locations based on episode_id
                        possible_paths = [
                            f"data/final_dataset/voice/mrcr_voice_episodes_audio/{episode_id}.wav",
                            f"data/final_dataset/voice/aime_voice_episodes_audio/{episode_id}.wav",
                            f"data/final_dataset/voice/simpleqa_voice_episodes_audio/{episode_id}.wav",
                            f"data/final_dataset/voice/gpqa_voice_episodes_audio/{episode_id}.wav",
                            f"data/final_dataset/voice/gpqa_diamond_voice_episodes_audio/{episode_id}.wav",
                            f"data/final_dataset/voice/browsecomp_voice_episodes_audio/{episode_id}.wav",
                        ]
                        for path in possible_paths:
                            if Path(path).exists():
                                input_file = path
                                break

                    results.append({
                        "episode_id": episode_id,
                        "success": True,
                        "audio_file": input_file,
                        "episode_json": "", # Will be determined later
                        "output_dir": str(episode_dir),
                        "duration": response_data.get("timing", {}).get("total_response_time", 0),
                        "error": None
                    })
                except Exception as e:
                    logger.warning(f"Failed to parse response.json in {episode_dir}: {e}")
                    results.append({
                        "episode_id": episode_id,
                        "success": False,
                        "audio_file": "",
                        "episode_json": "",
                        "output_dir": str(episode_dir),
                        "duration": 0,
                        "error": f"Failed to parse response.json: {e}"
                    })
            else:
                # Failed or incomplete
                error_msg = "Missing response.json or output.wav"
                if execution_log.exists():
                    try:
                        with open(execution_log) as f:
                            log_content = f.read()
                            if "Return code:" in log_content:
                                lines = log_content.split('\n')
                                for line in lines:
                                    if line.startswith("Return code:"):
                                        return_code = line.split(":", 1)[1].strip()
                                        if return_code != "0":
                                            error_msg = f"Process failed with return code {return_code}"
                                        break
                    except Exception:
                        pass

                # Infer audio file path for failed episodes too
                inferred_audio = ""
                possible_paths = [
                    f"data/final_dataset/voice/mrcr_voice_episodes_audio/{episode_id}.wav",
                    f"data/final_dataset/voice/aime_voice_episodes_audio/{episode_id}.wav",
                    f"data/final_dataset/voice/simpleqa_voice_episodes_audio/{episode_id}.wav",
                    f"data/final_dataset/voice/gpqa_voice_episodes_audio/{episode_id}.wav",
                    f"data/final_dataset/voice/gpqa_diamond_voice_episodes_audio/{episode_id}.wav",
                    f"data/final_dataset/voice/browsecomp_voice_episodes_audio/{episode_id}.wav",
                ]
                for path in possible_paths:
                    if Path(path).exists():
                        inferred_audio = path
                        break

                results.append({
                    "episode_id": episode_id,
                    "success": False,
                    "audio_file": inferred_audio,
                    "episode_json": "",
                    "output_dir": str(episode_dir),
                    "duration": 0,
                    "error": error_msg
                })

        successful = [r for r in results if r["success"]]
        failed = [r for r in results if not r["success"]]

        logger.info(f"Found {len(episode_dirs)} episodes: {len(successful)} successful, {len(failed)} failed")

        # Create a mock report structure
        mock_data = {
            "summary": {
                "total_files": len(results),
                "successful": len(successful),
                "failed": len(failed)
            },
            "results": results
        }

        return {"type": "scan", "data": mock_data, "report_path": "", "base_path": str(directory)}

    # Build jobs either from resume report or by scanning voice_dir
    jobs = []
    skipped = []
    benchmark_counts = {}
    processed_count = 0

    # Preload episodes for fast lookup
    episode_map = preload_episodes(dataset_dir)
    
    if args.resume_from is not None:
        logger.info(f"Resume requested from: {args.resume_from}")
        try:
            report_info = _load_resume_report(args.resume_from)
        except Exception as e:
            logger.error(f"Failed to load resume report: {e}")
            return 1
        data = report_info["data"]
        report_type = report_info["type"]

        # When resuming, use the resume directory as the output directory
        resume_dir = Path(args.resume_from)
        if report_type == "scan":
            # For directory scans, the resume_from path IS the output directory
            benchmark_output_dir = resume_dir
            logger.info(f"Using resume directory as output: {benchmark_output_dir}")
        else:
            # For report files, use the directory containing the report
            benchmark_output_dir = resume_dir.parent if resume_dir.is_file() else resume_dir
            logger.info(f"Using resume base directory as output: {benchmark_output_dir}")

        # Determine results list and default benchmark name
        if report_type == 'overall':
            results = data.get("results", [])
        elif report_type == 'scan':
            results = data.get("results", [])
            logger.info(f"Directory scan completed, found {len(results)} episodes")
        else:
            results = data.get("results", [])
        logger.info(f"Resume source: {report_type}, contains {len(results)} results")

        # Helper to check if previous output had audio
        def has_audio(prev_out_dir: Optional[str]) -> bool:
            if not prev_out_dir:
                return False
            try:
                return (Path(prev_out_dir) / "output.wav").exists()
            except Exception:
                return False

        for res in results:
            try:
                prev_success = bool(res.get("success", False))
                prev_out_dir = res.get("output_dir")
                # Treat missing audio as failure
                prev_has_audio = has_audio(prev_out_dir)
                # In resume mode, only include failed episodes by default
                # This means episodes without response.json or output.wav
                should_include = (not prev_success) or (not prev_has_audio)
                if not should_include:
                    continue

                audio_path_str = res.get("audio_file")
                episode_path_str = res.get("episode_json")
                benchmark = res.get("benchmark")
                # If benchmark is unknown or missing, try to detect from audio path
                if not benchmark or benchmark == "unknown":
                    if audio_path_str:
                        benchmark = detect_benchmark_from_audio(Path(audio_path_str))
                    else:
                        benchmark = "unknown"

                audio_file = Path(audio_path_str) if audio_path_str else None
                if not audio_file or not audio_file.exists():
                    # Try to reconstruct from prior output dir name
                    audio_id = None
                    if prev_out_dir:
                        audio_id = Path(prev_out_dir).name
                    if not audio_id and audio_path_str:
                        audio_id = Path(audio_path_str).stem
                    if audio_id:
                        # Search for matching audio under voice_dir
                        matches = list(voice_dir.rglob(f"{audio_id}.wav"))
                        audio_file = matches[0] if matches else None
                if audio_file is None:
                    skipped.append(Path(audio_path_str) if audio_path_str else Path("<unknown>"))
                    continue

                # Re-detect benchmark if it was unknown and we now have an audio file
                if benchmark == "unknown" and audio_file:
                    benchmark = detect_benchmark_from_audio(audio_file)

                episode_json = Path(episode_path_str) if episode_path_str else None
                if episode_json is None or not episode_json.exists():
                    episode_json = find_episode_json_for_audio_fast(audio_file, episode_map)
                    if episode_json is None:
                        skipped.append(audio_file)
                        continue

                jobs.append((audio_file, episode_json, benchmark))
                benchmark_counts[benchmark] = benchmark_counts.get(benchmark, 0) + 1
                processed_count += 1
            except Exception as e:
                logger.warning(f"Skipping one resume item due to error: {e}")
                continue

        if not jobs and not args.continue_all:
            logger.error("No jobs to resume were found from the provided report/path")
            return 1
        logger.info(f"Prepared {len(jobs)} resume jobs across benchmarks: {list(benchmark_counts.keys())}")

        # If --continue-all is specified, also add unprocessed episodes
        if args.continue_all:
            logger.info("Continue-all mode: scanning for unprocessed episodes...")

            # Get list of already processed episode IDs
            processed_episode_ids = set()
            for res in results:
                episode_id = res.get("episode_id") or ""
                if episode_id:
                    processed_episode_ids.add(episode_id)
                # Also extract from output_dir if episode_id is missing
                res_output_dir = res.get("output_dir", "")
                if res_output_dir and not episode_id:
                    dir_name = Path(res_output_dir).name
                    if dir_name.startswith("vera_"):
                        processed_episode_ids.add(dir_name)

            logger.info(f"Already processed {len(processed_episode_ids)} episodes")

            # Scan voice_dir for all episodes and add unprocessed ones
            additional_jobs = 0
            for audio_file in voice_dir.rglob(args.pattern):
                audio_stem = audio_file.stem
                if audio_stem not in processed_episode_ids:
                    # This episode hasn't been processed yet
                    episode_json = find_episode_json_for_audio_fast(audio_file, episode_map)
                    if episode_json:
                        benchmark = detect_benchmark_from_audio(audio_file)
                        jobs.append((audio_file, episode_json, benchmark))
                        benchmark_counts[benchmark] = benchmark_counts.get(benchmark, 0) + 1
                        additional_jobs += 1

                        # Check limit
                        if args.limit and len(jobs) >= args.limit:
                            break

            logger.info(f"Added {additional_jobs} unprocessed episodes to the queue")
            logger.info(f"Total jobs to process: {len(jobs)}")
    else:
        # Process audio files one by one using iterator (don't load all into memory)
        audio_files_iter = voice_dir.rglob(args.pattern)
        logger.info("Starting to process audio files...")
        for audio_file in audio_files_iter:
            # Check limit
            if args.limit and processed_count >= args.limit:
                break
            processed_count += 1
            if processed_count % 100 == 0:
                logger.info(f"Processed {processed_count} files...")
            episode_json = find_episode_json_for_audio_fast(audio_file, episode_map)
            if episode_json:
                benchmark = detect_benchmark_from_audio(audio_file)
                jobs.append((audio_file, episode_json, benchmark))
                benchmark_counts[benchmark] = benchmark_counts.get(benchmark, 0) + 1
            else:
                skipped.append(audio_file)
                logger.warning(f"No episode JSON found for {audio_file.name}")
        logger.info(f"Prepared {len(jobs)} jobs, skipped {len(skipped)} files")
        logger.info(f"Benchmarks found: {benchmark_counts}")
        if not jobs:
            logger.error("No valid audio-episode pairs found!")
            return 1
    
    # Create rate limiter
    rate_limiter = RateLimiter(max_requests_per_minute=args.rpm)
    
    # Create organized output structure: output_dir/model_name/benchmark/timestamp/
    batch_start = datetime.now()
    batch_id = batch_start.strftime("%Y%m%d_%H%M%S")
    
    # Group jobs by benchmark for organized processing
    jobs_by_benchmark = {}
    for audio_file, episode_json, benchmark in jobs:
        if benchmark not in jobs_by_benchmark:
            jobs_by_benchmark[benchmark] = []
        jobs_by_benchmark[benchmark].append((audio_file, episode_json, benchmark))
    
    logger.info(f"Starting batch processing: {batch_id}")
    logger.info(f"Max workers: {args.max_workers}")
    logger.info(f"Rate limit: {args.rpm} RPM")
    
    # Process jobs organized by benchmark
    all_results = []
    
    for benchmark, benchmark_jobs in jobs_by_benchmark.items():
        logger.info(f"Processing {benchmark} benchmark ({len(benchmark_jobs)} files)")

        # Create benchmark-specific output directory
        if args.resume_from is not None:
            # When resuming, use the existing resume directory
            benchmark_dir = benchmark_output_dir
            logger.info(f"Resuming in existing directory: {benchmark_dir}")
        else:
            # Normal mode: create new timestamped directory
            benchmark_dir = output_dir / str(args.model_name) / str(benchmark) / str(batch_id)
            benchmark_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Output directory: {benchmark_dir}")
        
        # Process jobs for this benchmark
        benchmark_results = []
        
        if args.max_workers == 1:
            # Sequential processing
            for i, (audio_file, episode_json, bench) in enumerate(benchmark_jobs, 1):
                logger.info(f"Processing {benchmark} {i}/{len(benchmark_jobs)}: {audio_file.name}")
                result = run_single_inference(
                    audio_file, episode_json, benchmark_dir, 
                    azure_script, rate_limiter, benchmark, args.max_retries
                )
                result["benchmark"] = benchmark
                benchmark_results.append(result)
                all_results.append(result)
        else:
            # Parallel processing for this benchmark
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
                future_to_job = {
                    executor.submit(
                        run_single_inference, 
                        audio_file, episode_json, benchmark_dir, 
                        azure_script, rate_limiter, benchmark, args.max_retries
                    ): (audio_file, episode_json, benchmark) 
                    for audio_file, episode_json, benchmark in benchmark_jobs
                }
            
                for future in concurrent.futures.as_completed(future_to_job):
                    audio_file, episode_json, bench = future_to_job[future]
                    try:
                        result = future.result()
                        result["benchmark"] = benchmark
                        benchmark_results.append(result)
                        all_results.append(result)
                    except Exception as e:
                        logger.error(f"Job failed with exception: {e}")
                        error_result = {
                            "audio_file": str(audio_file),
                            "episode_json": str(episode_json),
                            "output_dir": "",
                            "success": False,
                            "duration": 0,
                            "attempt": 0,
                            "benchmark": benchmark,
                            "error": f"Executor exception: {str(e)}"
                        }
                        benchmark_results.append(error_result)
                        all_results.append(error_result)
        
        # Save benchmark-specific report
        benchmark_report = create_benchmark_report(benchmark, benchmark_results, benchmark_dir, batch_start, args)
        benchmark_report_file = benchmark_dir / "benchmark_report.json"
        with open(benchmark_report_file, 'w') as f:
            json.dump(benchmark_report, f, indent=2)
        logger.info(f"Saved {benchmark} benchmark report: {benchmark_report_file}")
    
    # Overall Summary
    batch_end = datetime.now()
    batch_duration = (batch_end - batch_start).total_seconds()
    
    successful = [r for r in all_results if r["success"]]
    failed = [r for r in all_results if not r["success"]]
    
    logger.info("=== OVERALL BATCH SUMMARY ===")
    logger.info(f"Model: {args.model_name}")
    logger.info(f"Total files processed: {processed_count}")
    logger.info(f"Jobs processed: {len(jobs)}")
    logger.info(f"Successful: {len(successful)}")
    logger.info(f"Failed: {len(failed)}")
    logger.info(f"Skipped: {len(skipped)}")
    logger.info(f"Total time: {batch_duration:.1f}s")
    
    # Summary by benchmark
    logger.info("=== BENCHMARK BREAKDOWN ===")
    for benchmark, count in benchmark_counts.items():
        bench_successful = [r for r in all_results if r.get("benchmark") == benchmark and r["success"]]
        bench_failed = [r for r in all_results if r.get("benchmark") == benchmark and not r["success"]]
        logger.info(f"{benchmark}: {len(bench_successful)}/{count} successful ({len(bench_successful)/count*100:.1f}%)")
    
    if successful:
        avg_duration = sum(r["duration"] for r in successful) / len(successful)
        logger.info(f"Average processing time: {avg_duration:.1f}s per file")
    
    # Save overall batch report
    overall_report = {
        "batch_id": batch_id,
        "model": args.model_name,
        "start_time": batch_start.isoformat(),
        "end_time": batch_end.isoformat(),
        "duration_seconds": batch_duration,
        "config": {
            "voice_dir": str(voice_dir),
            "dataset_dir": str(dataset_dir),
            "output_dir": str(output_dir),
            "model_name": args.model_name,
            "max_workers": args.max_workers,
            "max_retries": args.max_retries,
            "limit": args.limit,
            "pattern": args.pattern,
            "rpm": args.rpm
        },
        "benchmarks": benchmark_counts,
        "summary": {
            "total_audio_files": processed_count,
            "jobs_processed": len(jobs),
            "successful": len(successful),
            "failed": len(failed),
            "skipped": len(skipped),
            "success_rate": len(successful) / len(jobs) if jobs else 0,
            "average_duration": sum(r["duration"] for r in successful) / len(successful) if successful else 0
        },
        "benchmark_breakdown": {
            benchmark: {
                "successful": len([r for r in all_results if r.get("benchmark") == benchmark and r["success"]]),
                "failed": len([r for r in all_results if r.get("benchmark") == benchmark and not r["success"]]),
                "total": count,
                "success_rate": len([r for r in all_results if r.get("benchmark") == benchmark and r["success"]]) / count if count > 0 else 0
            }
            for benchmark, count in benchmark_counts.items()
        },
        "results": all_results,
        "skipped_files": [str(f) for f in skipped]
    }
    
    # Save overall report in the main output directory
    overall_report_file = output_dir / args.model_name / f"overall_batch_report_{batch_id}.json"
    overall_report_file.parent.mkdir(parents=True, exist_ok=True)
    with open(overall_report_file, 'w') as f:
        json.dump(overall_report, f, indent=2)
    
    logger.info(f"Overall batch report saved: {overall_report_file}")
    
    # Print failed files
    if failed:
        logger.info("=== FAILED FILES ===")
        for result in failed:
            logger.info(f"Failed: {Path(result['audio_file']).name} - {result['error']}")
    
    return 0 if len(failed) == 0 else 1

if __name__ == "__main__":
    exit(main())
