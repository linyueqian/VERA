"""
Batch evaluation script to run multiple models on multiple datasets
Convenient wrapper around run_evaluation.py
"""

import os
import sys
import asyncio
import argparse
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from run_evaluation import TextModelEvaluator

# Load environment variables from .env file
load_dotenv()

def get_available_datasets():
    """Get list of available dataset files"""
    dataset_dir = Path(__file__).parent.parent.parent / 'data' / 'final_dataset' / 'text'
    if not dataset_dir.exists():
        return []
    return [f.stem.replace('_voice_episodes', '') for f in dataset_dir.glob('*_voice_episodes.json')]

def get_dataset_path(dataset_name):
    """Get full path to dataset file"""
    dataset_dir = Path(__file__).parent.parent.parent / 'data' / 'final_dataset' / 'text'
    return dataset_dir / f'{dataset_name}_voice_episodes.json'

async def main():
    parser = argparse.ArgumentParser(description='Batch evaluate text models on VERA datasets')
    parser.add_argument('--models', nargs='+',
                       choices=['gpt4o', 'gpt5-instant', 'gpt5-thinking', 'gemini-2.5-pro', 'gemini-2.5-flash'],
                       default=['gpt4o', 'gpt5-instant', 'gpt5-thinking', 'gemini-2.5-pro', 'gemini-2.5-flash'],
                       help='Models to evaluate')
    parser.add_argument('--datasets', nargs='+',
                       choices=get_available_datasets(),
                       default=get_available_datasets(),
                       help='Datasets to evaluate on')
    parser.add_argument('--max-episodes', type=int, 
                       help='Maximum episodes per dataset')
    parser.add_argument('--max-concurrent', type=int, default=16,
                       help='Maximum concurrent requests')
    parser.add_argument('--sequential', action='store_true',
                       help='Run evaluations sequentially instead of in parallel')
    parser.add_argument('--resume-from', type=str,
                       help='Resume from existing output directory (e.g., test_output/gemini-2.5-pro_browsecomp_20250917_215054)')

    args = parser.parse_args()
    
    print("Available datasets:", get_available_datasets())
    print(f"Selected models: {args.models}")
    print(f"Selected datasets: {args.datasets}")
    
    evaluator = TextModelEvaluator()
    
    # Create combinations of model and dataset
    tasks = []
    for model in args.models:
        for dataset in args.datasets:
            dataset_path = get_dataset_path(dataset)
            if not dataset_path.exists():
                print(f"Warning: Dataset not found: {dataset_path}")
                continue

            print(f"Queuing: {model} on {dataset}")
            task = evaluator.run_evaluation(
                model, str(dataset_path),
                args.max_episodes, args.max_concurrent,
                resume_from=args.resume_from
            )
            tasks.append((model, dataset, task))
    
    if not tasks:
        print("No valid model/dataset combinations found")
        return 1
    
    print(f"\nStarting {len(tasks)} evaluation tasks...")
    start_time = datetime.now()
    
    if args.sequential:
        # Run sequentially
        for model, dataset, task in tasks:
            print(f"\n--- Running {model} on {dataset} ---")
            try:
                await task
                print(f"✓ Completed {model} on {dataset}")
            except Exception as e:
                print(f"✗ Failed {model} on {dataset}: {e}")
    else:
        # Run in parallel
        results = await asyncio.gather(*[task for _, _, task in tasks], return_exceptions=True)
        
        # Print results
        for i, (model, dataset, _) in enumerate(tasks):
            if isinstance(results[i], Exception):
                print(f"✗ Failed {model} on {dataset}: {results[i]}")
            else:
                print(f"✓ Completed {model} on {dataset}")
    
    end_time = datetime.now()
    total_duration = (end_time - start_time).total_seconds()
    
    print(f"\nBatch evaluation completed in {total_duration:.2f} seconds")
    print(f"Results saved to test_output/ directory")

if __name__ == "__main__":
    exit(asyncio.run(main()))