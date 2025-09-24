"""
General Text Model Evaluation Script for VERA Datasets
Supports GPT-4o, GPT-5 Instant, GPT-5 Thinking with async processing by default
"""

import os
import sys
import json
import time
import asyncio
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
from dotenv import load_dotenv
import yaml

# Load environment variables from .env file
load_dotenv()

# Add models to path
models_path = str(Path(__file__).parent.parent.parent / 'models' / 'text')
if models_path not in sys.path:
    sys.path.insert(0, models_path)

# Add project root to path for absolute imports
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from gpt4o_azure import GPT4oAzureAdapter
from gpt5_instant_azure import GPT5InstantAzureAdapter
from gpt5_thinking_azure import GPT5ThinkingAzureAdapter
from gpt4o_openai_browse import GPT4oOpenAIBrowseAdapter
from gpt5_openai_browse import GPT5OpenAIBrowseAdapter
from gemini_25_pro_browse import Gemini25ProBrowseAdapter
from gemini_25_flash_browse import Gemini25FlashBrowseAdapter


class TextModelEvaluator:
    """General evaluator that can work with different text model adapters"""
    
    def __init__(self):
        # Load canonical config.yaml and overlay with .env for secrets
        self.config = self._load_config()
        self.models = {
            'gpt4o': self._create_gpt4o_adapter,
            'gpt5-instant': self._create_gpt5_instant_adapter,
            'gpt5-thinking': self._create_gpt5_thinking_adapter,
            'gemini-2.5-pro': self._create_gemini_25_pro_adapter,
            'gemini-2.5-flash': self._create_gemini_25_flash_adapter
        }
        self._current_dataset_name = None

    def set_dataset_context(self, dataset_name: str):
        self._current_dataset_name = dataset_name

    def _load_config(self) -> Dict[str, Any]:
        """Load config.yaml from project root; return empty dict if missing."""
        cfg_path = Path(__file__).parent.parent.parent / 'config.yaml'
        if not cfg_path.exists():
            return {}
        try:
            with open(cfg_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
        except Exception:
            return {}

    # --- Config helpers with .env overlay ---
    def _get_openai_api_key(self) -> Optional[str]:
        return os.getenv('OPENAI_API_KEY') or (self.config.get('api_keys', {}) or {}).get('openai_api_key')

    def _get_azure_api_key(self) -> Optional[str]:
        return (
            os.getenv('AZURE_OPENAI_API_KEY')
            or (self.config.get('azure', {}) or {}).get('api_key')
            or (self.config.get('api_keys', {}) or {}).get('azure_api_key')
        )

    def _get_azure_endpoint(self) -> Optional[str]:
        ep = os.getenv('AZURE_OPENAI_ENDPOINT') or (self.config.get('azure', {}) or {}).get('http_base_url')
        if ep:
            return ep.rstrip('/')
        return None

    def _get_gemini_api_key(self) -> Optional[str]:
        return os.getenv('GEMINI_API_KEY') or (self.config.get('api_keys', {}) or {}).get('gemini_api_key')
        
    def _create_gpt4o_adapter(self):
        """Create GPT-4o Azure adapter"""
        if (self._current_dataset_name or '').lower().startswith('browsecomp'):
            openai_key = self._get_openai_api_key()
            if not openai_key:
                raise ValueError("browsecomp requires OPENAI_API_KEY for OpenAI browse adapters")
            return GPT4oOpenAIBrowseAdapter(api_key=openai_key)
        azure_endpoint = self._get_azure_endpoint()
        api_key = self._get_azure_api_key()
        if not azure_endpoint or not api_key:
            raise ValueError("GPT-4o requires AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY environment variables")
        return GPT4oAzureAdapter(azure_endpoint, api_key)
        
    def _create_gpt5_instant_adapter(self):
        """Create GPT-5 Instant Azure adapter"""
        if (self._current_dataset_name or '').lower().startswith('browsecomp'):
            openai_key = self._get_openai_api_key()
            if not openai_key:
                raise ValueError("browsecomp requires OPENAI_API_KEY for OpenAI browse adapters")
            # instant variant => low reasoning effort
            return GPT5OpenAIBrowseAdapter(api_key=openai_key, reasoning_effort='low', reasoning_summary='auto')
        azure_endpoint = self._get_azure_endpoint()
        api_key = self._get_azure_api_key()
        if not azure_endpoint or not api_key:
            raise ValueError("GPT-5 Instant requires AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY environment variables")
        return GPT5InstantAzureAdapter(azure_endpoint, api_key)
        
    def _create_gpt5_thinking_adapter(self):
        """Create GPT-5 Thinking Azure adapter"""
        if (self._current_dataset_name or '').lower().startswith('browsecomp'):
            openai_key = self._get_openai_api_key()
            if not openai_key:
                raise ValueError("browsecomp requires OPENAI_API_KEY for OpenAI browse adapters")
            # thinking variant => high reasoning effort
            return GPT5OpenAIBrowseAdapter(api_key=openai_key, reasoning_effort='high', reasoning_summary='detailed')
        azure_endpoint = self._get_azure_endpoint()
        api_key = self._get_azure_api_key()
        if not azure_endpoint or not api_key:
            raise ValueError("GPT-5 Thinking requires AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY environment variables")
        return GPT5ThinkingAzureAdapter(azure_endpoint, api_key)

    def _create_gemini_25_pro_adapter(self):
        """Create Gemini 2.5 Pro adapter with browse support"""
        api_key = self._get_gemini_api_key()
        if not api_key:
            raise ValueError("Gemini 2.5 Pro requires GEMINI_API_KEY environment variable")
        return Gemini25ProBrowseAdapter(api_key=api_key)

    def _create_gemini_25_flash_adapter(self):
        """Create Gemini 2.5 Flash adapter with browse support"""
        api_key = self._get_gemini_api_key()
        if not api_key:
            raise ValueError("Gemini 2.5 Flash requires GEMINI_API_KEY environment variable")
        return Gemini25FlashBrowseAdapter(api_key=api_key)

    def load_dataset(self, dataset_path: str) -> Dict[str, Any]:
        """Load a VERA dataset JSON file"""
        with open(dataset_path, 'r', encoding='utf-8') as f:
            return json.load(f)
            
    def create_output_dir(self, model_name: str, dataset_name: str) -> str:
        """Create timestamped output directory"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"test_output/{model_name}_{dataset_name}_{timestamp}"
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        return output_dir

    def get_completed_episodes(self, output_dir: str) -> set:
        """Get set of episode IDs that have been completed in output directory"""
        completed = set()
        output_path = Path(output_dir)

        if not output_path.exists():
            return completed

        # Look for JSON result files
        for json_file in output_path.glob("*.json"):
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)

                # Check if it's an individual episode result
                if 'episode_id' in data:
                    completed.add(data['episode_id'])

                # Check if it's a batch result with individual episodes
                elif 'results' in data:
                    for result in data['results']:
                        if isinstance(result, dict) and 'episode_id' in result:
                            completed.add(result['episode_id'])

            except (json.JSONDecodeError, KeyError):
                continue

        return completed

    def filter_episodes_for_resume(self, episodes: List[Dict[str, Any]], output_dir: str) -> List[Dict[str, Any]]:
        """Filter episodes to skip already completed ones"""
        completed_ids = self.get_completed_episodes(output_dir)

        if not completed_ids:
            print("No completed episodes found, processing all episodes")
            return episodes

        print(f"Found {len(completed_ids)} completed episodes, skipping them")

        remaining_episodes = []
        for episode in episodes:
            episode_id = episode.get('id', '')
            if episode_id not in completed_ids:
                remaining_episodes.append(episode)
            else:
                print(f"Skipping completed episode: {episode_id}")

        print(f"Remaining episodes to process: {len(remaining_episodes)}/{len(episodes)}")
        return remaining_episodes
        
    async def run_evaluation(self, model_name: str, dataset_path: str,
                            max_episodes: Optional[int] = None,
                            max_concurrent: int = 16,
                            resume_from: Optional[str] = None) -> Dict[str, Any]:
        """Run evaluation with async processing by default"""
        print(f"Loading dataset: {dataset_path}")
        dataset = self.load_dataset(dataset_path)
        episodes = dataset.get('episodes', [])

        dataset_name = Path(dataset_path).stem.replace('_voice_episodes', '')

        # Handle resume functionality
        if resume_from:
            if not Path(resume_from).exists():
                raise ValueError(f"Resume directory does not exist: {resume_from}")

            print(f"Resuming from: {resume_from}")
            output_dir = resume_from

            # Filter out already completed episodes
            episodes = self.filter_episodes_for_resume(episodes, output_dir)

            if not episodes:
                print("All episodes already completed!")
                return {'message': 'All episodes already completed', 'skipped': True}

        else:
            # Create new output directory
            output_dir = self.create_output_dir(model_name, dataset_name)

        if max_episodes:
            episodes = episodes[:max_episodes]
            print(f"Limited to {max_episodes} episodes")

        print(f"Creating model adapter: {model_name}")
        self.set_dataset_context(dataset_name)
        adapter = self.models[model_name]()
        
        print(f"Starting async evaluation with {len(episodes)} episodes")
        start_time = time.time()
        
        # All models now support async batch processing
        results = await adapter.process_episodes_batch(episodes, output_dir, max_concurrent)
                    
        end_time = time.time()
        duration = end_time - start_time
        
        # Save summary
        summary = {
            'model': model_name,
            'dataset': dataset_name,
            'dataset_path': dataset_path,
            'output_directory': output_dir,
            'total_episodes': len(episodes),
            'processed': results.get('processed', 0),
            'successful': results.get('successful', 0),
            'failed': results.get('failed', 0),
            'duration_seconds': duration,
            'episodes_per_second': len(episodes) / duration if duration > 0 else 0,
            'timestamp': datetime.now().isoformat(),
            'max_concurrent': max_concurrent,
            'async_processing': True
        }
        
        summary_path = Path(output_dir) / 'evaluation_summary.json'
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
            
        print(f"\nEvaluation completed!")
        print(f"Model: {model_name}")
        print(f"Dataset: {dataset_name}")
        print(f"Episodes: {summary['successful']}/{summary['total_episodes']} successful")
        print(f"Duration: {duration:.2f}s ({summary['episodes_per_second']:.2f} episodes/sec)")
        print(f"Output: {output_dir}")
        
        return summary


def main():
    parser = argparse.ArgumentParser(description='Evaluate text models on VERA datasets')
    parser.add_argument('model', choices=['gpt4o', 'gpt5-instant', 'gpt5-thinking', 'gemini-2.5-pro', 'gemini-2.5-flash'],
                       help='Text model to evaluate')
    parser.add_argument('dataset', help='Path to dataset JSON file')
    parser.add_argument('--max-episodes', type=int, help='Maximum number of episodes to process')
    parser.add_argument('--max-concurrent', type=int, default=16, 
                       help='Maximum concurrent requests')
    
    args = parser.parse_args()
    
    # Validate dataset path
    if not Path(args.dataset).exists():
        print(f"Error: Dataset file not found: {args.dataset}")
        return 1
        
    evaluator = TextModelEvaluator()
    
    try:
        summary = asyncio.run(evaluator.run_evaluation(
            args.model, args.dataset, args.max_episodes, args.max_concurrent
        ))
        return 0
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
