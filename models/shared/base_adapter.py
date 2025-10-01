"""
Base adapter interface for VERA model implementations
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from pathlib import Path
import asyncio


@dataclass
class ModelConfig:
    """Base configuration for model adapters"""
    model_name: str
    temperature: float = 0.0
    max_tokens: int = 4096
    timeout: float = 300.0
    max_concurrent: int = 16


class BaseAdapter(ABC):
    """Base class for all VERA model adapters"""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.model_name = config.model_name

    @abstractmethod
    def process_episode(self, episode: Dict[str, Any], output_dir: str) -> Dict[str, Any]:
        """
        Process a single episode.

        Args:
            episode: Episode data containing turns and metadata
            output_dir: Directory to save outputs

        Returns:
            Standardized episode result
        """
        pass

    async def process_episodes_batch(
        self,
        episodes: List[Dict[str, Any]],
        output_dir: str,
        max_concurrent: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Process multiple episodes concurrently.

        Args:
            episodes: List of episodes to process
            output_dir: Directory to save outputs
            max_concurrent: Maximum concurrent episodes (uses config default if None)

        Returns:
            Standardized batch result
        """
        from .timing_utils import create_standardized_batch_result
        import time

        max_concurrent = max_concurrent or self.config.max_concurrent
        print(f"[{self.model_name}] Batch processing {len(episodes)} episodes (max {max_concurrent} concurrent)")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        start_time = time.time()
        semaphore = asyncio.Semaphore(max_concurrent)

        async def process_one(episode):
            async with semaphore:
                return await asyncio.to_thread(self.process_episode, episode, output_dir)

        tasks = [process_one(ep) for ep in episodes]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append({
                    "episode_id": episodes[i].get("id", f"episode_{i}"),
                    "turns": [],
                    "total_time": 0,
                    "success": False,
                    "error": str(result),
                    "metadata": {}
                })
            else:
                processed_results.append(result)

        total_time = time.time() - start_time
        return create_standardized_batch_result(
            episodes=processed_results,
            total_time=total_time,
            model_name=self.model_name
        )


class TextAdapter(BaseAdapter):
    """Base class for text-based model adapters"""

    def __init__(self, config: ModelConfig, api_key: str):
        super().__init__(config)
        self.api_key = api_key

    @abstractmethod
    def _make_api_request(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Make API request to text model"""
        pass


class VoiceAdapter(BaseAdapter):
    """Base class for voice model adapters"""

    def __init__(self, config: ModelConfig):
        super().__init__(config)

    @abstractmethod
    def _process_audio_input(self, audio_path: str, text_prompt: str) -> str:
        """Process audio input with text prompt"""
        pass


class RealtimeAdapter(BaseAdapter):
    """Base class for realtime model adapters"""

    def __init__(self, config: ModelConfig):
        super().__init__(config)

    @abstractmethod
    def _establish_connection(self) -> Any:
        """Establish connection to realtime model"""
        pass

    @abstractmethod
    def _send_audio_chunk(self, connection: Any, audio_data: bytes) -> None:
        """Send audio chunk to model"""
        pass

    @abstractmethod
    def _receive_response(self, connection: Any) -> str:
        """Receive response from model"""
        pass