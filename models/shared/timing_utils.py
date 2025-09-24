"""
Shared timing utilities for model adapters
"""

import time
import json
from typing import Dict, Any, List, Optional
from pathlib import Path


def make_timed_api_request(request_func, *args, **kwargs) -> Dict[str, Any]:
    """
    Execute an API request with timing information.

    Args:
        request_func: The function to call
        *args, **kwargs: Arguments to pass to the function

    Returns:
        Dict containing timing info and result
    """
    start_time = time.time()
    try:
        result = request_func(*args, **kwargs)
        end_time = time.time()
        return {
            "result": result,
            "timing": {
                "start_time": start_time,
                "end_time": end_time,
                "duration": end_time - start_time
            },
            "success": True
        }
    except Exception as e:
        end_time = time.time()
        return {
            "result": None,
            "timing": {
                "start_time": start_time,
                "end_time": end_time,
                "duration": end_time - start_time
            },
            "success": False,
            "error": str(e)
        }


def create_turn_result(
    turn_index: int,
    prompt: str,
    response: str,
    timing: Dict[str, float],
    success: bool = True,
    error: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Create a standardized turn result.

    Args:
        turn_index: Index of the turn
        prompt: The input prompt
        response: The model response
        timing: Timing information
        success: Whether the turn was successful
        error: Error message if any
        metadata: Additional metadata

    Returns:
        Standardized turn result dict
    """
    return {
        "turn_index": turn_index,
        "prompt": prompt,
        "response": response,
        "timing": timing,
        "success": success,
        "error": error,
        "metadata": metadata or {}
    }


def create_standardized_episode_result(
    episode_id: str,
    turns: List[Dict[str, Any]],
    total_time: float,
    success: bool = True,
    error: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Create a standardized episode result.

    Args:
        episode_id: Unique episode identifier
        turns: List of turn results
        total_time: Total processing time
        success: Whether the episode was successful
        error: Error message if any
        metadata: Additional metadata

    Returns:
        Standardized episode result dict
    """
    return {
        "episode_id": episode_id,
        "turns": turns,
        "total_time": total_time,
        "success": success,
        "error": error,
        "metadata": metadata or {},
        "num_turns": len(turns),
        "successful_turns": sum(1 for turn in turns if turn.get("success", True))
    }


def create_standardized_batch_result(
    episodes: List[Dict[str, Any]],
    total_time: float,
    model_name: str,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Create a standardized batch result.

    Args:
        episodes: List of episode results
        total_time: Total batch processing time
        model_name: Name of the model used
        metadata: Additional metadata

    Returns:
        Standardized batch result dict
    """
    successful_episodes = sum(1 for ep in episodes if ep.get("success", True))
    total_turns = sum(ep.get("num_turns", 0) for ep in episodes)
    successful_turns = sum(ep.get("successful_turns", 0) for ep in episodes)

    return {
        "model_name": model_name,
        "episodes": episodes,
        "summary": {
            "total_episodes": len(episodes),
            "successful_episodes": successful_episodes,
            "total_turns": total_turns,
            "successful_turns": successful_turns,
            "success_rate": successful_episodes / len(episodes) if episodes else 0,
            "turn_success_rate": successful_turns / total_turns if total_turns > 0 else 0
        },
        "total_time": total_time,
        "average_time_per_episode": total_time / len(episodes) if episodes else 0,
        "metadata": metadata or {},
        "timestamp": time.time()
    }


def save_results(results: Dict[str, Any], output_path: Path, indent: int = 2) -> None:
    """
    Save results to JSON file.

    Args:
        results: Results dictionary to save
        output_path: Path to save the results
        indent: JSON indentation level
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=indent, ensure_ascii=False)