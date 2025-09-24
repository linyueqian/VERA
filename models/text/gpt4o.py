"""
GPT-4o OpenAI Browse Adapter for VERA
Uses OpenAI Responses API with web_search_preview tool for browsecomp benchmark
"""

import os
import json
import time
import httpx
from typing import Dict, Any, List
from pathlib import Path

from ..shared.timing_utils import (
    make_timed_api_request,
    create_turn_result,
    create_standardized_episode_result,
    create_standardized_batch_result
)
from ..shared.base_adapter import TextAdapter, ModelConfig


class GPT4oOpenAIBrowseAdapter(TextAdapter):
    """OpenAI GPT-4o adapter using web_search_preview for browsecomp."""

    def __init__(self, api_key: str, api_base: str = "https://api.openai.com", api_version: str = "2025-02-01-preview"):
        config = ModelConfig(model_name="gpt-4o")
        super().__init__(config, api_key)
        self.api_base = api_base.rstrip('/')
        self.api_version = api_version

    async def process_episodes_batch(self, episodes: List[Dict[str, Any]], output_dir: str, max_concurrent: int = 16) -> Dict[str, Any]:
        """Batch process episodes concurrently."""
        print(f"[GPT-4o OpenAI Browse] Batch processing {len(episodes)} episodes (max {max_concurrent} concurrent)")
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        start = time.time()
        import asyncio
        semaphore = asyncio.Semaphore(max_concurrent)

        async def run_one(ep):
            async with semaphore:
                import asyncio
                return await asyncio.to_thread(self.process_episode, ep, output_dir)

        tasks = [run_one(ep) for ep in episodes]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        processed = []
        for i, r in enumerate(results):
            if isinstance(r, Exception):
                processed.append({
                    'episode_id': episodes[i].get('id', f'episode_{i}'),
                    'error': str(r),
                    'success': False
                })
            else:
                processed.append(r)

        duration = time.time() - start
        batch = create_standardized_batch_result(
            f"{self.model_name}_openai_browse",
            episodes,
            processed,
            duration,
            max_concurrent
        )
        batch_file = output_path / f"gpt4o_openai_browse_batch_{int(time.time())}.json"
        with open(batch_file, 'w') as f:
            json.dump(batch, f, indent=2)
        print(f"[GPT-4o OpenAI Browse] Batch completed: {batch['successful']}/{len(episodes)} successful")
        return batch

    def process_episode(self, episode_data: Dict[str, Any], output_dir: str) -> Dict[str, Any]:
        episode_id = episode_data.get('id', 'unknown')
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        session_start = time.time()
        turns_results: List[Dict[str, Any]] = []
        total_tokens = 0

        for turn_idx, turn in enumerate(episode_data.get('turns', [])):
            if turn.get('role') != 'user':
                continue
            turn_start = time.time()
            prompt = self._prepare_prompt(turn, episode_data, turn_idx)
            response_data = self._call_openai_responses(prompt)
            turn_time_ms = (time.time() - turn_start) * 1000

            model_metadata = {
                'model': self.model_name,
                'provider': 'openai',
                'response_id': response_data.get('id', '')
            }

            error = response_data.get('error') if 'error' in response_data else None
            turn_result_obj = create_turn_result(
                turn_index=turn_idx,
                user_input=turn.get('text_content', ''),
                response_data=response_data,
                total_turn_time_ms=turn_time_ms,
                model_metadata=model_metadata,
                error=error
            )
            tr = turn_result_obj.to_dict()
            turns_results.append(tr)
            if not error:
                total_tokens += response_data.get('usage', {}).get('total_tokens', 0)

        session_duration = time.time() - session_start
        return create_standardized_episode_result(
            episode_id=episode_id,
            model_name=f"{self.model_name}_openai_browse",
            turn_results=turns_results,
            session_duration=session_duration,
            total_tokens=total_tokens,
            reasoning_data={'responses': [t['model_response'] for t in turns_results]}
        )

    def _prepare_prompt(self, turn: Dict[str, Any], episode_data: Dict[str, Any], turn_idx: int) -> str:
        user_speech = turn.get('text_content', '')
        context_docs = episode_data.get('context_documents', [])
        parts: List[str] = []
        if context_docs:
            parts.append("Context Documents:")
            for i, doc in enumerate(context_docs):
                parts.append(f"Document {i+1}: {doc.get('content','')}")
            parts.append("")
        if turn_idx > 0:
            parts.append("Previous conversation:")
            for prev_idx in range(turn_idx):
                pt = episode_data['turns'][prev_idx]
                role = pt.get('role')
                if role == 'user':
                    parts.append(f"User: {pt.get('text_content','')}")
                elif role == 'assistant':
                    parts.append(f"Assistant: {pt.get('response','')}")
            parts.append("")
        parts.append(f"User: {user_speech}")
        return "\n".join(parts)

    def _make_api_request(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Make API request to OpenAI GPT-4o"""
        if len(messages) == 1 and messages[0].get("role") == "user":
            prompt = messages[0]["content"]
        else:
            # Convert messages to prompt format
            prompt_parts = []
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role == "user":
                    prompt_parts.append(f"User: {content}")
                elif role == "assistant":
                    prompt_parts.append(f"Assistant: {content}")
            prompt = "\n".join(prompt_parts)

        response_data = self._call_openai_responses(prompt)
        if "error" in response_data:
            raise Exception(response_data["error"])

        return response_data.get("output", {}).get("content", "")

    def _call_openai_responses(self, prompt: str) -> Dict[str, Any]:
        url = f"{self.api_base}/v1/responses"
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}
        params = {"api-version": self.api_version}
        payload = {
            "input": [{"role": "user", "content": prompt}],
            "model": self.model_name,
            "tools": [{"type": "web_search_preview", "search_context_size": "high"}],
            "truncation": "auto",
            "max_output_tokens": 8192
        }
        try:
            with httpx.Client(timeout=120.0) as client:
                resp = client.post(url, headers=headers, params=params, json=payload)
                resp.raise_for_status()
                return resp.json()
        except httpx.HTTPStatusError as e:
            return {"error": f"HTTP {e.response.status_code}: {e.response.text}", "status_code": e.response.status_code}
        except httpx.TimeoutException:
            return {"error": "Request timed out"}
        except Exception as e:
            return {"error": f"Unexpected error: {e}"}
