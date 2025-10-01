"""
Gemini 2.5 Flash Browse Adapter for VERA
Uses Google's Gemini API with google_search tool for browsecomp benchmark
Supports MRCR (chat history) and web search capabilities
"""

import os
import json
import time
import httpx
import asyncio
from typing import Dict, Any, List, Optional
from pathlib import Path
from google import genai
from google.genai import types

from ..shared.timing_utils import (
    create_turn_result,
    create_standardized_episode_result,
    create_standardized_batch_result,
)


class Gemini25FlashBrowseAdapter:
    """Gemini 2.5 Flash adapter with google_search tool for browsecomp."""

    def __init__(self, api_key: str, model_name: str = "gemini-2.5-flash"):
        self.api_key = api_key
        self.model_name = model_name
        self.client = genai.Client(api_key=api_key)

        # Configure the grounding tool for web search
        self.grounding_tool = types.Tool(
            google_search=types.GoogleSearch()
        )

        # Config with search enabled (for browsecomp)
        self.config_with_search = types.GenerateContentConfig(
            tools=[self.grounding_tool],
            temperature=0.7,
            top_p=0.95,
            max_output_tokens=8192
        )

        # Config without search (for other benchmarks)
        self.config_no_search = types.GenerateContentConfig(
            temperature=0.7,
            top_p=0.95,
            max_output_tokens=8192
        )

    async def process_episodes_batch(self, episodes: List[Dict[str, Any]], output_dir: str,
                                    max_concurrent: int = 16) -> Dict[str, Any]:
        """Process multiple episodes concurrently with search capabilities"""

        print(f"[Gemini 2.5 Flash Browse] Batch processing {len(episodes)} episodes (max {max_concurrent} concurrent)")
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        start = time.time()
        semaphore = asyncio.Semaphore(max_concurrent)

        async def process_one(ep):
            async with semaphore:
                # Run synchronous process_episode in thread pool
                return await asyncio.to_thread(self.process_episode, ep, output_dir)

        tasks = [process_one(ep) for ep in episodes]
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
            episodes=processed,
            total_time=duration,
            model_name=f"{self.model_name}_gemini_browse",
            metadata={"max_concurrent": max_concurrent},
        )

        batch_file = output_path / f"gemini_25_flash_browse_batch_{int(time.time())}.json"
        with open(batch_file, 'w') as f:
            json.dump(batch, f, indent=2)

        print(
            f"[Gemini 2.5 Flash Browse] Batch completed: "
            f"{batch['summary']['successful_episodes']}/{batch['summary']['total_episodes']} successful"
        )
        return batch

    def process_episode(self, episode_data: Dict[str, Any], output_dir: str) -> Dict[str, Any]:
        """Process a single episode with MRCR chat history and search support"""

        episode_id = episode_data.get('id', 'unknown')
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        print(f"[Gemini 2.5 Flash Browse] Processing episode {episode_id}")

        session_start = time.time()
        turns_results: List[Dict[str, Any]] = []
        total_tokens = 0

        # Initialize chat session for MRCR history tracking
        chat = self.client.chats.create(model=self.model_name)

        for turn_idx, turn in enumerate(episode_data.get('turns', [])):
            if turn.get('role') != 'user':
                continue

            turn_start = time.time()

            # Prepare prompt with context documents if available
            prompt = self._prepare_prompt_with_context(turn, episode_data, turn_idx)

            # Determine if we should use search based on episode track
            use_search = self._should_use_search(episode_data)

            # Call Gemini with or without search
            response_data = self._call_gemini_with_search(chat, prompt, use_search)

            turn_end = time.time()
            timing = {
                'start_time': turn_start,
                'end_time': turn_end,
                'duration': turn_end - turn_start,
            }

            # Extract metadata including grounding information
            model_metadata = {
                'model': self.model_name,
                'provider': 'google',
                'used_search': self._check_if_search_used(response_data)
            }

            # Add grounding metadata if search was used
            if response_data.get('grounding_metadata'):
                model_metadata['grounding'] = self._extract_grounding_info(response_data['grounding_metadata'])

            # Add timing information including TTFR
            timing_info = response_data.get('timing', {})
            if timing_info:
                model_metadata['timing'] = {
                    'time_to_first_response_ms': timing_info.get('time_to_first_response_ms', 0),
                    'request_start_timestamp': timing_info.get('request_start_timestamp', 0),
                    'first_response_timestamp': timing_info.get('first_response_timestamp', 0)
                }

            error = response_data.get('error') if 'error' in response_data else None

            # Create standardized turn result
            response_text = response_data.get('text', '')
            turn_result = create_turn_result(
                turn_index=turn_idx,
                prompt=prompt,
                response=response_text,
                timing=timing,
                success=(error is None),
                error=error,
                metadata=model_metadata,
            )
            turns_results.append(turn_result)

            if not error:
                # Extract token usage
                usage = response_data.get('usage_metadata', {})
                total_tokens += usage.get('total_token_count', 0)

        session_duration = time.time() - session_start

        # Create standardized episode result
        success = all(t.get('success', True) for t in turns_results)
        result = create_standardized_episode_result(
            episode_id=episode_id,
            turns=turns_results,
            total_time=session_duration,
            success=success,
            metadata={
                'model_name': f"{self.model_name}_gemini_browse",
                'total_tokens': total_tokens,
                'search_usage': self._calculate_search_usage(turns_results),
            },
        )

        # Save results to file
        results_file = output_path / f"gemini_25_flash_browse_{episode_id}.json"
        with open(results_file, 'w') as f:
            json.dump(result, f, indent=2)

        print(f"[Gemini 2.5 Flash Browse] Episode {episode_id} completed in {session_duration:.2f}s")
        return result

    def _prepare_prompt_with_context(self, turn: Dict[str, Any], episode_data: Dict[str, Any],
                                    turn_idx: int) -> str:
        """Prepare prompt with context documents and previous conversation"""

        user_speech = turn.get('text_content', '')
        context_docs = episode_data.get('context_documents', [])

        parts: List[str] = []

        # Add context documents if available (for browsecomp)
        if context_docs:
            parts.append("Context Documents:")
            for i, doc in enumerate(context_docs):
                parts.append(f"Document {i+1}: {doc.get('content', '')}")
            parts.append("")

        # Note: For MRCR, chat history is maintained by the chat session itself
        # We don't need to manually add it here as the Gemini chat API handles it

        # Add current user input with any prefix/postfix
        user_parts = []
        if turn.get('prefix_text'):
            user_parts.append(turn['prefix_text'])
        user_parts.append(user_speech)
        if turn.get('postfix_text'):
            user_parts.append(turn['postfix_text'])

        full_prompt = " ".join(user_parts)

        # If we have context docs, prepend them to the prompt
        if parts:
            full_prompt = "\n".join(parts) + "\n" + full_prompt

        return full_prompt

    def _should_use_search(self, episode_data: Dict[str, Any]) -> bool:
        """Determine if we should use search based on episode track/source"""

        # Check track in episode data
        track = episode_data.get('track', '').lower()

        # Check source in metadata
        source_dataset = episode_data.get('metadata', {}).get('source_dataset', '').lower()

        # Check episode ID (browsecomp episodes have 'browsecomp' in their ID)
        episode_id = episode_data.get('id', '').lower()

        # Use search only for browsecomp benchmark
        return 'browsecomp' in source_dataset or 'browsecomp' in episode_id

    def _call_gemini_with_search(self, chat, prompt: str, use_search: bool = True) -> Dict[str, Any]:
        """Call Gemini API with or without search tool using streaming for true TTFR"""

        try:
            # Record time to first response (TTFR)
            request_start = time.time()

            # Choose config based on whether we need search
            config = self.config_with_search if use_search else self.config_no_search

            # Use streaming to get true TTFR
            response_stream = chat.send_message_stream(
                prompt,
                config=config
            )

            # Collect response text and measure TTFR
            full_text = ""
            first_chunk_time = None
            final_response = None

            for chunk in response_stream:
                if first_chunk_time is None:
                    # Record time when first chunk arrives
                    first_chunk_time = time.time()

                # Accumulate text
                if hasattr(chunk, 'text') and chunk.text:
                    full_text += chunk.text

                # Keep the final response object for metadata
                final_response = chunk

            # Calculate TTFR
            if first_chunk_time is None:
                first_chunk_time = time.time()  # Fallback if no chunks

            ttfr_ms = (first_chunk_time - request_start) * 1000

            # Extract response data
            result = {
                'text': full_text,
                'usage_metadata': self._extract_usage_metadata(final_response) if final_response else {},
                'grounding_metadata': self._extract_grounding_metadata(final_response) if final_response else None,
                'candidates': self._extract_candidates(final_response) if final_response else [],
                'timing': {
                    'request_start_timestamp': request_start,
                    'first_chunk_timestamp': first_chunk_time,
                    'time_to_first_response_ms': ttfr_ms
                }
            }

            return result

        except Exception as e:
            return {
                'error': f"Gemini API error: {str(e)}",
                'text': '',
                'timing': {
                    'time_to_first_response_ms': 0
                }
            }

    def _format_response_for_timing_utils(self, response_data: Dict[str, Any]) -> Dict[str, Any]:
        """Format Gemini response to work with timing_utils expectations"""

        # Create GPT-5 style output format that timing_utils expects
        formatted = {
            'output': [
                {
                    'type': 'message',
                    'content': [
                        {
                            'type': 'output_text',
                            'text': response_data.get('text', '')
                        }
                    ]
                }
            ],
            'usage': {
                'total_tokens': response_data.get('usage_metadata', {}).get('total_token_count', 0),
                'prompt_tokens': response_data.get('usage_metadata', {}).get('prompt_token_count', 0),
                'completion_tokens': response_data.get('usage_metadata', {}).get('candidates_token_count', 0)
            }
        }

        # Add grounding metadata if present
        if response_data.get('grounding_metadata'):
            formatted['grounding_metadata'] = response_data['grounding_metadata']

        return formatted

    def _check_if_search_used(self, response_data: Dict[str, Any]) -> bool:
        """Check if the model used web search for this response"""

        grounding = response_data.get('grounding_metadata')
        if not grounding:
            return False
        return bool(grounding.get('web_search_queries')) or bool(grounding.get('grounding_chunks'))

    def _extract_grounding_info(self, grounding_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key grounding information for logging"""

        return {
            'search_queries': grounding_metadata.get('web_search_queries', []),
            'num_sources': len(grounding_metadata.get('grounding_chunks', [])),
            'has_citations': bool(grounding_metadata.get('grounding_supports'))
        }

    def _extract_usage_metadata(self, response) -> Dict[str, Any]:
        """Extract usage metadata from Gemini response object"""

        try:
            if hasattr(response, 'usage_metadata'):
                metadata = response.usage_metadata
                return {
                    'prompt_token_count': getattr(metadata, 'prompt_token_count', 0),
                    'candidates_token_count': getattr(metadata, 'candidates_token_count', 0),
                    'total_token_count': getattr(metadata, 'total_token_count', 0)
                }
        except:
            pass
        return {}

    def _extract_grounding_metadata(self, response) -> Optional[Dict[str, Any]]:
        """Extract grounding metadata from Gemini response object"""

        try:
            # Check if response has candidates
            if not hasattr(response, 'candidates') or not response.candidates:
                return None

            candidate = response.candidates[0]

            # Check for grounding_metadata attribute
            if not hasattr(candidate, 'grounding_metadata') or candidate.grounding_metadata is None:
                return None

            metadata = candidate.grounding_metadata
            result = {}

            # Extract web search queries
            if hasattr(metadata, 'web_search_queries') and metadata.web_search_queries:
                try:
                    result['web_search_queries'] = list(metadata.web_search_queries)
                except Exception:
                    result['web_search_queries'] = []

            # Extract grounding chunks (sources)
            if hasattr(metadata, 'grounding_chunks') and metadata.grounding_chunks:
                try:
                    chunks = []
                    for chunk in metadata.grounding_chunks:
                        if hasattr(chunk, 'web') and chunk.web:
                            chunks.append({
                                'uri': getattr(chunk.web, 'uri', ''),
                                'title': getattr(chunk.web, 'title', '')
                            })
                    result['grounding_chunks'] = chunks
                except Exception:
                    result['grounding_chunks'] = []

            # Extract grounding supports (citations)
            if hasattr(metadata, 'grounding_supports') and metadata.grounding_supports:
                try:
                    supports = []
                    for support in metadata.grounding_supports:
                        support_dict = {}
                        if hasattr(support, 'segment'):
                            support_dict['segment'] = {
                                'start_index': getattr(support.segment, 'start_index', 0),
                                'end_index': getattr(support.segment, 'end_index', 0),
                                'text': getattr(support.segment, 'text', '')
                            }
                        if hasattr(support, 'grounding_chunk_indices'):
                            support_dict['grounding_chunk_indices'] = list(support.grounding_chunk_indices)
                        supports.append(support_dict)
                    result['grounding_supports'] = supports
                except Exception:
                    result['grounding_supports'] = []

            return result if result else None

        except Exception as e:
            # Only print warning for unexpected errors
            if "NoneType" not in str(e):
                print(f"Warning: Could not extract grounding metadata: {e}")
            return None

    def _extract_candidates(self, response) -> List[Dict[str, Any]]:
        """Extract candidates information from response"""

        candidates = []
        try:
            if hasattr(response, 'candidates'):
                for candidate in response.candidates:
                    cand_dict = {}

                    # Extract finish reason
                    if hasattr(candidate, 'finish_reason'):
                        cand_dict['finish_reason'] = str(candidate.finish_reason)

                    # Extract safety ratings
                    if hasattr(candidate, 'safety_ratings'):
                        ratings = []
                        for rating in candidate.safety_ratings:
                            ratings.append({
                                'category': str(getattr(rating, 'category', '')),
                                'probability': str(getattr(rating, 'probability', ''))
                            })
                        cand_dict['safety_ratings'] = ratings

                    candidates.append(cand_dict)
        except:
            pass

        return candidates

    def _calculate_search_usage(self, turns_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate search usage statistics across all turns"""

        total_searches = 0
        total_sources = 0
        turns_with_search = 0

        for turn in turns_results:
            metadata = turn.get('model_metadata', {})
            if metadata.get('used_search'):
                turns_with_search += 1
                grounding = metadata.get('grounding', {})
                total_searches += len(grounding.get('search_queries', []))
                total_sources += grounding.get('num_sources', 0)

        return {
            'turns_with_search': turns_with_search,
            'total_searches': total_searches,
            'total_sources_used': total_sources,
            'search_usage_rate': turns_with_search / len(turns_results) if turns_results else 0
        }


def test_gemini_flash_browse():
    """Test the Gemini 2.5 Flash Browse adapter"""

    # Get API key from environment (you can also hardcode it here for testing)
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("Error: GEMINI_API_KEY environment variable not set")
        print("You can either:")
        print("1. Set environment variable: export GEMINI_API_KEY='your-key'")
        print("2. Or hardcode it in this function for testing")
        return

    # Create test episodes
    test_episodes = [
        {
            "id": "test_mrcr_1",
            "track": "mrcr",
            "turns": [
                {
                    "role": "user",
                    "text_content": "Who won the 2024 Euro football championship?"
                },
                {
                    "role": "user",
                    "text_content": "What was the final score?"
                },
                {
                    "role": "user",
                    "text_content": "Who were the top scorers in that tournament?"
                }
            ],
            "context_documents": []
        },
        {
            "id": "test_browsecomp_1",
            "track": "browsecomp",
            "turns": [
                {
                    "role": "user",
                    "text_content": "What are the latest developments in quantum computing in 2024?"
                }
            ],
            "context_documents": [
                {
                    "content": "Quantum computing has seen significant advances in error correction techniques."
                }
            ]
        }
    ]

    # Test single episode
    adapter = Gemini25FlashBrowseAdapter(api_key)

    try:
        # Test single episode processing
        print("\nTesting single episode processing...")
        result = adapter.process_episode(test_episodes[0], "test_output")
        print(f"Single episode result: {json.dumps(result, indent=2)[:500]}...")

        # Test batch processing
        print("\nTesting batch processing...")
        import asyncio
        batch_result = asyncio.run(
            adapter.process_episodes_batch(test_episodes, "test_output", max_concurrent=2)
        )
        print(f"Batch processing completed: {batch_result['successful']}/{batch_result['processed']} successful")

    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_gemini_flash_browse()
