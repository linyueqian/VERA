#!/usr/bin/env python3
"""
Phi-4 (Azure AI Inference) voice evaluator

Calls Azure AI Inference Chat Completions for the Phi-4 multimodal model
and evaluates VERA voice episodes. MRCR context documents are injected as
prior chat history (user/assistant turns) before the final user message.

Environment variables (defaults shown):
- PHI4_CHAT_COMPLETIONS_URL: full endpoint URL including api-version
- PHI4_API_KEY: Azure AI Inference API key
- PHI4_MODEL: model name (e.g. "Phi-4-multimodal-instruct")
"""

import os
import json
import time
import base64
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

import httpx
import asyncio
import sys


@dataclass
class Phi4Config:
    """Configuration for Phi-4 Azure voice evaluation"""
    api_url: str
    api_key: str
    model: str = "Phi-4-multimodal-instruct"
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 1024
    timeout: float = 90.0
    batch_size: int = 4
    max_concurrent: int = 8


class Phi4AzureVoiceEvaluator:
    """Evaluator that calls Azure AI Inference Chat Completions for Phi-4"""

    def __init__(self, config: Phi4Config):
        if not config.api_url:
            raise ValueError("PHI4_CHAT_COMPLETIONS_URL not set")
        if not config.api_key:
            raise ValueError("PHI4_API_KEY not set")
        self.config = config
        # Add project root so we can import streaming utils
        project_root = Path(__file__).parent.parent.parent
        sys.path.insert(0, str(project_root))
        try:
            from models.text.streaming_utils import measure_ttft_only  # type: ignore
            self._measure_ttft_only = measure_ttft_only
        except Exception:
            self._measure_ttft_only = None

    # ---- MRCR context parsing ----
    def parse_mrcr_context(self, context: str) -> List[Dict[str, str]]:
        """Parse MRCR context document into conversation messages (User:/Assistant:)"""
        messages: List[Dict[str, str]] = []
        role: Optional[str] = None
        buf: List[str] = []
        for line in context.splitlines():
            if line.startswith('User:'):
                if role and buf:
                    messages.append({'role': role, 'content': "\n".join(buf).strip()})
                role = 'user'
                buf = [line[len('User:'):].strip()]
            elif line.startswith('Assistant:'):
                if role and buf:
                    messages.append({'role': role, 'content': "\n".join(buf).strip()})
                role = 'assistant'
                buf = [line[len('Assistant:'):].strip()]
            else:
                buf.append(line)
        if role and buf:
            messages.append({'role': role, 'content': "\n".join(buf).strip()})
        return messages

    def _detect_task_type(self, episode: Dict[str, Any]) -> str:
        if episode.get("context_documents"):
            return "mrcr"
        return "standard"

    # ---- Message building ----
    def _encode_audio_wav(self, audio_path: str) -> Optional[Dict[str, Any]]:
        try:
            if not audio_path or not Path(audio_path).exists():
                return None
            b = Path(audio_path).read_bytes()
            b64 = base64.b64encode(b).decode('utf-8')
            return {"format": "wav", "data": b64}
        except Exception:
            return None

    def _to_typed_text(self, text: str) -> List[Dict[str, Any]]:
        # For chat/completions, use "text" type for content parts
        return [{"type": "text", "text": text or ""}]

    def _build_messages(self, episode: Dict[str, Any]) -> tuple[List[Dict[str, Any]], Optional[str]]:
        """Build messages array for Azure AI Inference chat/completions.

        - Inject MRCR context as prior chat history using typed input_text items
        - Final user message contains audio (if available) and text_content
        Returns (messages, audio_file_path)
        """
        messages: List[Dict[str, Any]] = []
        turns = episode.get("turns", []) or []
        turn0 = turns[0] if turns else {}
        audio_file = turn0.get("audio_file") or episode.get("audio_file") or ""
        # Fallbacks if audio path is missing or not found
        if not audio_file or not Path(audio_file).exists():
            episode_id = episode.get("id") or episode.get("episode_id") or ""
            candidate_ids = [episode_id]
            if episode_id.startswith("vera_"):
                candidate_ids.append(episode_id.replace("vera_", ""))
            possible_paths = []
            # Local test fixtures
            for cid in candidate_ids:
                if cid:
                    possible_paths.append(f"test_voice_episodes/audio/{cid}.wav")
            # Common dataset folders
            for cid in candidate_ids:
                if cid:
                    possible_paths.extend([
                        f"data/final_dataset/voice/aime_voice_episodes_audio/{cid}.wav",
                        f"data/final_dataset/voice/gpqa_diamond_voice_episodes_audio/{cid}.wav",
                        f"data/final_dataset/voice/mrcr_voice_episodes_audio/{cid}.wav",
                        f"data/final_dataset/voice/browsecomp_voice_episodes_audio/{cid}.wav",
                        f"data/final_dataset/voice/simpleqa_voice_episodes_audio/{cid}.wav",
                    ])
            for p in possible_paths:
                if Path(p).exists():
                    audio_file = p
                    break
        text_content = turn0.get("text_content", "")

        # Optional system prompt (as plain string, not typed content)
        messages.append({
            "role": "system",
            "content": "You are a helpful AI assistant. Analyze any provided audio and answer clearly."
        })

        # MRCR: inject context docs as chat history
        if self._detect_task_type(episode) == "mrcr":
            for doc in episode.get("context_documents", []) or []:
                content = doc.get("content", "")
                if not content:
                    continue
                for m in self.parse_mrcr_context(content):
                    messages.append({
                        "role": m.get("role", "user"),
                        # For MRCR history, send plain string content for compatibility
                        "content": m.get("content", "")
                    })

        # Final user message: include audio (if available) + text
        content_items: List[Dict[str, Any]] = []
        audio_obj = self._encode_audio_wav(audio_file)
        if audio_obj is not None:
            # For chat/completions, provide audio as a data URL via audio_url
            content_items.append({
                "type": "audio_url",
                "audio_url": {"url": f"data:audio/wav;base64,{audio_obj['data']}"}
            })
        if text_content:
            content_items.append({"type": "text", "text": text_content})
        if not content_items:
            # If no audio or text, fall back to plain string content
            messages.append({"role": "user", "content": "Please analyze this audio and respond."})
            return messages, audio_file or None

        messages.append({"role": "user", "content": content_items})
        return messages, audio_file or None

    # ---- API call ----
    def _call_phi4_chat(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        headers = {
            "Content-Type": "application/json",
            "api-key": self.config.api_key,
        }
        payload: Dict[str, Any] = {
            "messages": messages,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "max_tokens": self.config.max_tokens,
            "model": self.config.model,
        }
        with httpx.Client(timeout=self.config.timeout) as client:
            resp = client.post(self.config.api_url, headers=headers, json=payload)
            try:
                resp.raise_for_status()
                return resp.json()
            except httpx.HTTPStatusError:
                return {"error": f"HTTP {resp.status_code}: {resp.text}", "status_code": resp.status_code}
            except Exception as e:
                return {"error": f"Unexpected error: {e}"}

    def _measure_ttft(self, messages: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Measure TTFT via streaming without collecting full output."""
        if self._measure_ttft_only is None:
            return None
        headers = {
            "Content-Type": "application/json",
            "api-key": self.config.api_key,
        }
        payload: Dict[str, Any] = {
            "messages": messages,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "max_tokens": 128,
            "model": self.config.model,
        }
        try:
            # Endpoint behaves like OpenAI streaming
            return asyncio.run(self._measure_ttft_only(
                self.config.api_url,
                payload,
                headers,
                None,
                endpoint_type="openai",
                timeout=self.config.timeout,
            ))
        except Exception:
            return None

    def _stream_chat_collect(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Stream the chat completion, capture TTFT and collect text.
        Returns { 'ttft_ms': float|None, 'first_token_timestamp': float|None, 'text': str }
        """
        headers = {
            "Content-Type": "application/json",
            "api-key": self.config.api_key,
            "Accept": "text/event-stream",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
        payload: Dict[str, Any] = {
            "messages": messages,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "max_tokens": self.config.max_tokens,
            "model": self.config.model,
            "stream": True,
        }

        request_start = time.time()
        first_token_time: Optional[float] = None
        text_parts: List[str] = []

        def _extract_text_from_chunk(obj: Dict[str, Any]) -> str:
            # OpenAI chat style
            try:
                choices = obj.get("choices") or []
                if choices:
                    delta = choices[0].get("delta") or {}
                    c = delta.get("content")
                    if isinstance(c, str):
                        return c
                    if isinstance(c, list):
                        acc = []
                        for item in c:
                            t = item.get("type")
                            if t in ("output_text", "text", "output_text_delta"):
                                s = item.get("text", "")
                                if s:
                                    acc.append(s)
                        return "".join(acc)
            except Exception:
                pass
            # Responses/SSE style
            t = obj.get("type")
            if t == "response.output_text.delta":
                return obj.get("delta", "") or ""
            if t == "response.output_item.added":
                item = obj.get("item", {})
                if item.get("type") == "message":
                    acc = []
                    for content_item in item.get("content", []) or []:
                        if content_item.get("type") in ("output_text", "output_text_delta", "text"):
                            s = content_item.get("text", "")
                            if s:
                                acc.append(s)
                    return "".join(acc)
            # message.delta variant
            if t in ("message.delta", "message"):
                delta = obj.get("delta") or {}
                acc = []
                # delta.content may be list of typed items
                content = delta.get("content")
                if isinstance(content, str):
                    acc.append(content)
                elif isinstance(content, list):
                    for it in content:
                        if it.get("type") in ("output_text", "output_text_delta", "text"):
                            s = it.get("text", "")
                            if s:
                                acc.append(s)
                return "".join(acc)
            return ""

        try:
            with httpx.Client(timeout=self.config.timeout) as client:
                with client.stream("POST", self.config.api_url, headers=headers, json=payload) as resp:
                    resp.raise_for_status()
                    for raw_line in resp.iter_lines():
                        if not raw_line:
                            continue
                        line = raw_line.decode("utf-8") if isinstance(raw_line, (bytes, bytearray)) else raw_line
                        line = line.strip()
                        if not line:
                            continue
                        if line.startswith("event:"):
                            # Ignore event header; we only use data lines
                            continue
                        if not line.startswith("data:"):
                            continue
                        data_str = line[5:].strip()
                        if data_str == "[DONE]":
                            break
                        try:
                            obj = json.loads(data_str)
                        except Exception:
                            continue
                        piece = _extract_text_from_chunk(obj)
                        if piece:
                            if first_token_time is None:
                                first_token_time = time.time()
                            text_parts.append(piece)
            ttft_ms = (first_token_time - request_start) * 1000 if first_token_time else None
            return {"ttft_ms": ttft_ms, "first_token_timestamp": first_token_time, "text": "".join(text_parts)}
        except Exception:
            return {"ttft_ms": None, "first_token_timestamp": None, "text": ""}

    # ---- Async streaming/non-streaming helpers ----
    async def _astream_chat_collect(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        headers = {
            "Content-Type": "application/json",
            "api-key": self.config.api_key,
            "Accept": "text/event-stream",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
        payload: Dict[str, Any] = {
            "messages": messages,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "max_tokens": self.config.max_tokens,
            "model": self.config.model,
            "stream": True,
        }
        request_start = time.time()
        first_token_time: Optional[float] = None
        text_parts: List[str] = []

        def _extract_text(obj: Dict[str, Any]) -> str:
            try:
                choices = obj.get("choices") or []
                if choices:
                    delta = choices[0].get("delta") or {}
                    c = delta.get("content")
                    if isinstance(c, str):
                        return c
                    if isinstance(c, list):
                        acc = []
                        for it in c:
                            if it.get("type") in ("output_text", "text", "output_text_delta"):
                                s = it.get("text", "")
                                if s:
                                    acc.append(s)
                        return "".join(acc)
            except Exception:
                pass
            t = obj.get("type")
            if t == "response.output_text.delta":
                return obj.get("delta", "") or ""
            if t == "response.output_item.added":
                item = obj.get("item", {})
                if item.get("type") == "message":
                    acc = []
                    for ci in item.get("content", []) or []:
                        if ci.get("type") in ("output_text", "output_text_delta", "text"):
                            s = ci.get("text", "")
                            if s:
                                acc.append(s)
                    return "".join(acc)
            if t in ("message.delta", "message"):
                delta = obj.get("delta") or {}
                content = delta.get("content")
                if isinstance(content, str):
                    return content
                if isinstance(content, list):
                    acc = []
                    for it in content:
                        if it.get("type") in ("output_text", "output_text_delta", "text"):
                            s = it.get("text", "")
                            if s:
                                acc.append(s)
                    return "".join(acc)
            return ""

        try:
            async with httpx.AsyncClient(timeout=self.config.timeout) as client:
                async with client.stream("POST", self.config.api_url, headers=headers, json=payload) as resp:
                    resp.raise_for_status()
                    async for line in resp.aiter_lines():
                        if not line:
                            continue
                        line = line.strip()
                        if not line:
                            continue
                        if line.startswith("event:"):
                            continue
                        if not line.startswith("data:"):
                            continue
                        data_str = line[5:].strip()
                        if data_str == "[DONE]":
                            break
                        try:
                            obj = json.loads(data_str)
                        except Exception:
                            continue
                        piece = _extract_text(obj)
                        if piece:
                            if first_token_time is None:
                                first_token_time = time.time()
                            text_parts.append(piece)
            ttft_ms = (first_token_time - request_start) * 1000 if first_token_time else None
            return {"ttft_ms": ttft_ms, "first_token_timestamp": first_token_time, "text": "".join(text_parts)}
        except Exception:
            return {"ttft_ms": None, "first_token_timestamp": None, "text": ""}

    async def _acall_phi4_chat(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        headers = {"Content-Type": "application/json", "api-key": self.config.api_key}
        payload: Dict[str, Any] = {
            "messages": messages,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "max_tokens": self.config.max_tokens,
            "model": self.config.model,
        }
        try:
            async with httpx.AsyncClient(timeout=self.config.timeout) as client:
                resp = await client.post(self.config.api_url, headers=headers, json=payload)
                resp.raise_for_status()
                return resp.json()
        except httpx.HTTPStatusError as e:
            return {"error": f"HTTP {e.response.status_code}: {e.response.text}", "status_code": e.response.status_code}
        except Exception as e:
            return {"error": f"Unexpected error: {e}"}

    async def process_episode_async(self, episode: Dict[str, Any]) -> Dict[str, Any]:
        episode_id = episode.get("id") or episode.get("episode_id") or "unknown"
        task_type = self._detect_task_type(episode)
        try:
            messages, audio_path = self._build_messages(episode)
            stream_info = await self._astream_chat_collect(messages)
            ttft_ms = stream_info.get("ttft_ms")
            first_token_ts = stream_info.get("first_token_timestamp")
            streamed_text = stream_info.get("text", "")
            request_start = time.time() if first_token_ts is None else (first_token_ts - (ttft_ms/1000.0) if ttft_ms else time.time())
            if streamed_text:
                assistant_text = streamed_text
                usage = {}
            else:
                resp = await self._acall_phi4_chat(messages)
                assistant_text = ""
                try:
                    choice = (resp.get("choices") or [{}])[0]
                    msg = choice.get("message") or {}
                    content = msg.get("content")
                    if isinstance(content, str):
                        assistant_text = content
                    elif isinstance(content, list):
                        parts = []
                        for it in content:
                            if it.get("type") in ("output_text", "text"):
                                parts.append(it.get("text", ""))
                        assistant_text = "\n".join([p for p in parts if p])
                    else:
                        assistant_text = msg.get("content") or ""
                except Exception:
                    pass
                usage = resp.get("usage", {})
            request_end = time.time()
            return {
                "episode_id": episode_id,
                "task_type": task_type,
                "response": assistant_text or "",
                "expected_answer": episode.get("expected_answer", ""),
                "contains_answer": False,
                "audio_file": audio_path or "",
                "original_text": episode.get("question", episode.get("transcript", "")),
                "has_audio": bool(audio_path),
                "error": None,
                "timing": {
                    "request_start_time": request_start,
                    "request_end_time": request_end,
                    "first_token_time": first_token_ts,
                    "time_to_first_token_ms": ttft_ms,
                    "total_time": request_end - request_start,
                },
                "token_usage": {
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0),
                },
            }
        except Exception as e:
            return {
                "episode_id": episode_id,
                "task_type": task_type,
                "response": "",
                "expected_answer": episode.get("expected_answer", ""),
                "contains_answer": False,
                "audio_file": episode.get("audio_file", ""),
                "original_text": episode.get("question", episode.get("transcript", "")),
                "has_audio": bool(episode.get("audio_file")),
                "error": str(e),
                "token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            }

    async def _process_batch_async(self, episodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        sem = asyncio.Semaphore(self.config.max_concurrent)

        async def runner(ep: Dict[str, Any]) -> Dict[str, Any]:
            async with sem:
                return await self._process_episode_async(ep)

        tasks = [asyncio.create_task(runner(ep)) for ep in episodes]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        out: List[Dict[str, Any]] = []
        for i, r in enumerate(results):
            if isinstance(r, Exception):
                eid = episodes[i].get("id") or episodes[i].get("episode_id") or f"episode_{i}"
                out.append({
                    "episode_id": eid,
                    "task_type": self._detect_task_type(episodes[i]),
                    "response": "",
                    "expected_answer": episodes[i].get("expected_answer", ""),
                    "contains_answer": False,
                    "audio_file": episodes[i].get("audio_file", ""),
                    "original_text": episodes[i].get("question", episodes[i].get("transcript", "")),
                    "has_audio": bool(episodes[i].get("audio_file")),
                    "error": str(r),
                    "token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                })
            else:
                out.append(r)
        return out
    # ---- Batch processing ----
    def process_episode(self, episode: Dict[str, Any], output_dir: str = None) -> Dict[str, Any]:
        """Synchronous wrapper for process_episode_async"""
        return asyncio.run(self.process_episode_async(episode))

    def process_batch(self, episodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for episode in episodes:
            episode_id = episode.get("id") or episode.get("episode_id") or "unknown"
            task_type = self._detect_task_type(episode)
            try:
                messages, audio_path = self._build_messages(episode)
                # Stream to get TTFT and possibly the full text
                stream_info = self._stream_chat_collect(messages)
                ttft_ms = stream_info.get("ttft_ms")
                first_token_ts = stream_info.get("first_token_timestamp")
                streamed_text = stream_info.get("text", "")
                request_start = time.time() if first_token_ts is None else (first_token_ts - (ttft_ms/1000.0) if ttft_ms else time.time())
                # If we got streamed text, use it; else fall back to non-streaming call
                if streamed_text:
                    assistant_text_stream = streamed_text
                    request_end = time.time()
                    response = {"choices": [{"message": {"content": assistant_text_stream}}], "usage": {}}
                else:
                    response = self._call_phi4_chat(messages)
                    request_end = time.time()

                # Extract assistant content (string or typed array)
                assistant_text = ""
                try:
                    choice = (response.get("choices") or [{}])[0]
                    message = choice.get("message") or {}
                    content = message.get("content")
                    if isinstance(content, str):
                        assistant_text = content
                    elif isinstance(content, list):
                        parts: List[str] = []
                        for it in content:
                            if it.get("type") in ("output_text", "text"):
                                parts.append(it.get("text", ""))
                            elif it.get("type") == "tool_result":
                                parts.append(it.get("content", ""))
                        assistant_text = "\n".join([p for p in parts if p])
                    else:
                        # Some endpoints mirror OpenAI: .message.content as string
                        assistant_text = message.get("content") or ""
                except Exception:
                    pass

                usage = response.get("usage", {})
                result = {
                    "episode_id": episode_id,
                    "task_type": task_type,
                    "response": assistant_text or "",
                    "expected_answer": episode.get("expected_answer", ""),
                    "contains_answer": False,
                    "audio_file": audio_path or "",
                    "original_text": episode.get("question", episode.get("transcript", "")),
                    "has_audio": bool(audio_path),
                    "error": response.get("error"),
                    "timing": {
                        "request_start_time": request_start,
                        "request_end_time": request_end,
                        "first_token_time": first_token_ts,
                        "time_to_first_token_ms": ttft_ms,
                        "total_time": request_end - request_start,
                    },
                    "token_usage": {
                        "prompt_tokens": usage.get("prompt_tokens", 0),
                        "completion_tokens": usage.get("completion_tokens", 0),
                        "total_tokens": usage.get("total_tokens", 0),
                    },
                }
                results.append(result)
            except Exception as e:
                results.append({
                    "episode_id": episode_id,
                    "task_type": task_type,
                    "response": "",
                    "expected_answer": episode.get("expected_answer", ""),
                    "contains_answer": False,
                    "audio_file": episode.get("audio_file", ""),
                    "original_text": episode.get("question", episode.get("transcript", "")),
                    "has_audio": bool(episode.get("audio_file")),
                    "error": str(e),
                    "token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                })
        return results

    # ---- Output shaping (text_output style) ----
    def _to_text_output(self, result: Dict[str, Any], start_time: float) -> Dict[str, Any]:
        response = result.get("response", "")
        episode_id = result.get("episode_id", "unknown")
        expected_answer = result.get("expected_answer", "")
        token_usage = result.get("token_usage", {})
        audio_file = result.get("audio_file", "")
        task_type = result.get("task_type", "standard")

        timing = result.get("timing", {})
        session_duration = timing.get("total_time", time.time() - start_time)
        request_start = timing.get("request_start_time", start_time)
        request_end = timing.get("request_end_time", start_time + session_duration)

        ttft_ms = timing.get("time_to_first_token_ms")
        turn_result = {
            "turn_index": 0,
            "user_input": audio_file or "Audio input processed",
            "model_response": response,
            "total_turn_latency_ms": session_duration * 1000,
            "api_latency_ms": session_duration * 1000,
            "processing_overhead_ms": 0.0,
            "token_usage": {
                "completion_tokens": token_usage.get("completion_tokens", 0),
                "completion_tokens_details": {
                    "accepted_prediction_tokens": 0,
                    "audio_tokens": token_usage.get("prompt_tokens", 0) if result.get("has_audio") else 0,
                    "reasoning_tokens": 0,
                    "rejected_prediction_tokens": 0,
                },
                "prompt_tokens": token_usage.get("prompt_tokens", 0),
                "prompt_tokens_details": {
                    "audio_tokens": token_usage.get("prompt_tokens", 0) if result.get("has_audio") else 0,
                    "cached_tokens": 0,
                },
                "total_tokens": token_usage.get("total_tokens", 0),
            },
            "timing_metadata": {
                "api_latency_ms": session_duration * 1000,
                "time_to_first_token_ms": ttft_ms if ttft_ms is not None else None,
                "request_start_timestamp": request_start,
                "request_end_timestamp": request_end,
                "failed": bool(result.get("error")),
                "timed_out": False,
                "error": bool(result.get("error")),
            },
            "model_metadata": {
                "model": "phi-4-multimodal",
                "deployment": self.config.model,
                "response_id": f"phi4_{episode_id}_{int(time.time())}",
                "finish_reason": "stop",
            },
            "error": result.get("error"),
        }

        episode_result = {
            "episode_id": episode_id,
            "model_name": f"phi-4-{task_type}-azure",
            "task_type": task_type,
            "session_duration": session_duration,
            "total_turns_processed": 1,
            "total_tokens_used": token_usage.get("total_tokens", 0),
            "turn_results": [turn_result],
            "timing_data": {
                "total_session_time_ms": session_duration * 1000,
                "average_total_turn_latency_ms": session_duration * 1000,
                "average_api_latency_ms": session_duration * 1000,
                "average_processing_overhead_ms": 0.0,
                "total_api_time_ms": session_duration * 1000,
                "total_processing_overhead_ms": 0.0,
            },
            "reasoning_data": {
                "responses": [response],
                "average_response_length": len(response),
            },
            "success": not bool(result.get("error")),
            "expected_answer": expected_answer,
            "contains_answer": result.get("contains_answer", False),
            "has_audio": result.get("has_audio", False),
        }
        return episode_result

    def evaluate_dataset(self, dataset_file: str, output_dir: str, max_episodes: Optional[int] = None) -> List[Dict[str, Any]]:
        data = json.loads(Path(dataset_file).read_text())
        episodes = data.get("episodes", [])
        if max_episodes:
            episodes = episodes[:max_episodes]

        # Process in mini-batches for memory friendliness
        batch_size = self.config.batch_size
        all_episode_results: List[Dict[str, Any]] = []

        # Output directory
        ts = time.strftime("%Y%m%d_%H%M%S")
        src = Path(dataset_file).stem.replace("_voice_episodes", "")
        out_dir = Path(output_dir) / f"phi4_azure_{src}_{ts}"
        out_dir.mkdir(parents=True, exist_ok=True)

        for i in range(0, len(episodes), batch_size):
            batch = episodes[i:i+batch_size]
            print(f"[Phi-4 Azure] Processing batch {i//batch_size+1}/{(len(episodes)-1)//batch_size+1} ({len(batch)})")
            batch_start = time.time()
            # Prefer async concurrent processing; fallback to sync if event loop issues
            try:
                results = asyncio.run(self._process_batch_async(batch))
            except RuntimeError:
                results = self.process_batch(batch)
            for r in results:
                all_episode_results.append(self._to_text_output(r, batch_start))

        # Save outputs
        batch_id = int(time.time() * 1000)
        summary = {
            "model": "phi-4-azure",
            "dataset": src,
            "dataset_path": str(dataset_file),
            "output_directory": str(out_dir),
            "total_episodes": len(all_episode_results),
            "processed": len(all_episode_results),
            "successful": sum(1 for r in all_episode_results if r.get("success")),
            "failed": sum(1 for r in all_episode_results if not r.get("success")),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }

        (out_dir / "evaluation_summary.json").write_text(json.dumps(summary, indent=2))
        (out_dir / f"phi4_azure_batch_{batch_id}.json").write_text(json.dumps({
            "model_name": "phi-4-azure",
            "results": all_episode_results
        }, indent=2))

        print(f"[Phi-4 Azure] ✓ Completed. Results saved to: {out_dir}")
        return all_episode_results


def main():
    parser = argparse.ArgumentParser(description="Phi-4 Azure voice evaluator for VERA")
    parser.add_argument("--dataset-dir", type=Path, required=True, help="Directory containing voice datasets")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory for results")
    parser.add_argument("--specific-dataset", type=str, default=None, help="Specific dataset file to evaluate")
    parser.add_argument("--max-episodes", type=int, default=None, help="Max episodes to process")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--max-tokens", type=int, default=1024, help="Max tokens to generate")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for processing")
    parser.add_argument("--max-concurrent", type=int, default=8, help="Max concurrent requests (async)")

    args = parser.parse_args()

    api_url = os.getenv(
        "PHI4_CHAT_COMPLETIONS_URL",
        "https://your-phi4-endpoint.services.ai.azure.com/models/chat/completions?api-version=2024-05-01-preview",
    )
    api_key = os.getenv("PHI4_API_KEY", "")
    model = os.getenv("PHI4_MODEL", "Phi-4-multimodal-instruct")

    config = Phi4Config(
        api_url=api_url,
        api_key=api_key,
        model=model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        batch_size=args.batch_size,
        max_concurrent=args.max_concurrent,
    )
    evaluator = Phi4AzureVoiceEvaluator(config)

    # Resolve dataset file
    if args.specific_dataset:
        dataset_file = args.dataset_dir / args.specific_dataset if not Path(args.specific_dataset).is_absolute() else Path(args.specific_dataset)
    else:
        voice_files = sorted(Path(args.dataset_dir).glob("*_voice_episodes.json"))
        if not voice_files:
            print("Error: No voice dataset files found")
            return
        dataset_file = voice_files[0]
        print(f"Using dataset: {dataset_file}")

    evaluator.evaluate_dataset(str(dataset_file), str(args.output_dir), max_episodes=args.max_episodes)


if __name__ == "__main__":
    main()
