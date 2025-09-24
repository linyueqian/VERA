import os
import json
import time
from typing import List, Optional
import requests

from .utils import _env
from .explain import ExplainSynthesizer
from .mrcr_context import load_context_documents_from_audio_file, is_mrcr_episode


class StandardProblemSolver:
    """
    Problem solver using Azure OpenAI Responses API.
    Based on the working solver.py pattern with Azure endpoints.
    """

    def __init__(self, explainer: ExplainSynthesizer, audio_file_path: Optional[str] = None):
        self.explainer = explainer
        self.audio_file_path = audio_file_path

        # Load MRCR context only if this is an MRCR file
        self.context_documents = []
        if audio_file_path and "mrcr" in str(audio_file_path).lower():
            self.context_documents = load_context_documents_from_audio_file(audio_file_path)

    def _detect_benchmark_type(self) -> str:
        """Detect benchmark type from audio file path."""
        if not self.audio_file_path:
            return "unknown"

        from pathlib import Path
        audio_name = Path(self.audio_file_path).name.lower()

        if "browsecomp" in audio_name:
            return "browsecomp"
        elif "mrcr" in audio_name:
            return "mrcr"
        elif any(name in audio_name for name in ["aime", "gpqa", "simpleqa"]):
            return "reasoning"
        else:
            return "unknown"

    def _openai_responses_stream(self, request_text: str) -> None:
        """Stream responses from OpenAI responses endpoint with web search."""

        api_key = _env("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY missing")

        openai_base = _env("OPENAI_ENDPOINT", "https://api.openai.com/v1")
        url = f"{openai_base.rstrip('/')}/responses"
        model = _env("OPENAI_MODEL", "gpt-5")

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }

        tools = [
            {"type": "web_search_preview", "search_context_size": _env("WEB_SEARCH_CONTEXT_SIZE", "medium")},
            {"type": "code_interpreter", "container": {"type": "auto"}},
        ]

        summary_mode = (_env("OPENAI_REASONING_SUMMARY", "detailed") or "").strip().lower()
        reasoning_params = {"effort": "low"}
        if summary_mode and summary_mode not in ("disabled",):
            reasoning_params["summary"] = summary_mode

        # Build the request text with MRCR context if available (following gpt5_thinking_azure.py approach)
        if self.context_documents:
            episode_id = None
            if self.audio_file_path:
                from pathlib import Path
                episode_id = Path(self.audio_file_path).stem

            # Check if this is MRCR (same logic as gpt5_thinking_azure.py)
            is_mrcr = 'mrcr' in (episode_id or '').lower()

            if is_mrcr and self.context_documents:
                # For MRCR, the context IS the conversation history - add it directly as text
                context_parts = []
                for doc in self.context_documents:
                    doc_content = doc.get('content', '')
                    if doc_content:
                        # Add it directly as it's already formatted as User:/Assistant: dialogue
                        context_parts.append(doc_content)
                        print(f"!!!MRCR: Added context document with {len(doc_content)} characters")

                if context_parts:
                    context_parts.append("")  # Add blank line after context
                    context_parts.append(f"User: {request_text}")
                    request_text = "\n".join(context_parts)
                    print(f"!!!MRCR: Built combined prompt with {len(request_text)} total characters")

        # Payload following OpenAI responses API format with tools
        payload = {
            "model": model,
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": request_text}
                    ]
                }
            ],
            "reasoning": reasoning_params,
            "tools": tools,
            "tool_choice": "auto",
            "parallel_tool_calls": True,
            "stream": True,
            "truncation": "auto",
            "store": True,
            "service_tier": "auto",
            "text": {
                "format": {"type": "text"},
                "verbosity": "medium",
            },
        }

        params = None  # OpenAI doesn't use api-version parameter

        # Call the shared streaming logic
        self._shared_responses_stream(url, headers, params, payload)

    def _azure_responses_stream(self, request_text: str) -> None:
        """Stream responses from Azure OpenAI responses endpoint."""

        # Get Azure API configuration
        api_key = _env("AZURE_OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("AZURE_OPENAI_API_KEY missing")

        azure_endpoint = _env("AZURE_OPENAI_ENDPOINT")
        if not azure_endpoint:
            raise RuntimeError("AZURE_OPENAI_ENDPOINT missing")

        url = f"{azure_endpoint.rstrip('/')}/openai/responses"
        api_version = _env("AZURE_OPENAI_API_VERSION", "2025-04-01-preview")
        model = "gpt-5"  # Use gpt-5 for responses endpoint

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "api-key": api_key  # Azure also needs this header
        }

        # Build the request text with MRCR context if available (following gpt5_thinking_azure.py approach)
        if self.context_documents:
            episode_id = None
            if self.audio_file_path:
                from pathlib import Path
                episode_id = Path(self.audio_file_path).stem

            # Check if this is MRCR (same logic as gpt5_thinking_azure.py)
            is_mrcr = 'mrcr' in (episode_id or '').lower()

            if is_mrcr and self.context_documents:
                # For MRCR, the context IS the conversation history - add it directly as text
                context_parts = []
                for doc in self.context_documents:
                    doc_content = doc.get('content', '')
                    if doc_content:
                        # Add it directly as it's already formatted as User:/Assistant: dialogue
                        context_parts.append(doc_content)
                        print(f"!!!MRCR: Added context document with {len(doc_content)} characters")

                if context_parts:
                    context_parts.append("")  # Add blank line after context
                    context_parts.append(f"User: {request_text}")
                    request_text = "\n".join(context_parts)
                    print(f"!!!MRCR: Built combined prompt with {len(request_text)} total characters")

        # Reasoning configuration (low effort, detailed summary)
        reasoning_params = {"effort": "low", "summary": "detailed"}

        # Payload following Azure responses API format
        payload = {
            "model": model,
            "input": [
                {
                    "role": "user",
                    "content": request_text
                }
            ],
            "reasoning": reasoning_params,
            "stream": True,
            "max_output_tokens": 16384,
            "text": {
                "verbosity": "high"
            }
        }

        # Add API version parameter
        params = {"api-version": api_version}

        # Call the shared streaming logic
        self._shared_responses_stream(url, headers, params, payload)

    def _shared_responses_stream(self, url: str, headers: dict, params: Optional[dict], payload: dict) -> None:
        """Shared streaming logic for both Azure and OpenAI endpoints."""

        with requests.post(url, headers=headers, params=params, data=json.dumps(payload), stream=True, timeout=300) as resp:
            resp.raise_for_status()

            rs_buf: List[str] = []
            out_buf: List[str] = []  # For complete GPT-5 response (including reasoning)
            final_answer_buf: List[str] = []  # For actual final answer only (without reasoning)
            final_answer_sent = False

            def extract_text_segments(payload: object) -> List[str]:
                segments: List[str] = []
                if isinstance(payload, str):
                    if payload:
                        segments.append(payload)
                elif isinstance(payload, dict):
                    text_val = payload.get("text")
                    if isinstance(text_val, str) and text_val:
                        segments.append(text_val)
                    nested = payload.get("content")
                    if nested is not None:
                        segments.extend(extract_text_segments(nested))
                elif isinstance(payload, list):
                    for item in payload:
                        segments.extend(extract_text_segments(item))
                return segments

            def iter_sse_json(response):
                buf = bytearray()
                for chunk in response.iter_content(chunk_size=4096):
                    if not chunk:
                        continue
                    if isinstance(chunk, str):
                        chunk = chunk.encode("utf-8")
                    buf.extend(chunk)
                    start_idx = 0
                    while True:
                        newline_idx = buf.find(b"\n", start_idx)
                        if newline_idx == -1:
                            break
                        line = buf[start_idx:newline_idx].strip()
                        start_idx = newline_idx + 1
                        if not line or line.startswith(b":"):
                            continue
                        if not line.startswith(b"data:"):
                            continue
                        data_part = line[5:].strip()
                        if data_part == b"[DONE]":
                            return
                        try:
                            yield json.loads(data_part.decode("utf-8"))
                        except Exception:
                            continue
                    if start_idx > 0:
                        del buf[:start_idx]

            for obj in iter_sse_json(resp):
                et = obj.get("type") or ""

                if et == "response.reasoning_summary_text.delta":
                    text = (obj.get("delta") or obj.get("text") or "")
                    if text:
                        rs_buf.append(text)
                elif et == "response.reasoning_summary_text.done":
                    text = (obj.get("text") or "").strip()
                    summary = ("".join(rs_buf) or text).strip()
                    rs_buf = []
                    if summary:
                        print(f"!!!Reasoning summary: {summary}")
                        # Add reasoning summary to out_buf for complete GPT-5 response
                        out_buf.append(f"Reasoning:\n{summary}\n\n")
                        self.explainer.push_thought(summary)
                elif et == "response.output_text.delta":
                    delta_payload = obj.get("delta")
                    texts = extract_text_segments(delta_payload)
                    if not texts:
                        texts = extract_text_segments(obj.get("text"))
                    out_buf.extend(texts)
                    final_answer_buf.extend(texts)  # Also collect for final answer
                    # Send delta to explainer for real-time streaming, no final answer detection here
                elif et == "response.output_text.done":
                    final_piece = obj.get("text")
                    piece_texts = extract_text_segments(final_piece)
                    out_buf.extend(piece_texts)
                    final_answer_buf.extend(piece_texts)  # Also collect for final answer
                    role = obj.get("role") or None
                    # Don't treat intermediate output_text.done as final answer
                    # Wait for response.completed to determine the actual final answer
                elif et == "response.content_part.done":
                    part = obj.get("part") or {}
                    if isinstance(part, dict) and part.get("type") == "output_text":
                        part_texts = extract_text_segments(part.get("text"))
                        out_buf.extend(part_texts)
                        final_answer_buf.extend(part_texts)
                        # Don't treat content_part.done as final answer either
                        # Wait for response.completed
                elif et == "response.content_part.added":
                    part = obj.get("part") or {}
                    if isinstance(part, dict) and part.get("type") == "output_text":
                        part_texts = extract_text_segments(part.get("text"))
                        out_buf.extend(part_texts)
                        final_answer_buf.extend(part_texts)
                        # Don't treat content_part.added as final answer either
                        # Wait for response.completed
                elif et == "response.output_item.done":
                    item = obj.get("item") or {}
                    if not isinstance(item, dict):
                        continue
                    item_type = item.get("type")
                    if item_type == "message" and item.get("role") == "assistant":
                        contents = item.get("content")
                        texts = extract_text_segments(contents)
                        if texts:
                            out_buf.extend(texts)
                            final_answer_buf.extend(texts)
                            # Don't treat output_item.done as final answer either
                            # Wait for response.completed to get the complete answer
                    elif item_type == "web_search_call":
                        action = item.get("action") or {}
                        query = action.get("query") or action.get("url") or ""
                        status = item.get("status") or "completed"
                        if query:
                            thought = f"Web search {status}: {query}"
                        else:
                            thought = f"Web search {status}"
                        print(f"!!!Web search: {thought}")
                        # Add web search to out_buf for complete GPT-5 response
                        out_buf.append(f"{thought}\n")
                        self.explainer.push_thought(thought)
                    elif item_type == "code_interpreter_call":
                        outputs = item.get("outputs") or []
                        output_texts: List[str] = []
                        for output in outputs:
                            if isinstance(output, dict):
                                if output.get("type") == "text":
                                    output_texts.extend(extract_text_segments(output.get("text")))
                                elif output.get("type") == "logs":
                                    log_text = output.get("logs")
                                    if isinstance(log_text, str) and log_text:
                                        output_texts.append(log_text)
                        combined = " ".join(output_texts).strip()
                        if combined:
                            snippet = combined[:400]
                            thought = f"Code interpreter output: {snippet}"
                        else:
                            thought = "Code interpreter call completed"
                        print(f"!!!Code interpreter: {thought}")
                        # Add code interpreter output to out_buf for complete GPT-5 response
                        out_buf.append(f"{thought}\n")
                        self.explainer.push_thought(thought)
                elif et == "response.completed":
                    # Use final_answer_buf which contains only the output text (without reasoning summaries)
                    final_text = "".join(final_answer_buf).strip()

                    # If final_answer_buf is empty, try to extract from response object
                    if not final_text:
                        resp_obj = obj.get("response", {}) or {}
                        items = resp_obj.get("output", []) or []
                        texts: List[str] = []
                        for item in items:
                            if not isinstance(item, dict):
                                continue
                            if item.get("type") == "message":
                                texts.extend(extract_text_segments(item.get("content")))
                        final_text = "".join(texts).strip()

                    # If we haven't sent a final answer yet, send it now
                    if final_text and not final_answer_sent:
                        print(f"!!!Final answer from GPT-5 (length: {len(final_text)}): {final_text[:200]}...")
                        self.explainer.push_thought(f"This is my final answer: {final_text}")
                        final_answer_sent = True
                    elif not final_text and not final_answer_sent:
                        # No output text found - GPT-5 might still be thinking
                        print(f"!!!No final answer in output text, GPT-5 may still be processing")
                        print(f"!!!Reasoning buffer had {len(''.join(rs_buf))} chars")
                        print(f"!!!Full output buffer had {len(''.join(out_buf))} chars")
                        self.explainer.push_thought("This is my final answer: [Still processing, no final answer yet]")
                        final_answer_sent = True

                    # Signal that solver has finished - explainer should now generate final explanation with long budget
                    print("!!!Solver streaming completed, sending completion signal")
                    self.explainer.push_thought(None)

            # Store both the raw GPT-5 response and debug info
            full_text = "".join(out_buf).strip()
            self.explainer.gpt5_response = full_text  # Raw GPT-5 response
            print(f"StandardProblemSolver: Generated {len(full_text)} characters")
            print(f"Generated text: {full_text[:200]}...")
            print(f"Explainer spoken_explanation: {len(self.explainer.spoken_explanation)} characters")
            print(f"Spoken explanation: {self.explainer.spoken_explanation[:200]}...")

    def start(self, request: str) -> None:
        """
        Start solving the problem and feed explanation to the explainer.
        Compatible interface with original solver.
        """
        print(f"StandardProblemSolver: Processing request: {request[:100]}...")

        # Detect benchmark type and choose appropriate endpoint
        benchmark_type = self._detect_benchmark_type()

        # Use OpenAI for browsecomp (needs web search), Azure for others
        if benchmark_type == "browsecomp":
            print(f"Using OpenAI endpoint for browsecomp (web search enabled)")
            self._openai_responses_stream(request)
        else:
            print(f"Using Azure endpoint for {benchmark_type} (reasoning optimized)")
            self._azure_responses_stream(request)


class ProblemSolverFactory:
    """Factory to create appropriate solver based on configuration."""

    @staticmethod
    def create(explainer: ExplainSynthesizer, audio_file_path: Optional[str] = None):
        """Create a problem solver based on available configuration."""

        # Check if we should use the standard solver
        use_standard = _env("USE_STANDARD_SOLVER", "true").lower() == "true"

        if use_standard:
            return StandardProblemSolver(explainer, audio_file_path)
        else:
            # Fall back to original solver if explicitly disabled
            from .solver import ProblemSolver
            return ProblemSolver(explainer)