from __future__ import annotations

import os
import httpx
import asyncio
import time
import random
from typing import Optional, Tuple

from .base import BaseAccuracyGrader, GradeLabel, GradeResult
from .prompts import get_accuracy_prompt


class LLMAccuracyGrader(BaseAccuracyGrader):
    """LLM-backed accuracy grader using Azure OpenAI chat completions.

    Notes:
    - Requires environment variables: AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY
    - By default uses deployment "gpt-4o" and api-version "2024-10-21"
    - Does not stream; single-turn prompt per grading task
    """

    def __init__(
        self,
        deployment_name: str = "gpt-4o",
        api_version: str = "2024-10-21",
        temperature: float = 0.0,
        max_retries: int = 3,
        base_delay: float = 1.0,
    ) -> None:
        self.azure_endpoint = (os.getenv("AZURE_OPENAI_ENDPOINT") or "").rstrip("/")
        self.api_key = os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("AZURE_API_KEY")
        self.deployment_name = deployment_name
        self.api_version = api_version
        self.temperature = temperature
        self.max_retries = max_retries
        self.base_delay = base_delay

    def _ensure_env(self) -> None:
        if not self.azure_endpoint or not self.api_key:
            raise RuntimeError(
                "LLMAccuracyGrader requires AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY"
            )

    async def _chat(self, system: str, user: str) -> str:
        self._ensure_env()
        url = f"{self.azure_endpoint}/openai/deployments/{self.deployment_name}/chat/completions"
        headers = {"Content-Type": "application/json", "api-key": self.api_key}
        params = {"api-version": self.api_version}
        payload = {
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": self.temperature,
            "max_tokens": 512,
        }
        
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                async with httpx.AsyncClient(timeout=60.0) as client:
                    r = await client.post(url, headers=headers, params=params, json=payload)
                    r.raise_for_status()
                    data = r.json()
                    return data["choices"][0]["message"]["content"].strip()
                    
            except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPStatusError) as e:
                last_exception = e
                
                if attempt == self.max_retries:
                    # Last attempt failed, re-raise the exception
                    break
                
                # Calculate delay with exponential backoff and jitter
                delay = self.base_delay * (2 ** attempt) + random.uniform(0, 1)
                
                # Special handling for rate limits (429)
                if isinstance(e, httpx.HTTPStatusError) and e.response.status_code == 429:
                    # For rate limits, wait longer
                    delay = max(delay, 5.0 + random.uniform(0, 5))
                    print(f"Rate limit hit, retrying in {delay:.1f}s (attempt {attempt + 1}/{self.max_retries + 1})")
                elif isinstance(e, httpx.ConnectError):
                    print(f"Connection error, retrying in {delay:.1f}s (attempt {attempt + 1}/{self.max_retries + 1})")
                elif isinstance(e, httpx.TimeoutException):
                    print(f"Timeout error, retrying in {delay:.1f}s (attempt {attempt + 1}/{self.max_retries + 1})")
                else:
                    print(f"HTTP error {e.response.status_code if hasattr(e, 'response') else 'unknown'}, retrying in {delay:.1f}s (attempt {attempt + 1}/{self.max_retries + 1})")
                
                await asyncio.sleep(delay)
        
        # If we get here, all retries failed
        raise last_exception

    def _parse_binary(self, content: str) -> Tuple[Optional[str], Optional[bool], Optional[float], Optional[str]]:
        # Very light parsing for fields we care about
        extracted = None
        correct_flag = None
        confidence = None
        reasoning = None
        for line in content.splitlines():
            l = line.strip()
            if l.lower().startswith("extracted_final_answer:"):
                extracted = l.split(":", 1)[1].strip()
                extracted = None if extracted.lower() == "none" else extracted
            elif l.lower().startswith("correct:"):
                v = l.split(":", 1)[1].strip().lower()
                if v in {"yes", "no"}:
                    correct_flag = v == "yes"
            elif l.lower().startswith("confidence:"):
                v = l.split(":", 1)[1].strip().replace("%", "")
                try:
                    confidence = float(v)
                except Exception:
                    confidence = None
            elif l.lower().startswith("reasoning:"):
                reasoning = l.split(":", 1)[1].strip()
        return extracted, correct_flag, confidence, reasoning

    def _parse_triad(self, content: str) -> GradeLabel:
        c = content.strip().upper()
        if c.startswith("A"):
            return GradeLabel.CORRECT
        if c.startswith("B"):
            return GradeLabel.INCORRECT
        if c.startswith("C"):
            return GradeLabel.NOT_ATTEMPTED
        # default fallback if model deviates
        return GradeLabel.INCORRECT

    async def grade_async(
        self,
        question: str,
        ground_truth: str,
        predicted_answer: str,
        benchmark: Optional[str] = None,
    ) -> GradeResult:
        prompt = get_accuracy_prompt(
            question=question, ground_truth=ground_truth, predicted_answer=predicted_answer, benchmark=benchmark
        )

        # Constrain output: A/B/C only
        system = (
            "You are an academic grader. Return only a single capital letter (A/B/C) per instructions."
        )

        content = await self._chat(system=system, user=prompt)

        # triad mode
        label = self._parse_triad(content)
        return GradeResult(
            label=label,
            extracted_final_answer=None,
            reasoning=None,
            correct_flag=None,
            confidence=None,
            raw_model_output=content,
        )

    def grade(
        self,
        question: str,
        ground_truth: str,
        predicted_answer: str,
        benchmark: Optional[str] = None,
    ) -> GradeResult:
        async def _run():
            return await self.grade_async(question, ground_truth, predicted_answer, benchmark)
        try:
            return asyncio.run(_run())
        except RuntimeError:
            # If already inside an event loop
            loop = asyncio.get_event_loop()  # type: ignore
            return loop.run_until_complete(_run())
