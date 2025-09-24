from __future__ import annotations

import os
import asyncio
import httpx
from typing import Optional, Dict, Any


class LLMTextNormalizer:
    """
    LLM-based text normalizer for mathematical and technical content.
    
    Uses LLM to convert spoken/transcribed text to canonical written form
    before WER calculation, especially useful for mathematical expressions.
    """
    
    def __init__(
        self,
        deployment_name: str = "gpt-4o",
        api_version: str = "2024-10-21", 
        temperature: float = 0.0,
        max_retries: int = 2,
    ):
        self.azure_endpoint = (os.getenv("AZURE_OPENAI_ENDPOINT") or "").rstrip("/")
        self.api_key = os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("AZURE_API_KEY") 
        self.deployment_name = deployment_name
        self.api_version = api_version
        self.temperature = temperature
        self.max_retries = max_retries
        
        self.normalization_prompt = """You are a text normalizer for mathematical and technical content. Your job is to convert spoken/transcribed text into standard written mathematical notation.

Key transformations:
- Convert spelled numbers to digits: "two" → "2", "sixteen" → "16"
- Convert function notation: "P of x" → "P(x)", "f of sixteen" → "f(16)"
- Convert mathematical operations: "plus" → "+", "minus" → "-", "times" → "×" or "*"
- Convert mathematical expressions: "x squared" → "x²" or "x^2"
- Convert equations: "equals" → "="
- Convert mathematical terms consistently
- Preserve mathematical meaning exactly

Return ONLY the normalized text, no explanations or comments.

Examples:
Input: "P of x equals two x squared plus three x plus one"
Output: "P(x) = 2x² + 3x + 1"

Input: "The leading coefficient for P of x is two and for Q of x it's negative two"  
Output: "The leading coefficient for P(x) is 2 and for Q(x) it's -2"

Input: "f of sixteen equals fifty four"
Output: "f(16) = 54"
"""

    def _ensure_env(self) -> None:
        if not self.azure_endpoint or not self.api_key:
            raise RuntimeError("LLMTextNormalizer requires AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY")

    async def _call_llm(self, text: str) -> str:
        """Call LLM to normalize text."""
        self._ensure_env()
        
        url = f"{self.azure_endpoint}/openai/deployments/{self.deployment_name}/chat/completions"
        headers = {"Content-Type": "application/json", "api-key": self.api_key}
        params = {"api-version": self.api_version}
        
        payload = {
            "messages": [
                {"role": "system", "content": self.normalization_prompt},
                {"role": "user", "content": f"Normalize this text:\n{text}"}
            ],
            "temperature": self.temperature,
            "max_tokens": 1000,
        }
        
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    r = await client.post(url, headers=headers, params=params, json=payload)
                    r.raise_for_status()
                    data = r.json()
                    return data["choices"][0]["message"]["content"].strip()
            except Exception as e:
                last_exception = e
                if attempt < self.max_retries:
                    await asyncio.sleep(1.0 * (attempt + 1))
                    
        # If all retries failed, return original text
        print(f"LLM normalization failed: {last_exception}, returning original text")
        return text
    
    def normalize_text(self, text: str) -> str:
        """Synchronous text normalization."""
        try:
            return asyncio.run(self._normalize_async(text))
        except RuntimeError:
            # If already in event loop
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(self._normalize_async(text))
    
    async def _normalize_async(self, text: str) -> str:
        """Async text normalization."""
        if not text or not text.strip():
            return text
            
        return await self._call_llm(text)
    
    async def normalize_batch_async(self, texts: list[str]) -> list[str]:
        """Normalize multiple texts in parallel."""
        tasks = [self._normalize_async(text) for text in texts]
        return await asyncio.gather(*tasks, return_exceptions=False)
    
    def normalize_batch(self, texts: list[str]) -> list[str]:
        """Synchronous batch normalization."""
        try:
            return asyncio.run(self.normalize_batch_async(texts))
        except RuntimeError:
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(self.normalize_batch_async(texts))


class LLMAwareWERCalculator:
    """WER Calculator with LLM-based text normalization."""
    
    def __init__(self, normalizer: Optional[LLMTextNormalizer] = None):
        self.normalizer = normalizer or LLMTextNormalizer()
        
        # Import here to avoid circular imports
        from .wer_calculator import WERCalculator
        self.wer_calculator = WERCalculator()
    
    def calculate_normalized_wer(
        self, 
        reference: str, 
        hypothesis: str, 
        return_details: bool = False,
        normalize_both: bool = True
    ) -> Dict[str, Any]:
        """
        Calculate WER with LLM normalization.
        
        Args:
            reference: Ground truth text
            hypothesis: Predicted text (e.g., ASR output)
            return_details: Include detailed alignment info
            normalize_both: If True, normalize both texts. If False, only normalize hypothesis.
        
        Returns:
            Dict with standard WER, normalized WER, and normalization details
        """
        # Calculate standard WER first
        standard_result = self.wer_calculator.calculate_wer(reference, hypothesis, return_details)
        
        try:
            # Normalize texts using LLM
            if normalize_both:
                normalized_ref = self.normalizer.normalize_text(reference)
                normalized_hyp = self.normalizer.normalize_text(hypothesis)
            else:
                normalized_ref = reference
                normalized_hyp = self.normalizer.normalize_text(hypothesis)
            
            # Calculate normalized WER
            normalized_result = self.wer_calculator.calculate_wer(
                normalized_ref, normalized_hyp, return_details
            )
            
            # Combine results
            result = {
                "standard_wer": standard_result["wer"],
                "normalized_wer": normalized_result["wer"],
                "wer_improvement": standard_result["wer"] - normalized_result["wer"],
                "normalization_applied": True,
                "normalized_reference": normalized_ref,
                "normalized_hypothesis": normalized_hyp,
                "original_reference": reference,
                "original_hypothesis": hypothesis,
            }
            
            # Use normalized WER as primary metric
            result.update({
                "wer": normalized_result["wer"],
                "substitutions": normalized_result["substitutions"],
                "deletions": normalized_result["deletions"],
                "insertions": normalized_result["insertions"],
                "total_words": normalized_result["total_words"],
                "reference_length": normalized_result["reference_length"],
                "hypothesis_length": normalized_result["hypothesis_length"],
            })
            
            if return_details:
                result.update({
                    "standard_details": standard_result,
                    "normalized_details": normalized_result,
                    "alignment": normalized_result.get("alignment"),
                    "reference_words": normalized_result.get("reference_words"),
                    "hypothesis_words": normalized_result.get("hypothesis_words"),
                })
                
        except Exception as e:
            print(f"LLM normalization failed: {e}, falling back to standard WER")
            result = standard_result.copy()
            result.update({
                "standard_wer": standard_result["wer"],
                "normalized_wer": standard_result["wer"], 
                "wer_improvement": 0.0,
                "normalization_applied": False,
                "normalization_error": str(e),
            })
        
        return result
    
    async def calculate_normalized_wer_async(
        self,
        reference: str,
        hypothesis: str, 
        return_details: bool = False,
        normalize_both: bool = True
    ) -> Dict[str, Any]:
        """Async version of normalized WER calculation."""
        # Calculate standard WER
        standard_result = self.wer_calculator.calculate_wer(reference, hypothesis, return_details)
        
        try:
            # Normalize texts
            if normalize_both:
                normalized_ref, normalized_hyp = await asyncio.gather(
                    self.normalizer._normalize_async(reference),
                    self.normalizer._normalize_async(hypothesis)
                )
            else:
                normalized_ref = reference
                normalized_hyp = await self.normalizer._normalize_async(hypothesis)
            
            # Calculate normalized WER
            normalized_result = self.wer_calculator.calculate_wer(
                normalized_ref, normalized_hyp, return_details
            )
            
            result = {
                "standard_wer": standard_result["wer"],
                "normalized_wer": normalized_result["wer"],
                "wer_improvement": standard_result["wer"] - normalized_result["wer"],
                "normalization_applied": True,
                "normalized_reference": normalized_ref,
                "normalized_hypothesis": normalized_hyp,
            }
            
            result.update({
                "wer": normalized_result["wer"],
                "substitutions": normalized_result["substitutions"],
                "deletions": normalized_result["deletions"],
                "insertions": normalized_result["insertions"],
                "total_words": normalized_result["total_words"],
                "reference_length": normalized_result["reference_length"],
                "hypothesis_length": normalized_result["hypothesis_length"],
            })
            
            if return_details:
                result.update({
                    "standard_details": standard_result,
                    "normalized_details": normalized_result,
                    "alignment": normalized_result.get("alignment"),
                })
                
        except Exception as e:
            result = standard_result.copy()
            result.update({
                "standard_wer": standard_result["wer"],
                "normalized_wer": standard_result["wer"],
                "wer_improvement": 0.0,
                "normalization_applied": False,
                "normalization_error": str(e),
            })
        
        return result