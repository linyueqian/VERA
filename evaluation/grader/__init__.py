"""Grader package: accuracy-first grading utilities.

This module provides:
- Prompt templates per benchmark
- Grader base classes and result types
- Heuristic and LLM-backed graders
- Voice evaluation with ASR and WER calculation
- A small CLI for grading single triplets or batch outputs
"""

from .base import GradeResult, GradeLabel
from .prompts import get_accuracy_prompt
from .llm_grader import LLMAccuracyGrader
from .voice_grader import VoiceAccuracyGrader
from .asr_processor import ASRProcessor
from .wer_calculator import WERCalculator

__all__ = [
    "GradeResult",
    "GradeLabel",
    "get_accuracy_prompt",
    "LLMAccuracyGrader",
    "VoiceAccuracyGrader",
    "ASRProcessor",
    "WERCalculator",
]
