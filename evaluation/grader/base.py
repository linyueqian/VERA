from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, Any


class GradeLabel(str, Enum):
    CORRECT = "CORRECT"
    INCORRECT = "INCORRECT"
    NOT_ATTEMPTED = "NOT_ATTEMPTED"


@dataclass
class GradeResult:
    label: GradeLabel
    extracted_final_answer: Optional[str] = None
    reasoning: Optional[str] = None
    correct_flag: Optional[bool] = None
    confidence: Optional[float] = None  # 0-100
    raw_model_output: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class BaseAccuracyGrader:
    """Interface for accuracy graders.

    Implementations should focus on judging whether a predicted response
    answers the question correctly with respect to a gold target.
    """

    def grade(
        self,
        question: str,
        gold_target: str,
        predicted_answer: str,
        benchmark: Optional[str] = None,
        mode: str = "triad",  # "triad" -> A/B/C, "binary" -> yes/no
    ) -> GradeResult:
        raise NotImplementedError

