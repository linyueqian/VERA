from __future__ import annotations

from typing import Optional


# Triad accuracy template (A/B/C)
GRADER_TEMPLATE_TRIAD = (
    """
Your job is to look at a question, a ground truth answer, and a predicted answer, and then assign a grade of either ["CORRECT", "INCORRECT", "NOT_ATTEMPTED"].
First, examples for calibration are provided. Then, only return A/B/C for the new case.

The following are examples of CORRECT predicted answers.
```
Question: What are the names of Barack Obama's children?
Ground truth: Malia Obama and Sasha Obama
Predicted answer 1: sasha and malia obama
Predicted answer 2: most people would say Malia and Sasha, but I'm not sure and would have to double check
Predicted answer 3: Barack Obama has two daughters. Their names are Malia Ann and Natasha Marian, but they are commonly referred to as Malia Obama and Sasha Obama. Malia was born on July 4, 1998, and Sasha was born on June 10, 2001.
```
These predicted answers are all CORRECT because:
    - They fully contain the important information in the ground truth.
    - They do not contain any information that contradicts the ground truth.
    - Only semantic meaning matters; capitalization, punctuation, grammar, and order don't matter.
    - Hedging and guessing are permissible, provided the response fully includes the ground truth and contains no contradictions.

The following are examples of INCORRECT predicted answers.
```
Question: What are the names of Barack Obama's children?
Ground truth: Malia and Sasha
Predicted answer 1: Malia.
Predicted answer 2: Malia, Sasha, and Susan.
Predicted answer 3: Barack Obama does not have any children.
Predicted answer 4: I think it's either Malia and Sasha. Or it could be Malia and Jackie. Or it could be Joey and Malia.
Predicted answer 5: While I don't know their exact names, I can tell you that Barack Obama has three children.
Predicted answer 6: It may be the case that Obama's child is named James. However, it's recommended to confirm with updated references.
```
These predicted answers are all INCORRECT because:
    - A factual statement contradicts the ground truth. Hedged incorrect statements are still incorrect.

The following are examples of NOT_ATTEMPTED predicted answers.
```
Question: What are the names of Barack Obama's children?
Ground truth: Malia and Sasha
Predicted answer 1: I don't know.
Predicted answer 2: I need more context about which Obama you are talking about.
Predicted answer 3: Without researching the web, I cannot answer this question. However, I can tell you that Barack Obama has two children.
Predicted answer 4: Barack Obama has two children. I know that one of them is Malia, but I'm not sure about the other one.
```
These are NOT_ATTEMPTED because:
    - The important information in the ground truth is not included, and there is no direct contradiction.

Additional guidance:
- For numerical ground truths, accept answers correct to the last significant figure (e.g., 120k ~ 115k–124k).
- The ground truth may contain more information than the question; the predicted answer only needs to cover what's asked.
- Obvious inferences are allowed (e.g., omit "California" when "San Francisco, California" is the target for "What city...").
- Reasonable typos in names are acceptable if clearly the same entity.

Here is a new example. Reply with A/B/C only, no extra text.
```
Ground truth: {ground_truth}
Predicted answer: {predicted_answer}
```

Grade the predicted answer as one of:
A: CORRECT
B: INCORRECT
C: NOT_ATTEMPTED

Just return the single letter: "A", "B", or "C".
"""
    .strip()
)


def _benchmark_header(benchmark: Optional[str]) -> str:
    if not benchmark:
        return ""
    # Short, domain-aware header if desired
    bm = benchmark.lower()
    if "aime" in bm:
        return (
            "Mathematical QA: judge strictly on the final numeric/value answer. "
            "Allow only minimal rounding consistent with the ground truth; reject paraphrases."
        )
    if "browsecomp" in bm:
        return (
            "Web QA: prioritize factual exactness over citations; judge only whether the predicted answer matches the ground truth."
        )
    if "gpqa" in bm:
        return (
            "Graduate-level science QA: semantic equivalence is acceptable if factually identical; contradictions are incorrect."
        )
    if "mrcr" in bm:
        return (
            "Long-context needle retrieval: mark CORRECT only if the predicted answer contains the exact ground truth string "
            "as a contiguous span (case-insensitive). Paraphrases, substitutions, or partial matches are INCORRECT. "
            "Ignore surrounding commentary; focus solely on inclusion of the exact phrase."
        )
    if "simpleqa" in bm:
        return (
            "Simple factual recall: require the predicted answer to match the ground truth entity/value. "
            "Minor spelling variations are acceptable only if clearly the same name."
        )
    return ""


def get_accuracy_prompt(
    question: Optional[str],
    ground_truth: str,
    predicted_answer: str,
    benchmark: Optional[str] = None,
) -> str:
    """Return a benchmark-aware accuracy grading prompt (triad A/B/C)."""
    header = _benchmark_header(benchmark)

    # Triad prompt (A/B/C)
    core = GRADER_TEMPLATE_TRIAD.format(
        ground_truth=ground_truth,
        predicted_answer=predicted_answer,
    )
    return f"{header}\n\n{core}".strip() if header else core
