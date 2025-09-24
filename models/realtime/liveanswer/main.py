import threading
from pathlib import Path
from typing import List, Optional

from .explain import ExplainSynthesizer
from .audio import AudioGenerator
from .solver_standard import StandardProblemSolver as ProblemSolver


def main_request(request: str, audio_file_path: Optional[str] = None) -> tuple[bytes, float, str, str]:
    """
    Orchestrates the pipeline:
    - Create ExplainSynthesizer
    - Run AudioGenerator and ProblemSolver concurrently
    - Return resulting MP3 bytes, time to first response, GPT-5 response, and Groq explanation
    """

    explainer = ExplainSynthesizer(request=request)
    audio_gen = AudioGenerator(explainer=explainer)
    solver = ProblemSolver(explainer=explainer, audio_file_path=audio_file_path)

    audio_bytes_holder: List[bytes] = []
    time_to_first_response_holder: List[float] = []

    def run_audio():
        audio_bytes, time_to_first_response = audio_gen.start()
        audio_bytes_holder.append(audio_bytes)
        time_to_first_response_holder.append(time_to_first_response)

    t_audio = threading.Thread(target=run_audio, name="audio_gen")
    t_solver = threading.Thread(target=lambda: solver.start(request), name="problem_solver")

    t_audio.start()
    t_solver.start()

    t_audio.join()
    t_solver.join()

    audio_bytes = audio_bytes_holder[0] if audio_bytes_holder else b""
    time_to_first_response = time_to_first_response_holder[0] if time_to_first_response_holder else 0.0

    # Get both the GPT-5 response and the Groq explanation
    gpt5_response = getattr(explainer, 'gpt5_response', 'GPT-5 response not captured')
    groq_explanation = getattr(explainer, 'spoken_explanation', 'Groq explanation not captured')

    if audio_bytes:
        output_path = Path(__file__).resolve().parents[1] / "liveanswer-output.mp3"
        output_path.write_bytes(audio_bytes)

    return audio_bytes, time_to_first_response, gpt5_response, groq_explanation

