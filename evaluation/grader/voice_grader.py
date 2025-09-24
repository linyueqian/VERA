from __future__ import annotations

import os
import asyncio
from typing import Optional, Dict, Any
from pathlib import Path

from .base import BaseAccuracyGrader, GradeLabel, GradeResult
from .llm_grader import LLMAccuracyGrader
from .asr_processor import ASRProcessor
from .wer_calculator import WERCalculator
from .llm_normalizer import LLMAwareWERCalculator


class VoiceAccuracyGrader(BaseAccuracyGrader):
    """
    Voice accuracy grader that processes audio files through ASR then uses LLM grading.
    
    Pipeline:
    1. Audio file → ASR → transcript 
    2. Transcript → LLM grader → accuracy grade
    3. Also calculates WER between ASR transcript and expected text
    """
    
    def __init__(
        self,
        asr_provider: str = "azure",
        llm_deployment_name: str = "gpt-4o", 
        llm_api_version: str = "2024-10-21",
        llm_temperature: float = 0.0,
        max_retries: int = 3,
        base_delay: float = 1.0,
        azure_speech_key: Optional[str] = None,
        azure_speech_region: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        openai_base_url: Optional[str] = None,
    ):
        # Initialize ASR processor
        self.asr_processor = ASRProcessor(
            provider=asr_provider,
            azure_speech_key=azure_speech_key,
            azure_speech_region=azure_speech_region, 
            openai_api_key=openai_api_key,
            openai_base_url=openai_base_url,
        )
        
        # Initialize LLM grader for semantic evaluation
        self.llm_grader = LLMAccuracyGrader(
            deployment_name=llm_deployment_name,
            api_version=llm_api_version,
            temperature=llm_temperature,
            max_retries=max_retries,
            base_delay=base_delay,
        )
        
        # WER calculator with LLM normalization for transcript quality
        self.wer_calculator = WERCalculator()
        self.llm_wer_calculator = LLMAwareWERCalculator()
    
    def _extract_audio_path_from_response(self, response_data: Dict[str, Any]) -> Optional[str]:
        """Extract audio file path from voice output response data."""
        if isinstance(response_data, dict):
            # Check audio_info section 
            if "audio_info" in response_data and "output_file" in response_data["audio_info"]:
                return response_data["audio_info"]["output_file"]
            
            # Check for direct audio path
            if "output_audio_path" in response_data:
                return response_data["output_audio_path"]
                
            # Check conversation transcript for audio response
            if "conversation_transcript" in response_data:
                for turn in response_data["conversation_transcript"]:
                    if turn.get("type") == "audio_response" and "audio_file" in turn:
                        return turn["audio_file"]
        
        return None
    
    async def grade_voice_response_async(
        self,
        question: str,
        ground_truth: str,
        voice_response_path_or_data: str | Dict[str, Any],
        expected_transcript: Optional[str] = None,
        benchmark: Optional[str] = None,
        calculate_wer: bool = True,
    ) -> Dict[str, Any]:
        """
        Grade a voice response (audio file or response data structure).
        
        Args:
            question: The question asked
            ground_truth: Ground truth answer
            voice_response_path_or_data: Path to audio file OR response data dict
            expected_transcript: Expected transcript text (for WER calculation)
            benchmark: Benchmark name for grading context
            calculate_wer: Whether to calculate WER metrics
            
        Returns:
            Dictionary containing grading results and ASR/WER metrics
        """
        # Extract audio file path
        if isinstance(voice_response_path_or_data, str):
            audio_path = voice_response_path_or_data
            response_data = None
        else:
            response_data = voice_response_path_or_data
            audio_path = self._extract_audio_path_from_response(response_data)
            
            if not audio_path:
                return {
                    "success": False,
                    "error": "Could not find audio file path in response data",
                    "asr_result": None,
                    "llm_grade": None,
                    "wer_metrics": None
                }
        
        # Ensure audio file exists
        audio_path = Path(audio_path)
        if not audio_path.exists():
            return {
                "success": False,
                "error": f"Audio file not found: {audio_path}",
                "asr_result": None,
                "llm_grade": None, 
                "wer_metrics": None
            }
        
        # Step 1: Transcribe audio
        asr_result = await self.asr_processor.transcribe_async(str(audio_path))
        
        if not asr_result["success"]:
            return {
                "success": False,
                "error": f"ASR failed: {asr_result['error']}",
                "asr_result": asr_result,
                "llm_grade": None,
                "wer_metrics": None
            }
        
        transcript = asr_result["text"]
        
        # Step 2: Grade transcript using LLM
        try:
            llm_grade = await self.llm_grader.grade_async(
                question=question,
                ground_truth=ground_truth,
                predicted_answer=transcript,
                benchmark=benchmark
            )
        except Exception as e:
            return {
                "success": False,
                "error": f"LLM grading failed: {str(e)}",
                "asr_result": asr_result,
                "llm_grade": None,
                "wer_metrics": None
            }
        
        # Step 3: Calculate WER if expected transcript provided
        wer_metrics = None
        if calculate_wer and expected_transcript:
            # Use LLM-aware WER calculation for better mathematical content handling
            wer_metrics = await self.llm_wer_calculator.calculate_normalized_wer_async(
                reference=expected_transcript,
                hypothesis=transcript,
                return_details=True,
                normalize_both=True  # Normalize both reference and hypothesis
            )
        
        return {
            "success": True,
            "error": None,
            "asr_result": asr_result,
            "llm_grade": llm_grade,
            "wer_metrics": wer_metrics,
            "transcript": transcript,
            "audio_path": str(audio_path)
        }
    
    def grade_voice_response(
        self,
        question: str,
        ground_truth: str,
        voice_response_path_or_data: str | Dict[str, Any],
        expected_transcript: Optional[str] = None,
        benchmark: Optional[str] = None,
        calculate_wer: bool = True,
    ) -> Dict[str, Any]:
        """Sync wrapper for voice response grading."""
        async def _run():
            return await self.grade_voice_response_async(
                question, ground_truth, voice_response_path_or_data,
                expected_transcript, benchmark, calculate_wer
            )
        
        try:
            return asyncio.run(_run())
        except RuntimeError:
            # If already inside an event loop
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(_run())
    
    def batch_grade_voice_responses(
        self,
        grading_tasks: list[Dict[str, Any]],
    ) -> list[Dict[str, Any]]:
        """
        Grade multiple voice responses in batch.
        
        Args:
            grading_tasks: List of dicts with keys:
                - question: str
                - ground_truth: str  
                - voice_response_path_or_data: str | Dict
                - expected_transcript: Optional[str]
                - benchmark: Optional[str]
                - calculate_wer: Optional[bool] = True
        """
        async def _batch_grade():
            tasks = []
            for task in grading_tasks:
                tasks.append(self.grade_voice_response_async(
                    question=task["question"],
                    ground_truth=task["ground_truth"],
                    voice_response_path_or_data=task["voice_response_path_or_data"],
                    expected_transcript=task.get("expected_transcript"),
                    benchmark=task.get("benchmark"),
                    calculate_wer=task.get("calculate_wer", True)
                ))
            
            return await asyncio.gather(*tasks, return_exceptions=True)
        
        try:
            results = asyncio.run(_batch_grade())
        except RuntimeError:
            loop = asyncio.get_event_loop()
            results = loop.run_until_complete(_batch_grade())
        
        # Handle exceptions in results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append({
                    "success": False,
                    "error": f"Exception in task {i}: {str(result)}",
                    "asr_result": None,
                    "llm_grade": None,
                    "wer_metrics": None
                })
            else:
                processed_results.append(result)
        
        return processed_results
    
    # Implement base class interface for compatibility
    def grade(
        self,
        question: str,
        ground_truth: str,
        predicted_answer: str | Dict[str, Any],  # Can be transcript or voice response data
        benchmark: Optional[str] = None,
    ) -> GradeResult:
        """
        Grade method for base class compatibility.
        
        If predicted_answer is a string, treat as transcript and grade directly.
        If predicted_answer is a dict, treat as voice response data and process through ASR.
        """
        if isinstance(predicted_answer, str):
            # Direct transcript grading
            return self.llm_grader.grade(
                question=question,
                ground_truth=ground_truth, 
                predicted_answer=predicted_answer,
                benchmark=benchmark
            )
        elif isinstance(predicted_answer, dict):
            # Voice response grading
            result = self.grade_voice_response(
                question=question,
                ground_truth=ground_truth,
                voice_response_path_or_data=predicted_answer,
                benchmark=benchmark
            )
            
            if result["success"]:
                return result["llm_grade"]
            else:
                # Return error as incorrect grade
                return GradeResult(
                    label=GradeLabel.INCORRECT,
                    extracted_final_answer=None,
                    reasoning=result["error"],
                    correct_flag=False,
                    confidence=None,
                    raw_model_output=None,
                    metadata={"voice_grading_error": result["error"]}
                )
        else:
            raise ValueError(f"Invalid predicted_answer type: {type(predicted_answer)}")