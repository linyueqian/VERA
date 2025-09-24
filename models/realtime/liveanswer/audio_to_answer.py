import os
import sys
import time
import json
from pathlib import Path
from typing import Tuple, Optional

from .main import main_request
from .stt_service import AzureSTTService


class AudioToAnswer:
    """Process audio input to generate audio answer using Azure STT + LiveAnswer."""

    def __init__(self,
                 speech_key: Optional[str] = None,
                 speech_region: Optional[str] = None,
                 enable_diarization: bool = False):
        """
        Initialize the audio-to-answer pipeline.

        Args:
            speech_key: Azure Speech API key
            speech_region: Azure region
            enable_diarization: Whether to use speaker diarization
        """
        self.stt_service = AzureSTTService(speech_key, speech_region)
        self.enable_diarization = enable_diarization

    def process_audio_file(self,
                          audio_file_path: str,
                          output_dir: Optional[str] = None,
                          verbose: bool = True) -> Tuple[str, bytes, dict]:
        """
        Process audio file: STT -> LiveAnswer -> TTS.

        Args:
            audio_file_path: Path to input audio file
            output_dir: Directory for output files (defaults to current dir)
            verbose: Print progress messages

        Returns:
            Tuple of (transcript, answer_audio_bytes, metadata)
        """
        start_time = time.time()

        # Step 1: Transcribe audio
        if verbose:
            print(f"[1/3] Transcribing audio file: {audio_file_path}")

        if self.enable_diarization:
            transcript, stt_metadata = self.stt_service.transcribe_with_diarization(audio_file_path)
        else:
            transcript, stt_metadata = self.stt_service.transcribe_file(audio_file_path)

        transcription_time = time.time() - start_time

        if verbose:
            print(f"    Transcription: '{transcript[:100]}{'...' if len(transcript) > 100 else ''}'")
            print(f"    Time taken: {transcription_time:.2f}s")

        # Step 2: Generate answer
        if verbose:
            print(f"[2/3] Generating answer...")

        answer_start = time.time()
        answer_audio_bytes, time_to_first_response, gpt5_response, groq_explanation = main_request(transcript, audio_file_path)
        answer_time = time.time() - answer_start

        if verbose:
            print(f"    Time to first response: {time_to_first_response:.2f}s")
            print(f"    Total generation time: {answer_time:.2f}s")

        # Step 3: Save outputs
        if output_dir is None:
            output_dir = os.getcwd()
        else:
            os.makedirs(output_dir, exist_ok=True)

        # Save answer audio
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        answer_path = Path(output_dir) / f"answer_{timestamp}.mp3"
        answer_path.write_bytes(answer_audio_bytes)

        # Save transcript
        transcript_path = Path(output_dir) / f"transcript_{timestamp}.txt"
        transcript_path.write_text(transcript)

        # Save GPT-5 response (raw solver output)
        gpt5_response_path = Path(output_dir) / f"gpt5_response_{timestamp}.txt"
        gpt5_response_path.write_text(gpt5_response)

        # Save Groq explanation (what was spoken)
        groq_explanation_path = Path(output_dir) / f"groq_explanation_{timestamp}.txt"
        groq_explanation_path.write_text(groq_explanation)

        # Save detailed timing info
        timing_path = Path(output_dir) / f"timing_{timestamp}.json"
        timing_data = {
            'time_to_first_audio_chunk': time_to_first_response,
            'transcription_time': transcription_time,
            'answer_generation_time': answer_time,
            'total_processing_time': time.time() - start_time,
            'transcript_length_chars': len(transcript),
            'audio_output_size_bytes': len(answer_audio_bytes),
            'timestamp': timestamp
        }
        timing_path.write_text(json.dumps(timing_data, indent=2))

        if verbose:
            print(f"[3/3] Outputs saved:")
            print(f"    Answer audio: {answer_path}")
            print(f"    Transcript: {transcript_path}")
            print(f"    GPT-5 response: {gpt5_response_path}")
            print(f"    Groq explanation: {groq_explanation_path}")
            print(f"    Timing data: {timing_path}")

        # Compile metadata
        metadata = {
            'input_audio': audio_file_path,
            'transcript': transcript,
            'gpt5_response': gpt5_response,
            'groq_explanation': groq_explanation,
            'transcript_length': len(transcript),
            'stt_metadata': stt_metadata,
            'answer_audio_path': str(answer_path),
            'transcript_path': str(transcript_path),
            'gpt5_response_path': str(gpt5_response_path),
            'groq_explanation_path': str(groq_explanation_path),
            'timing_path': str(timing_path),
            'timings': {
                'transcription_time': transcription_time,
                'answer_generation_time': answer_time,
                'time_to_first_response': time_to_first_response,
                'total_time': time.time() - start_time
            }
        }

        return transcript, answer_audio_bytes, metadata

    def process_audio_stream(self, audio_stream):
        """
        Future: Process audio stream in real-time.
        Currently not implemented - placeholder for future enhancement.
        """
        raise NotImplementedError(
            "Real-time audio streaming not yet implemented. "
            "Use process_audio_file() for file-based processing."
        )


def main():
    """CLI entry point for audio-to-answer processing."""
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Process audio to generate answer")
    parser.add_argument("audio_file", help="Path to input audio file")
    parser.add_argument("--output-dir", default=None, help="Output directory")
    parser.add_argument("--diarization", action="store_true", help="Enable speaker diarization")
    parser.add_argument("--save-metadata", action="store_true", help="Save metadata JSON")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output")

    args = parser.parse_args()

    try:
        # Initialize processor
        processor = AudioToAnswer(enable_diarization=args.diarization)

        # Process audio
        transcript, audio_bytes, metadata = processor.process_audio_file(
            audio_file_path=args.audio_file,
            output_dir=args.output_dir,
            verbose=not args.quiet
        )

        # Optionally save metadata
        if args.save_metadata:
            metadata_path = Path(args.output_dir or os.getcwd()) / "metadata.json"
            # Convert metadata to JSON-serializable format
            json_metadata = {
                k: v if not isinstance(v, bytes) else f"<bytes: {len(v)} bytes>"
                for k, v in metadata.items()
            }
            metadata_path.write_text(json.dumps(json_metadata, indent=2))
            if not args.quiet:
                print(f"    Metadata: {metadata_path}")

        if not args.quiet:
            print(f"\nProcessing complete! Total time: {metadata['timings']['total_time']:.2f}s")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()