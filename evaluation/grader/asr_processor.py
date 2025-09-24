from __future__ import annotations

import os
import asyncio
import tempfile
import subprocess
import json
import time
import re
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path

try:
    import azure.cognitiveservices.speech as speechsdk
    AZURE_AVAILABLE = True
except ImportError:
    AZURE_AVAILABLE = False

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from groq import Groq, RateLimitError
    from pydub import AudioSegment
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False


def _preprocess_audio_for_groq(input_path: Path) -> Path:
    """
    Preprocess audio file to 16kHz mono FLAC using ffmpeg for Groq.
    FLAC provides lossless compression for faster upload times.
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    with tempfile.NamedTemporaryFile(suffix='.flac', delete=False) as temp_file:
        output_path = Path(temp_file.name)

    try:
        subprocess.run([
            'ffmpeg',
            '-hide_banner',
            '-loglevel', 'error',
            '-i', str(input_path),
            '-ar', '16000',
            '-ac', '1',
            '-c:a', 'flac',
            '-y',
            str(output_path)
        ], check=True)
        return output_path
    except subprocess.CalledProcessError as e:
        output_path.unlink(missing_ok=True)
        raise RuntimeError(f"FFmpeg conversion failed: {e}")


def _find_longest_common_sequence(sequences: List[str], match_by_words: bool = True) -> str:
    """
    Find the optimal alignment between sequences with longest common sequence and sliding window matching.
    """
    if not sequences:
        return ""

    # Convert input based on matching strategy
    if match_by_words:
        sequences = [
            [word for word in re.split(r'(\s+\w+)', seq) if word]
            for seq in sequences
        ]
    else:
        sequences = [list(seq) for seq in sequences]

    left_sequence = sequences[0]
    left_length = len(left_sequence)
    total_sequence = []

    for right_sequence in sequences[1:]:
        max_matching = 0.0
        right_length = len(right_sequence)
        max_indices = (left_length, left_length, 0, 0)

        # Try different alignments
        for i in range(1, left_length + right_length + 1):
            # Add epsilon to favor longer matches
            eps = float(i) / 10000.0

            left_start = max(0, left_length - i)
            left_stop = min(left_length, left_length + right_length - i)
            left = left_sequence[left_start:left_stop]

            right_start = max(0, i - left_length)
            right_stop = min(right_length, i)
            right = right_sequence[right_start:right_stop]

            if len(left) != len(right):
                raise RuntimeError("Mismatched subsequences detected during transcript merging.")

            matches = sum(a == b for a, b in zip(left, right))

            # Normalize matches by position and add epsilon
            matching = matches / float(i) + eps

            # Require at least 2 matches
            if matches > 1 and matching > max_matching:
                max_matching = matching
                max_indices = (left_start, left_stop, right_start, right_stop)

        # Use the best alignment found
        left_start, left_stop, right_start, right_stop = max_indices

        # Take left half from left sequence and right half from right sequence
        left_mid = (left_stop + left_start) // 2
        right_mid = (right_stop + right_start) // 2

        total_sequence.extend(left_sequence[:left_mid])
        left_sequence = right_sequence[right_mid:]
        left_length = len(left_sequence)

    # Add remaining sequence
    total_sequence.extend(left_sequence)

    # Join back into text
    if match_by_words:
        return ''.join(total_sequence)
    return ''.join(total_sequence)


class ASRProcessor:
    """Process audio files to extract text transcripts using local insanely-fast-whisper, Azure Speech, OpenAI Whisper, or Groq."""

    def __init__(
        self,
        provider: str = "local",  # "local", "azure", "openai", or "groq"
        azure_speech_key: Optional[str] = None,
        azure_speech_region: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        openai_base_url: Optional[str] = None,
        groq_api_key: Optional[str] = None,
        # Local Whisper settings
        conda_env: str = "whisper_env",
        device_id: str = "0",
        model_name: str = "openai/whisper-large-v3",
        batch_size: int = 24,
        hf_token: Optional[str] = None
    ):
        self.provider = provider.lower()

        if self.provider == "local":
            self.conda_env = conda_env
            self.device_id = device_id
            self.model_name = model_name
            self.batch_size = batch_size
            self.hf_token = hf_token or os.getenv("HF_TOKEN", "")

        elif self.provider == "azure":
            if not AZURE_AVAILABLE:
                raise RuntimeError("Azure Speech SDK not available. Install with: pip install azure-cognitiveservices-speech")

            self.azure_speech_key = azure_speech_key or os.getenv("AZURE_SPEECH_KEY")
            self.azure_speech_region = azure_speech_region or os.getenv("AZURE_SPEECH_REGION")

            if not self.azure_speech_key or not self.azure_speech_region:
                raise RuntimeError("Azure Speech requires AZURE_SPEECH_KEY and AZURE_SPEECH_REGION")

            self.speech_config = speechsdk.SpeechConfig(
                subscription=self.azure_speech_key,
                region=self.azure_speech_region
            )
            self.speech_config.speech_recognition_language = "en-US"

        elif self.provider == "openai":
            if not OPENAI_AVAILABLE:
                raise RuntimeError("OpenAI not available. Install with: pip install openai")

            self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
            self.openai_base_url = openai_base_url or os.getenv("OPENAI_BASE_URL")

            if not self.openai_api_key:
                raise RuntimeError("OpenAI Whisper requires OPENAI_API_KEY")

            self.openai_client = openai.OpenAI(
                api_key=self.openai_api_key,
                base_url=self.openai_base_url
            )

        elif self.provider == "groq":
            if not GROQ_AVAILABLE:
                raise RuntimeError("Groq not available. Install with: pip install groq")

            self.groq_api_key = groq_api_key or os.getenv("GROQ_API_KEY")

            if not self.groq_api_key:
                raise RuntimeError("Groq Whisper requires GROQ_API_KEY")

            self.groq_client = Groq(
                api_key=self.groq_api_key
            )
        else:
            raise ValueError(f"Unsupported ASR provider: {provider}. Use 'local', 'azure', 'openai', or 'groq'")

    def transcribe_audio(self, audio_path: str) -> Dict[str, Any]:
        """Transcribe audio file and return transcript with metadata."""
        audio_path = Path(audio_path)

        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        if self.provider == "local":
            return self._transcribe_local(audio_path)
        elif self.provider == "azure":
            return self._transcribe_azure(audio_path)
        elif self.provider == "openai":
            return self._transcribe_openai(audio_path)
        elif self.provider == "groq":
            return self._transcribe_groq(audio_path)

    def _transcribe_local(self, audio_path: Path) -> Dict[str, Any]:
        """Transcribe using local insanely-fast-whisper."""
        try:
            # Create temporary output file
            with tempfile.NamedTemporaryFile(mode='w+', suffix='.json', delete=False) as temp_file:
                temp_output_path = temp_file.name

            # Build bash script to properly initialize conda and run command
            # When using CUDA_VISIBLE_DEVICES, always use device-id "0" since we're masking other GPUs
            bash_script = f"""
            set -e
            source /opt/conda/etc/profile.d/conda.sh
            conda activate {self.conda_env}
            export CUDA_VISIBLE_DEVICES={self.device_id}
            export HF_HUB_OFFLINE=0
            export HUGGING_FACE_HUB_TOKEN={self.hf_token}
            insanely-fast-whisper \\
                --file-name "{audio_path}" \\
                --transcript-path "{temp_output_path}" \\
                --model-name "{self.model_name}" \\
                --device-id "0" \\
                --batch-size {self.batch_size} \\
                --timestamp word \\
                --task transcribe
            """

            # Run the bash script
            result = subprocess.run(
                ["bash", "-c", bash_script],
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout
            )

            if result.returncode != 0:
                return {
                    "text": "",
                    "confidence": None,
                    "provider": "local",
                    "success": False,
                    "error": f"Command failed: {result.stderr}",
                    "words": []
                }

            # Read output file
            with open(temp_output_path, 'r') as f:
                whisper_output = json.load(f)

            # Clean up temp file
            os.unlink(temp_output_path)

            # Extract text and word timing
            full_text = whisper_output.get("text", "").strip()
            chunks = whisper_output.get("chunks", [])

            # Convert chunks to word timing format
            words = []
            for chunk in chunks:
                word_text = chunk.get("text", "").strip()
                timestamp = chunk.get("timestamp", [0.0, 0.0])
                if len(timestamp) >= 2 and word_text:
                    words.append({
                        "word": word_text,
                        "offset": timestamp[0],
                        "duration": timestamp[1] - timestamp[0],
                        "end": timestamp[1],
                        "confidence": 1.0  # Whisper doesn't provide confidence
                    })

            return {
                "text": full_text,
                "confidence": None,  # Whisper doesn't provide overall confidence
                "provider": "local",
                "success": True,
                "error": None,
                "words": words
            }

        except subprocess.TimeoutExpired:
            return {
                "text": "",
                "confidence": None,
                "provider": "local",
                "success": False,
                "error": "Transcription timed out (10 minutes)",
                "words": []
            }
        except Exception as e:
            return {
                "text": "",
                "confidence": None,
                "provider": "local",
                "success": False,
                "error": str(e),
                "words": []
            }

    def _transcribe_azure(self, audio_path: Path) -> Dict[str, Any]:
        """Transcribe using Azure Speech Services with continuous recognition for long audio."""
        audio_config = speechsdk.audio.AudioConfig(filename=str(audio_path))

        # Enable word-level timing information
        self.speech_config.request_word_level_timestamps()

        speech_recognizer = speechsdk.SpeechRecognizer(
            speech_config=self.speech_config,
            audio_config=audio_config
        )

        # Use continuous recognition for long audio files
        all_results = []
        all_words = []  # Store word-level timing information
        done = False
        error_msg = None

        def handle_recognized(evt):
            if evt.result.text:
                all_results.append(evt.result.text)

                # Extract word-level timing if available
                if hasattr(evt.result, 'json') and evt.result.json:
                    try:
                        import json
                        result_json = json.loads(evt.result.json)

                        # Extract word timing from NBest results
                        if 'NBest' in result_json and result_json['NBest']:
                            best_result = result_json['NBest'][0]
                            if 'Words' in best_result:
                                for word_info in best_result['Words']:
                                    word_data = {
                                        'word': word_info.get('Word', ''),
                                        'offset': word_info.get('Offset', 0) / 10000000,  # Convert to seconds
                                        'duration': word_info.get('Duration', 0) / 10000000,  # Convert to seconds
                                        'confidence': word_info.get('Confidence', 0)
                                    }
                                    # Calculate end time
                                    word_data['end'] = word_data['offset'] + word_data['duration']
                                    all_words.append(word_data)
                    except Exception as e:
                        # If parsing fails, continue without word timing
                        pass

        def handle_session_stopped(evt):
            nonlocal done
            done = True

        def handle_canceled(evt):
            nonlocal done, error_msg
            if evt.result.reason == speechsdk.ResultReason.Canceled:
                cancellation = evt.result.cancellation_details
                error_msg = f"Speech recognition canceled: {cancellation.reason}"
                if cancellation.error_details:
                    error_msg += f" - {cancellation.error_details}"
            done = True

        # Connect callbacks
        speech_recognizer.recognized.connect(handle_recognized)
        speech_recognizer.session_stopped.connect(handle_session_stopped)
        speech_recognizer.canceled.connect(handle_canceled)

        # Start continuous recognition
        speech_recognizer.start_continuous_recognition()

        # Wait for recognition to complete
        import time
        timeout = 300  # 5 minutes timeout for long audio
        start_time = time.time()

        while not done and (time.time() - start_time) < timeout:
            time.sleep(0.1)

        # Stop recognition
        speech_recognizer.stop_continuous_recognition()

        # Return results
        if error_msg:
            return {
                "text": "",
                "confidence": None,
                "provider": "azure",
                "success": False,
                "error": error_msg,
                "words": []
            }
        elif all_results:
            # Join all recognized text segments
            full_text = " ".join(all_results).strip()
            return {
                "text": full_text,
                "confidence": None,  # Overall confidence not available in continuous recognition
                "provider": "azure",
                "success": True,
                "error": None,
                "words": all_words  # Include word-level timing
            }
        else:
            return {
                "text": "",
                "confidence": None,
                "provider": "azure",
                "success": False,
                "error": "No speech recognized",
                "words": []
            }
    
    def _transcribe_openai(self, audio_path: Path) -> Dict[str, Any]:
        """Transcribe using OpenAI Whisper API."""
        try:
            with open(audio_path, 'rb') as audio_file:
                transcript = self.openai_client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    language="en"
                )

            return {
                "text": transcript.text.strip(),
                "confidence": None,  # Whisper API doesn't return confidence
                "provider": "openai",
                "success": True,
                "error": None,
                "words": []
            }
        except Exception as e:
            return {
                "text": "",
                "confidence": None,
                "provider": "openai",
                "success": False,
                "error": str(e),
                "words": []
            }

    def _transcribe_groq(self, audio_path: Path) -> Dict[str, Any]:
        """Transcribe using Groq Whisper API with automatic chunking for large files."""
        try:
            # Check file size to determine if we need chunking
            file_size_mb = audio_path.stat().st_size / (1024 * 1024)

            # Use chunked approach for files > 20MB (conservative threshold)
            if file_size_mb > 20:
                print(f"  Large file detected ({file_size_mb:.1f}MB), using chunked transcription...")
                return self._transcribe_groq_chunked(audio_path)
            else:
                return self._transcribe_groq_simple(audio_path)
        except Exception as e:
            return {
                "text": "",
                "confidence": None,
                "provider": "groq",
                "success": False,
                "error": str(e),
                "words": []
            }

    def _transcribe_groq_simple(self, audio_path: Path) -> Dict[str, Any]:
        """Simple Groq transcription for smaller files."""
        try:
            with open(audio_path, 'rb') as audio_file:
                transcription = self.groq_client.audio.transcriptions.create(
                    file=audio_file,
                    model="whisper-large-v3",
                    response_format="verbose_json",
                    timestamp_granularities=["word", "segment"],
                    language="en",
                    temperature=0.0
                )

            # Extract word-level timing if available
            words = []
            if hasattr(transcription, 'words') and transcription.words:
                for word_info in transcription.words:
                    # Handle both object and dict formats
                    if hasattr(word_info, 'word'):
                        # Object format
                        words.append({
                            'word': word_info.word,
                            'offset': word_info.start,
                            'duration': word_info.end - word_info.start,
                            'end': word_info.end,
                            'confidence': 1.0  # Groq doesn't provide word-level confidence
                        })
                    elif isinstance(word_info, dict):
                        # Dict format
                        words.append({
                            'word': word_info.get('word', ''),
                            'offset': word_info.get('start', 0),
                            'duration': word_info.get('end', 0) - word_info.get('start', 0),
                            'end': word_info.get('end', 0),
                            'confidence': 1.0  # Groq doesn't provide word-level confidence
                        })

            return {
                "text": transcription.text.strip(),
                "confidence": None,  # Groq doesn't return overall confidence
                "provider": "groq",
                "success": True,
                "error": None,
                "words": words
            }
        except Exception as e:
            return {
                "text": "",
                "confidence": None,
                "provider": "groq",
                "success": False,
                "error": str(e),
                "words": []
            }

    def _transcribe_groq_chunked(self, audio_path: Path, chunk_length: int = 600, overlap: int = 10) -> Dict[str, Any]:
        """Transcribe large audio files using chunked approach for Groq."""
        processed_path = None
        try:
            # Preprocess audio
            processed_path = _preprocess_audio_for_groq(audio_path)
            audio = AudioSegment.from_file(processed_path, format="flac")

            duration = len(audio)
            print(f"  Audio duration: {duration/1000:.2f}s")

            # Calculate number of chunks
            chunk_ms = chunk_length * 1000
            overlap_ms = overlap * 1000
            total_chunks = (duration // (chunk_ms - overlap_ms)) + 1
            print(f"  Processing {total_chunks} chunks...")

            results = []

            # Process each chunk
            for i in range(total_chunks):
                start = i * (chunk_ms - overlap_ms)
                end = min(start + chunk_ms, duration)

                print(f"  Processing chunk {i+1}/{total_chunks} ({start/1000:.1f}s - {end/1000:.1f}s)")

                chunk = audio[start:end]
                result, _ = self._transcribe_single_groq_chunk(chunk, i+1, total_chunks)
                results.append((result, start))

            # Merge transcripts
            final_result = self._merge_groq_transcripts(results)
            return {
                "text": final_result["text"],
                "confidence": None,
                "provider": "groq",
                "success": True,
                "error": None,
                "words": final_result.get("words", [])
            }

        except Exception as e:
            return {
                "text": "",
                "confidence": None,
                "provider": "groq",
                "success": False,
                "error": f"Chunked transcription failed: {str(e)}",
                "words": []
            }
        finally:
            # Clean up temp files
            if processed_path:
                processed_path.unlink(missing_ok=True)

    def _transcribe_single_groq_chunk(self, chunk: AudioSegment, chunk_num: int, total_chunks: int) -> Tuple[Dict[str, Any], float]:
        """Transcribe a single audio chunk with Groq API."""
        while True:
            with tempfile.NamedTemporaryFile(suffix='.flac') as temp_file:
                chunk.export(temp_file.name, format='flac')

                start_time = time.time()
                try:
                    result = self.groq_client.audio.transcriptions.create(
                        file=("chunk.flac", temp_file, "audio/flac"),
                        model="whisper-large-v3",
                        language="en",
                        response_format="verbose_json"
                    )
                    api_time = time.time() - start_time

                    print(f"    Chunk {chunk_num}/{total_chunks} processed in {api_time:.2f}s")
                    return result, api_time

                except RateLimitError as e:
                    print(f"    Rate limit hit for chunk {chunk_num} - retrying in 60 seconds...")
                    time.sleep(60)
                    continue

                except Exception as e:
                    print(f"    Error transcribing chunk {chunk_num}: {str(e)}")
                    raise

    def _merge_groq_transcripts(self, results: List[Tuple[Dict[str, Any], int]]) -> Dict[str, Any]:
        """Merge transcription chunks and handle overlaps."""
        print("  Merging transcription results...")

        # Process word-level timestamps
        words = []

        for chunk, chunk_start_ms in results:
            # Convert Pydantic model to dict if needed
            data = chunk.model_dump() if hasattr(chunk, 'model_dump') else chunk

            # Process word timestamps if available
            if isinstance(data, dict) and 'words' in data and data['words'] is not None and len(data['words']) > 0:
                chunk_words = data['words']
                for word in chunk_words:
                    # Adjust word timestamps based on chunk start time
                    word['start'] = word['start'] + (chunk_start_ms / 1000)
                    word['end'] = word['end'] + (chunk_start_ms / 1000)
                words.extend(chunk_words)
            elif hasattr(chunk, 'words') and getattr(chunk, 'words') is not None:
                chunk_words = getattr(chunk, 'words')
                processed_words = []
                for word in chunk_words:
                    if hasattr(word, 'model_dump'):
                        word_dict = word.model_dump()
                    else:
                        word_dict = {
                            'word': getattr(word, 'word', ''),
                            'start': getattr(word, 'start', 0) + (chunk_start_ms / 1000),
                            'end': getattr(word, 'end', 0) + (chunk_start_ms / 1000)
                        }
                    processed_words.append(word_dict)
                words.extend(processed_words)

        # Simple text merging for now
        texts = []
        for chunk, _ in results:
            data = chunk.model_dump() if hasattr(chunk, 'model_dump') else chunk

            if isinstance(data, dict):
                text = data.get('text', '')
            else:
                text = getattr(chunk, 'text', '')

            texts.append(text)

        merged_text = " ".join(texts)

        # Convert words to expected format
        formatted_words = []
        for word in words:
            formatted_words.append({
                'word': word.get('word', ''),
                'offset': word.get('start', 0),
                'duration': word.get('end', 0) - word.get('start', 0),
                'end': word.get('end', 0),
                'confidence': 1.0
            })

        return {
            "text": merged_text,
            "words": formatted_words
        }

    async def transcribe_async(self, audio_path: str) -> Dict[str, Any]:
        """Async wrapper for transcription."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.transcribe_audio, audio_path)

    def batch_transcribe(self, audio_paths: list[str]) -> Dict[str, Dict[str, Any]]:
        """Transcribe multiple audio files."""
        results = {}
        for audio_path in audio_paths:
            try:
                results[audio_path] = self.transcribe_audio(audio_path)
            except Exception as e:
                results[audio_path] = {
                    "text": "",
                    "confidence": None,
                    "provider": self.provider,
                    "success": False,
                    "error": str(e)
                }
        return results

    async def batch_transcribe_async(self, audio_paths: list[str]) -> Dict[str, Dict[str, Any]]:
        """Async batch transcription."""
        tasks = [self.transcribe_async(path) for path in audio_paths]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        output = {}
        for path, result in zip(audio_paths, results):
            if isinstance(result, Exception):
                output[path] = {
                    "text": "",
                    "confidence": None,
                    "provider": self.provider,
                    "success": False,
                    "error": str(result)
                }
            else:
                output[path] = result
                
        return output