import os
import azure.cognitiveservices.speech as speechsdk
from typing import Optional, Tuple
from pathlib import Path


class AzureSTTService:
    """Azure Speech-to-Text service for audio file transcription."""

    def __init__(self,
                 speech_key: Optional[str] = None,
                 speech_region: Optional[str] = None):
        """
        Initialize Azure STT service.

        Args:
            speech_key: Azure Speech API key (defaults to env var)
            speech_region: Azure region (defaults to env var)
        """
        self.speech_key = speech_key or os.environ.get("AZURE_SPEECH_API_KEY")
        self.speech_region = speech_region or os.environ.get("AZURE_SPEECH_REGION")

        if not self.speech_key or not self.speech_region:
            raise ValueError(
                "Azure Speech credentials not found. "
                "Set AZURE_SPEECH_API_KEY and AZURE_SPEECH_REGION environment variables."
            )

        self.speech_config = speechsdk.SpeechConfig(
            subscription=self.speech_key,
            region=self.speech_region
        )

        # Set recognition language (can be made configurable)
        self.speech_config.speech_recognition_language = "en-US"

        # Enable detailed recognition results
        self.speech_config.request_word_level_timestamps()

    def transcribe_file(self, audio_file_path: str) -> Tuple[str, dict]:
        """
        Transcribe audio file to text.

        Args:
            audio_file_path: Path to audio file (WAV, MP3, etc.)

        Returns:
            Tuple of (transcript, metadata dict with timing info)
        """
        audio_path = Path(audio_file_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_file_path}")

        # Create audio config from file
        audio_config = speechsdk.audio.AudioConfig(filename=str(audio_path))

        # Create recognizer
        recognizer = speechsdk.SpeechRecognizer(
            speech_config=self.speech_config,
            audio_config=audio_config
        )

        # Collect all results
        all_results = []
        done = False

        def handle_recognized(evt):
            if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
                all_results.append({
                    'text': evt.result.text,
                    'offset': evt.result.offset,
                    'duration': evt.result.duration
                })

        def stop_continuous(evt):
            nonlocal done
            done = True

        # Connect callbacks
        recognizer.recognized.connect(handle_recognized)
        recognizer.session_stopped.connect(stop_continuous)
        recognizer.canceled.connect(stop_continuous)

        # Start continuous recognition
        recognizer.start_continuous_recognition()

        # Wait for completion
        import time
        while not done:
            time.sleep(0.5)

        recognizer.stop_continuous_recognition()

        # Combine results
        full_transcript = ' '.join(r['text'] for r in all_results)

        metadata = {
            'segments': all_results,
            'total_segments': len(all_results),
            'file_path': str(audio_path),
            'language': self.speech_config.speech_recognition_language
        }

        return full_transcript.strip(), metadata

    def transcribe_with_diarization(self, audio_file_path: str) -> Tuple[str, dict]:
        """
        Transcribe audio with speaker diarization (who said what).

        Args:
            audio_file_path: Path to audio file

        Returns:
            Tuple of (transcript with speaker labels, metadata)
        """
        audio_path = Path(audio_file_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_file_path}")

        # Create audio config
        audio_config = speechsdk.audio.AudioConfig(filename=str(audio_path))

        # Enable diarization
        self.speech_config.set_property(
            speechsdk.PropertyId.SpeechServiceConnection_LanguageIdMode, "Continuous"
        )

        # Create conversation transcriber
        conversation_transcriber = speechsdk.transcription.ConversationTranscriber(
            speech_config=self.speech_config,
            audio_config=audio_config
        )

        transcription_results = []
        done = False

        def handle_transcribed(evt):
            if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
                transcription_results.append({
                    'speaker_id': evt.result.speaker_id or 'Unknown',
                    'text': evt.result.text,
                    'offset': evt.result.offset,
                    'duration': evt.result.duration
                })

        def stop_cb(evt):
            nonlocal done
            done = True

        # Connect callbacks
        conversation_transcriber.transcribed.connect(handle_transcribed)
        conversation_transcriber.session_stopped.connect(stop_cb)
        conversation_transcriber.canceled.connect(stop_cb)

        # Start transcription
        conversation_transcriber.start_transcribing_async()

        # Wait for completion
        import time
        while not done:
            time.sleep(0.5)

        conversation_transcriber.stop_transcribing_async()

        # Format output with speaker labels
        formatted_transcript = []
        current_speaker = None

        for segment in transcription_results:
            speaker = segment['speaker_id']
            if speaker != current_speaker:
                formatted_transcript.append(f"\n[Speaker {speaker}]: {segment['text']}")
                current_speaker = speaker
            else:
                formatted_transcript.append(segment['text'])

        full_transcript = ' '.join(formatted_transcript).strip()

        metadata = {
            'segments': transcription_results,
            'total_segments': len(transcription_results),
            'speakers': list(set(s['speaker_id'] for s in transcription_results)),
            'file_path': str(audio_path),
            'language': self.speech_config.speech_recognition_language
        }

        return full_transcript, metadata