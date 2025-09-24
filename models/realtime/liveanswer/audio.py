import re
import time
import threading

import azure.cognitiveservices.speech as speechsdk  # type: ignore

from .utils import _env
from .explain import ExplainSynthesizer


class AzureSpeechClient:
    def __init__(self) -> None:
        self.key = _env("AZURE_SPEECH_KEY")
        self.region = _env("AZURE_SPEECH_REGION")
        self.voice = _env("AZURE_SPEECH_VOICE", "en-US-JennyNeural")
        self.output_format_name = _env("AZURE_SPEECH_FORMAT", "Audio24Khz160KBitRateMonoMp3")
        self._sdk_ready = bool(self.key and self.region and speechsdk is not None)

        if self._sdk_ready:
            self.speech_config = speechsdk.SpeechConfig(subscription=self.key, region=self.region)
            self.speech_config.speech_synthesis_voice_name = self.voice
            fmt = getattr(speechsdk.SpeechSynthesisOutputFormat, self.output_format_name)
            self.speech_config.set_speech_synthesis_output_format(fmt)
        else:
            self.speech_config = None


class AudioGenerator:
    def __init__(self, explainer: ExplainSynthesizer):
        self.explainer = explainer
        self.azure = AzureSpeechClient()
        self.all_sound = bytearray()
        self._stop_event = threading.Event()
        self.request_start_time: float = time.monotonic()
        self.start_time: float = 0.0
        self._stream_req = None
        self._generated_seconds = 0.0
        self._bitrate_bps = self._guess_bitrate(self.azure.output_format_name)

    @staticmethod
    def _guess_bitrate(format_name: str) -> float:
        if not format_name:
            return 160_000.0
        match = re.search(r"(\d+)KBitRate", format_name)
        if match:
            return float(match.group(1)) * 1000.0
        sr_match = re.search(r"Audio(\d+)Khz", format_name)
        if sr_match:
            sample_rate = float(sr_match.group(1)) * 1000.0
            bit_depth = 16.0
            if "8Bit" in format_name:
                bit_depth = 8.0
            elif "24Bit" in format_name:
                bit_depth = 24.0
            return sample_rate * bit_depth
        return 160_000.0

    def _update_generated_seconds(self, byte_count: int) -> None:
        if byte_count <= 0:
            return
        bitrate = self._bitrate_bps or 160_000.0
        self._generated_seconds += (byte_count * 8.0) / bitrate

    def _watch_need_more_explanation(self) -> None:
        if not self.explainer.spoken_explanation:
            # first = self.explainer.pop_more_explanation(max_token=80)
            # first = self.explainer.pop_more_explanation(max_token=64)
            # first = self.explainer.pop_more_explanation(max_token=32)
            first = self.explainer.pop_more_explanation()
            print(f"!!!AudioGen: First chunk from explainer: '{first[:100]}...' ({len(first) if first else 0} chars)")
            if first and self._stream_req is not None:
                print(f"!!!AudioGen: Writing first chunk to TTS stream")
                self._stream_req.input_stream.write(first)

        # time_margin = 10.0
        time_margin = 10.0
        while not self._stop_event.is_set():
            if self.start_time == 0.0:
                elapsed = 0.0
            else:
                elapsed = time.monotonic() - self.start_time
            total_estimated = self._generated_seconds
            remaining = total_estimated - elapsed
            print(f"!!!total_estimated: {total_estimated}, elapsed: {elapsed}, remaining: {remaining}")

            if remaining <= time_margin:
                more = self.explainer.pop_more_explanation()
                print(f"!!!AudioGen: Got more chunk: '{more[:100] if more else None}...' ({len(more) if more else 0} chars)")
                if more is not None:
                    if more and self._stream_req is not None:
                        print(f"!!!AudioGen: Writing more chunk to TTS stream")
                        self._stream_req.input_stream.write(more)
                else:
                    print(f"!!!AudioGen: No more chunks, closing TTS stream")
                    if self._stream_req is not None:
                        self._stream_req.input_stream.close()
                    return

            time.sleep(max(0.5, remaining - time_margin))

    def start(self) -> tuple[bytes, float]:
        self.all_sound.clear()
        self._generated_seconds = 0.0
        self._stop_event.clear()
        self.start_time = 0.0  # Reset start time

        if not getattr(self.azure, "_sdk_ready", False):
            raise RuntimeError("Azure Speech SDK or credentials not available for streaming synthesis")

        try:
            region = self.azure.region
            key = self.azure.key
            voice = self.azure.voice

            tts_endpoint = f"wss://{region}.tts.speech.microsoft.com/cognitiveservices/websocket/v2"
            cfg = speechsdk.SpeechConfig(endpoint=tts_endpoint, subscription=key)
            cfg.speech_synthesis_voice_name = voice
            fmt = getattr(speechsdk.SpeechSynthesisOutputFormat, self.azure.output_format_name)
            cfg.set_speech_synthesis_output_format(fmt)
            cfg.set_property(speechsdk.PropertyId.SpeechSynthesis_RtfTimeoutThreshold, "4")
            cfg.set_property(speechsdk.PropertyId.SpeechSynthesis_FrameTimeoutInterval, str(int(60*1000))) # 60s

            req = speechsdk.SpeechSynthesisRequest(speechsdk.SpeechSynthesisRequestInputType.TextStream)
            self._stream_req = req
            synth = speechsdk.SpeechSynthesizer(speech_config=cfg, audio_config=None)

            def on_synthesizing(evt):
                if self.start_time == 0.0:
                    self.start_time = time.monotonic()
                data_bytes = evt.result.audio_data
                if data_bytes:
                    self.all_sound.extend(data_bytes)
                    self._update_generated_seconds(len(data_bytes))
            
            def on_synthesis_started(evt):
                print(f"!!!TTS: synthesis started")
            def on_synthesis_completed(evt):
                print(f"!!!TTS: synthesis completed")
            def on_synthesis_canceled(evt):
                print(f"!!!TTS: synthesis canceled - {evt}")
            def on_synthesis_error(evt):
                print(f"!!!TTS: synthesis error - {evt}")

            synth.synthesizing.connect(on_synthesizing)
            synth.synthesis_started.connect(on_synthesis_started)
            synth.synthesis_completed.connect(on_synthesis_completed)
            synth.synthesis_canceled.connect(on_synthesis_canceled)
            # Note: synthesis_error might not exist in all SDK versions

            fut = synth.speak_async(req)

            t_watcher = threading.Thread(target=self._watch_need_more_explanation, name="watcher", daemon=True)
            t_watcher.start()

            r = fut.get()
            if r.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                print(f"!!!synthesis completed")
            elif r.reason == speechsdk.ResultReason.Canceled:
                print(f"!!!synthesis canceled: {r.cancellation_details.reason}, {r.cancellation_details.error_details}")
            else:
                print(f"!!!synthesis failed: {r.reason}")
            t_watcher.join()
        finally:
            self._stop_event.set()
            self._stream_req = None

        # Calculate time to first response, with fallback if TTS never started
        if self.start_time > 0.0:
            time_to_first_response = self.start_time - self.request_start_time
        else:
            # TTS never started, use current time as fallback
            time_to_first_response = time.monotonic() - self.request_start_time
            print(f"!!!AudioGen: TTS never started, using fallback timing: {time_to_first_response:.2f}s")

        return bytes(self.all_sound), time_to_first_response
