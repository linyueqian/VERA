import os
import re
import json
import base64
import argparse
import time
import math
import asyncio
import io
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import soundfile as sf
import librosa
from dotenv import load_dotenv

try:
    # Google Gemini Realtime SDK
    from google import genai
    from google.genai import types
except Exception as e:  # pragma: no cover
    genai = None
    types = None


def require_env(name: str) -> str:
    val = os.getenv(name)
    if not val:
        raise RuntimeError(f"Missing env: {name}")
    return val


def build_config(model_name: str, system_instruction: str, is_browse: bool = False) -> types.LiveConnectConfig:
    """Build config per model.
    - Thinking model: minimal AUDIO + voice + disabled AAD (matches examples).
    - Others: add AAD disable + transcription.
    """
    config_dict = {
        "system_instruction": system_instruction,
        "response_modalities": ['AUDIO'],
        "realtime_input_config": {'automatic_activity_detection': {'disabled': True}},
        "speech_config": types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name='Puck')
            )
        ),
    }

    # Add transcription for non-thinking models
    if 'thinking' not in (model_name or '').lower():
        config_dict["output_audio_transcription"] = {}

    # Add search tool for BrowseComp
    if is_browse:
        config_dict["tools"] = [{'google_search': {}}]

    return types.LiveConnectConfig(**config_dict)


def load_first_user_audio(episode_json: Path) -> Path:
    data = json.loads(episode_json.read_text())
    eps = data.get("episodes", [])
    if not eps:
        raise RuntimeError("No episodes found in JSON")
    turns = eps[0].get("turns", [])
    ep_id = eps[0].get("id")
    for t in turns:
        if t.get("role") == "user" and t.get("audio_file"):
            p = Path(t["audio_file"])  # absolute path supported
            if p.exists():
                return p
            # Fallback: try repo-local test_voice_episodes/audio/{id}.wav
            if ep_id:
                cand = (Path.cwd() / "test_voice_episodes/audio/" / f"{ep_id}.wav").resolve()
                if cand.exists():
                    return cand
            # Fallback: try same basename under test_voice_episodes/audio
            cand2 = (Path.cwd() / "test_voice_episodes/audio/" / Path(t["audio_file"]).name).resolve()
            if cand2.exists():
                return cand2
            return p  # will fail later with clear error
    raise RuntimeError("No user turn with audio_file found")


def to_mono_pcm16(path: Path, target_sr: int = 16_000) -> Tuple[np.ndarray, int]:
    wav, sr = sf.read(str(path), dtype="int16")
    if wav.ndim == 2:
        wav = wav.mean(axis=1).astype(np.int16)
    if sr == target_sr:
        return wav, sr
    try:
        import scipy.signal as ss

        g = np.gcd(sr, target_sr)
        up = target_sr // g
        down = sr // g
        wav_f = wav.astype(np.float32) / 32768.0
        y = ss.resample_poly(wav_f, up, down)
        return (np.clip(y, -1.0, 1.0) * 32767).astype(np.int16), target_sr
    except Exception:
        ratio = target_sr / sr
        idx = (np.arange(int(len(wav) * ratio)) / ratio).astype(np.int64)
        y = wav[np.minimum(idx, len(wav) - 1)]
        return y.astype(np.int16), target_sr


def parse_context_documents_for_system_instruction(episode_data: dict) -> str:
    """Build a system instruction string using context_documents if present.

    For MRCR-like tasks, we include the whole conversation as prior history.
    Otherwise, we concatenate document content as helpful background.
    """
    if not episode_data:
        return "You are a helpful assistant."

    eps = episode_data.get("episodes", [])
    if not eps:
        return "You are a helpful assistant."

    first = eps[0]
    docs = first.get("context_documents", []) or []
    episode_id = (first.get("id") or "").lower()
    track = first.get("track") or ""

    is_mrcr = ("mrcr" in episode_id) or (track == "long_context")

    if not docs:
        return "You are a helpful assistant. Be concise."

    if is_mrcr:
        parts = ["You are a helpful assistant. Prior conversation history follows. Use it for context.\n\n"]
        for i, d in enumerate(docs, 1):
            content = d.get("content") or ""
            if content:
                parts.append(f"[Document {i}]\n{content}\n\n")
        return "".join(parts)
    else:
        parts = ["You are a helpful assistant. The following background may be useful.\n\n"]
        for i, d in enumerate(docs, 1):
            content = d.get("content") or ""
            if content:
                parts.append(f"[Context {i}]\n{content}\n\n")
        return "".join(parts)


def iter_frames_pcm16(pcm16: np.ndarray, sr: int, frame_ms: int = 30):
    frame_samples = sr * frame_ms // 1000
    for i in range(0, len(pcm16), frame_samples):
        chunk = pcm16[i : i + frame_samples]
        if chunk.size > 0:
            yield chunk.tobytes()


def frame_iter(data: np.ndarray, frame_ms: int = 30, sr: int = 16_000) -> List[bytes]:
    """Convert audio data to frames with proper padding"""
    if data.ndim == 2:
        data = data.mean(axis=1).astype(np.int16)

    frame_samples = sr * frame_ms // 1000
    pad = (-len(data)) % frame_samples
    if pad:
        data = np.pad(data, (0, pad))

    total_frames = len(data) // frame_samples
    frames: List[bytes] = []
    for i in range(total_frames):
        chunk = data[i * frame_samples : (i + 1) * frame_samples]
        frames.append(chunk.tobytes())

    return frames


def parse_rate_from_mime(mime: str, default: int = 24_000) -> int:
    if not mime:
        return default
    m = re.search(r"rate=(\d+)", mime)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return default
    return default


def _is_browsecomp(episode_data: Optional[dict]) -> bool:
    try:
        if not episode_data:
            return False
        ep = (episode_data.get("episodes") or [{}])[0]
        eid = (ep.get("id") or "").lower()
        track = (ep.get("track") or "").lower()
        return ("browsecomp" in eid) or (track == "browsecomp")
    except Exception:
        return False


async def end_audio_turn(sess):
    """End audio turn with fallback methods"""
    try:
        await sess.send_realtime_input(audio_stream_end=True)
    except TypeError:
        try:
            await sess.end_realtime_input()
        except AttributeError:
            pass


async def run_gemini_audio_roundtrip(args, episode_data: Optional[dict]):
    if genai is None:
        raise RuntimeError("google-genai SDK not available. Please install google-genai.")

    # Resolve input audio and symlink/copy for convenience
    input_wav_path = Path(args.out_dir) / "input.wav"
    if args.mode == "audio":
        user_audio = load_first_user_audio(Path(args.episode_json))
        if input_wav_path.exists() or input_wav_path.is_symlink():
            try:
                input_wav_path.unlink()
            except FileNotFoundError:
                pass
        try:
            os.symlink(user_audio, input_wav_path)
        except OSError:
            sf_data, sr = sf.read(str(user_audio))
            sf.write(str(input_wav_path), sf_data, sr)

    # Prepare audio efficiently - avoid unnecessary conversions when possible
    print("Loading and optimizing audio format...")

    try:
        # Use the existing optimized conversion function first (much faster for compatible formats)
        pcm16_data, actual_sr = to_mono_pcm16(input_wav_path, target_sr=16000)
        audio_bytes = pcm16_data.tobytes()
        duration = len(pcm16_data) / actual_sr
        print(f"Audio prepared efficiently: {len(audio_bytes)} bytes at {actual_sr}Hz, duration: {duration:.2f}s")

    except Exception as e:
        print(f"Falling back to librosa conversion: {e}")
        # Fallback to librosa if the optimized method fails
        y, sr = librosa.load(str(input_wav_path), sr=16000)

        # Convert to PCM16 format in memory buffer
        buffer = io.BytesIO()
        sf.write(buffer, y, sr, format='RAW', subtype='PCM_16')
        buffer.seek(0)
        audio_bytes = buffer.read()
        duration = len(y)/sr
        print(f"Audio prepared (librosa fallback): {len(audio_bytes)} bytes at {sr}Hz, duration: {duration:.2f}s")

    # Build Gemini Live config
    system_instruction = parse_context_documents_for_system_instruction(episode_data)
    is_browse = _is_browsecomp(episode_data)

    # Select model
    model = os.getenv("GEMINI_REALTIME_MODEL", "gemini-2.5-flash-preview-native-audio-dialog")
    if getattr(args, "model", None):
        model = args.model

    # Get user question for initial content
    user_question = ""
    try:
        turns = episode_data.get("episodes", [{}])[0].get("turns", []) if episode_data else []
        for t in turns:
            if t.get("role") == "user":
                user_question = t.get("text_content", "")
                break
    except Exception:
        pass

    api_key = require_env("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key, http_options={"api_version": "v1beta"})

    # Output buffers
    audio_buf = bytearray()
    text_parts: List[str] = []
    transcript_parts: List[str] = []
    thought_parts: List[str] = []
    inline_data_parts: List[dict] = []  # Store inline_data for debugging/analysis

    start_time = time.time()
    audio_send_complete_time = None
    first_response_time = None
    first_audio_chunk_time = None  # Track first actual audio chunk
    detected_recv_sr: Optional[int] = None

    # Build config using the improved build_config function
    config = build_config(model, system_instruction, is_browse)

    print(f"Using efficient single-blob audio input for model: {model}")

    # Run improved session with efficient audio input
    is_thinking = 'thinking' in (model or '').lower()

    async with client.aio.live.connect(model=model, config=config) as sess:
        print("Connected to Gemini Live session")

        # Start a user turn explicitly so generation only starts when we complete it.
        if not is_thinking:
            try:
                await sess.send_client_content(
                    turns={
                        "role": "user",
                        "parts": ([{"text": user_question}] if user_question else []),
                    },
                    turn_complete=False,
                )
            except Exception as e:
                print(f"Warning: unable to send initial client content: {e}")

        sending_done = asyncio.Event()
        stop_sending = asyncio.Event()

        async def sender():
            nonlocal audio_send_complete_time

            # Send entire audio as single blob - much more efficient!
            print(f"Sending audio input: {len(audio_bytes)} bytes")
            send_start = time.time()

            await sess.send_realtime_input(
                audio=types.Blob(data=audio_bytes, mime_type="audio/pcm;rate=16000")
            )

            send_time = time.time() - send_start
            print(f"Audio input sent in {send_time:.3f}s (efficient single-blob method)")

            # Signal end of audio stream and then complete the user turn.
            await end_audio_turn(sess)
            print("Audio stream ended")
            try:
                await sess.send_client_content(turns={"role": "user", "parts": []}, turn_complete=True)
                print("Turn completed")
                # Mark the time when audio sending is fully complete
                audio_send_complete_time = time.time()
            except Exception as e:
                print(f"Warning: unable to complete turn: {e}")
                # Still mark completion time even if turn completion failed
                audio_send_complete_time = time.time()
            sending_done.set()

        async def receiver():
            nonlocal first_response_time, detected_recv_sr, audio_send_complete_time, inline_data_parts, first_audio_chunk_time
            response_count = 0
            recv_via_stream = False
            fallback_inline_audio = bytearray()

            try:
                async for resp in sess.receive():
                    response_count += 1

                    if first_response_time is None and audio_send_complete_time is not None:
                        first_response_time = time.time() - audio_send_complete_time
                        print(f"  First response received {first_response_time:.3f}s after audio send completed")

                    # Check for go_away message
                    if hasattr(resp, 'go_away') and resp.go_away:
                        print(f"  [{response_count}] Go away: {resp.go_away.time_left}s left")

                    # Primary: streaming audio bytes
                    data_bytes = getattr(resp, "data", None)
                    if isinstance(data_bytes, (bytes, bytearray)) and data_bytes:
                        # Detect first audio chunk for accurate timing
                        if first_audio_chunk_time is None and audio_send_complete_time is not None:
                            first_audio_chunk_time = time.time() - audio_send_complete_time
                            print(f"  🎵 First audio chunk received {first_audio_chunk_time:.3f}s after audio send completed")

                        audio_buf.extend(data_bytes)
                        recv_via_stream = True
                        print(f"  [{response_count}] Audio: {len(data_bytes)} bytes (stream)")

                    # Check for direct text
                    if isinstance(getattr(resp, "text", None), str) and resp.text:
                        text_parts.append(resp.text)
                        print(f"  [{response_count}] Text: {resp.text[:50]}...")

                    # Check for direct thought (thinking models)
                    if hasattr(resp, 'thought') and resp.thought:
                        if isinstance(resp.thought, str):
                            thought_parts.append(resp.thought)
                            print(f"  [{response_count}] Thought: {resp.thought[:50]}...")
                        elif isinstance(resp.thought, bool):
                            print(f"  [{response_count}] Thought flag: <enabled>")

                    # Check server_content
                    sc = getattr(resp, "server_content", None)
                    if sc is not None:
                        # If server provided an output transcription for audio
                        try:
                            ot = getattr(sc, "output_transcription", None)
                            if ot is not None:
                                t = getattr(ot, "text", None)
                                if isinstance(t, str) and t:
                                    transcript_parts.append(t)
                                    print(f"  [{response_count}] Transcript: {t[:80]}...")
                        except Exception:
                            pass

                        # Check content.parts (fallback capture only if no stream bytes)
                        content = getattr(sc, "content", None)
                        parts = getattr(content, "parts", None) if content is not None else None
                        if parts:
                            for part in parts:
                                # Capture ALL inline_data for debugging/analysis
                                if getattr(part, "inline_data", None):
                                    inline_data = part.inline_data
                                    mime_type = getattr(inline_data, "mime_type", "")
                                    data_size = len(getattr(inline_data, "data", b""))

                                    # Detect first audio chunk for accurate timing
                                    if first_audio_chunk_time is None and audio_send_complete_time is not None and mime_type.startswith("audio/pcm"):
                                        first_audio_chunk_time = time.time() - audio_send_complete_time
                                        print(f"  🎵 First audio chunk received {first_audio_chunk_time:.3f}s after audio send completed")

                                    data_info = {
                                        "response_count": response_count,
                                        "mime_type": mime_type,
                                        "data_size": data_size,
                                        "location": "content_parts"
                                    }
                                    inline_data_parts.append(data_info)
                                    print(f"  [{response_count}] Inline data: {data_info['mime_type']}, {data_info['data_size']} bytes (content)")

                                if getattr(part, "inline_data", None) and getattr(part.inline_data, "mime_type", "").startswith("audio/pcm"):
                                    if not recv_via_stream and getattr(part.inline_data, "data", None):
                                        fallback_inline_audio.extend(part.inline_data.data)
                                        mime = getattr(part.inline_data, "mime_type", "")
                                        if "rate=" in mime:
                                            try:
                                                detected_recv_sr = int(mime.split("rate=")[-1].split(";")[0])
                                            except Exception:
                                                pass
                                        print(f"  [{response_count}] Inline audio: {len(part.inline_data.data)} bytes (fallback)")
                                txt = getattr(part, "text", None)
                                if isinstance(txt, str) and txt:
                                    text_parts.append(txt)
                                    print(f"  [{response_count}] Content text: {txt[:50]}...")

                                # Handle thoughts in content parts
                                th = getattr(part, 'thought', None)
                                if th:
                                    if isinstance(th, str):
                                        thought_parts.append(th)
                                        print(f"  [{response_count}] Content thought: {th[:50]}...")
                                    elif isinstance(th, bool):
                                        print(f"  [{response_count}] Content thought flag: <enabled>")

                        # Check model_turn parts
                        if hasattr(sc, 'model_turn') and sc.model_turn and hasattr(sc.model_turn, 'parts') and sc.model_turn.parts:
                            for part in sc.model_turn.parts:
                                # Capture ALL inline_data for debugging/analysis
                                if hasattr(part, 'inline_data') and part.inline_data:
                                    inline_data = part.inline_data
                                    mime_type = getattr(inline_data, "mime_type", "")
                                    data_size = len(getattr(inline_data, "data", b""))

                                    # Detect first audio chunk for accurate timing
                                    if first_audio_chunk_time is None and audio_send_complete_time is not None and mime_type.startswith("audio/pcm"):
                                        first_audio_chunk_time = time.time() - audio_send_complete_time
                                        print(f"  🎵 First audio chunk received {first_audio_chunk_time:.3f}s after audio send completed")

                                    data_info = {
                                        "response_count": response_count,
                                        "mime_type": mime_type,
                                        "data_size": data_size,
                                        "location": "model_turn_parts"
                                    }
                                    inline_data_parts.append(data_info)
                                    print(f"  [{response_count}] Inline data: {data_info['mime_type']}, {data_info['data_size']} bytes (model_turn)")

                                # Handle tool calls
                                if getattr(part, "executable_code", None) is not None:
                                    code = getattr(part.executable_code, "code", None)
                                    if code:
                                        text_parts.append(f"\n[tool:code]\n{code}\n")
                                if getattr(part, "code_execution_result", None) is not None:
                                    out = getattr(part.code_execution_result, "output", None)
                                    if out:
                                        text_parts.append(f"\n[tool:result]\n{out}\n")

                                # Handle audio in model turn
                                if hasattr(part, 'inline_data') and part.inline_data and hasattr(part.inline_data, 'data'):
                                    if not recv_via_stream:
                                        fallback_inline_audio.extend(part.inline_data.data)
                                        mime = getattr(part.inline_data, 'mime_type', '')
                                        if 'rate=' in mime:
                                            try:
                                                detected_recv_sr = int(mime.split('rate=')[-1].split(';')[0])
                                            except Exception:
                                                pass
                                        print(f"  [{response_count}] Model audio: {len(part.inline_data.data)} bytes (fallback)")

                                # Handle text in model turn
                                if hasattr(part, 'text') and part.text:
                                    text_parts.append(part.text)
                                    print(f"  [{response_count}] Model text: {part.text[:50]}...")

                                # Handle thoughts in model turn
                                if hasattr(part, 'thought') and part.thought:
                                    if isinstance(part.thought, str):
                                        thought_parts.append(part.thought)
                                        print(f"  [{response_count}] Model thought: {part.thought[:50]}...")
                                    elif isinstance(part.thought, bool):
                                        print(f"  [{response_count}] Model thought flag: <enabled>")

                        # Handle session state
                        if getattr(sc, "interrupted", False):
                            print(f"  [{response_count}] Session interrupted (will end input)")
                            try:
                                stop_sending.set()
                                await end_audio_turn(sess)
                            except Exception:
                                pass
                        if getattr(sc, "generation_complete", False):
                            print(f"  [{response_count}] Generation complete (waiting for stream to close)")
                        if getattr(sc, "turn_complete", False):
                            print(f"  [{response_count}] Turn complete (continuing to collect)")

                # Use fallback inline audio only if we didn't get streaming data
                if (not recv_via_stream) and fallback_inline_audio:
                    audio_buf.extend(fallback_inline_audio)
                    print(f"Used fallback inline audio: {len(fallback_inline_audio)} bytes")

            except Exception as e:
                print(f"  Receiver ended: {type(e).__name__}: {e}")

        # Run sender and receiver concurrently
        sender_task = asyncio.create_task(sender())
        receiver_task = asyncio.create_task(receiver())

        await sender_task
        await receiver_task

    total_time = time.time() - start_time
    voice_name = "Puck"  # Default voice for metadata

    # Write audio output
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_wav = out_dir / "output.wav"
    sample_rate = detected_recv_sr or args.tts_sr if getattr(args, "tts_sr", None) else 24_000
    if audio_buf:
        pcm = np.frombuffer(bytes(audio_buf), dtype=np.int16)
        sf.write(str(out_wav), pcm, sample_rate, subtype="PCM_16")
        duration = len(pcm) / sample_rate
        print(f"\n✓ Audio saved: {out_wav}")
        print(f"  Duration: {duration:.2f}s")
        print(f"  Size: {len(audio_buf)} bytes")
        print(f"  Sample rate: {sample_rate}Hz")
    else:
        print("\n✗ No audio received")

    # Save text parts (debug) if any
    if text_parts:
        full_text = "".join(text_parts)
        debug_text_file = out_dir / "debug_text_parts.txt"
        debug_text_file.write_text(full_text)
        print(f"\n✓ Text parts saved: {debug_text_file}")
        print(f"  Content: {full_text[:200]}...")

    # Save transcript if any
    if transcript_parts:
        full_transcript = "".join(transcript_parts)
        transcript_file = out_dir / "transcript.txt"
        transcript_file.write_text(full_transcript)
        print(f"\n✓ Transcript saved: {transcript_file}")
        print(f"  Content: {full_transcript[:200]}...")

    # Save thoughts if any
    if thought_parts:
        string_thoughts = [t for t in thought_parts if isinstance(t, str)]
        if string_thoughts:
            full_thoughts = "\n\n".join(string_thoughts)
            thought_file = out_dir / "thoughts.txt"
            thought_file.write_text(full_thoughts)
            print(f"\n✓ Thoughts saved: {thought_file}")
            print(f"  Content: {full_thoughts[:200]}...")

    # Save inline_data information if any
    if inline_data_parts:
        inline_data_file = out_dir / "inline_data_info.json"
        with open(inline_data_file, "w") as f:
            json.dump(inline_data_parts, f, indent=2)
        print(f"\n✓ Inline data info saved: {inline_data_file}")
        print(f"  Found {len(inline_data_parts)} inline_data instances")

        # Summary of inline data types
        mime_types = {}
        total_size = 0
        for item in inline_data_parts:
            mime_type = item["mime_type"]
            size = item["data_size"]
            mime_types[mime_type] = mime_types.get(mime_type, 0) + size
            total_size += size

        print(f"  Total size: {total_size} bytes")
        for mime_type, size in mime_types.items():
            print(f"  {mime_type}: {size} bytes")

    # Save structured output to JSON + transcript with improved data
    response_data = {
        "user_question": user_question,
        "assistant_response": ("".join(transcript_parts) if transcript_parts else ("".join(text_parts) if text_parts else "")),
        "assistant_thoughts": [t for t in thought_parts if isinstance(t, str)],
        "assistant_transcript": "".join(transcript_parts) if transcript_parts else "",
        "assistant_text_parts": text_parts,
        "inline_data_info": inline_data_parts,
        "timing": {
            "time_to_first_response": first_response_time,  # Includes tool execution time
            "time_to_first_audio_chunk": first_audio_chunk_time,  # Accurate audio-only timing
            "total_response_time": total_time,
        },
        "audio_info": {
            "input_file": str(input_wav_path) if args.mode == "audio" else None,
            "output_file": str(out_wav) if audio_buf else None,
            "audio_length_bytes": len(audio_buf),
            "audio_duration": (len(audio_buf) / 2) / sample_rate if audio_buf else 0,
            "sample_rate": sample_rate,
            "detected_sample_rate": detected_recv_sr,
        },
        "metadata": {
            "mode": args.mode,
            "episode_id": (episode_data or {}).get("episodes", [{}])[0].get("id") if episode_data else None,
            "model": model,
            "voice": voice_name,
        },
    }

    response_json = out_dir / "response.json"
    with open(response_json, "w") as f:
        json.dump(response_data, f, indent=2)

    transcript_txt = out_dir / "conversation.txt"
    with open(transcript_txt, "w") as f:
        f.write("=== CONVERSATION TRANSCRIPT ===\n\n")
        f.write(f"User Question: {user_question}\n\n")

        # Use transcript if available, otherwise text parts
        if transcript_parts:
            f.write(f"Assistant Response (Transcript): {''.join(transcript_parts)}\n\n")
        elif text_parts:
            f.write(f"Assistant Response (Text): {''.join(text_parts)}\n\n")
        else:
            f.write("Assistant Response: [No text response]\n\n")

        # Include thoughts if available
        if thought_parts:
            string_thoughts = [t for t in thought_parts if isinstance(t, str)]
            if string_thoughts:
                f.write(f"Assistant Thoughts: {''.join(string_thoughts)}\n\n")

        f.write("=== TIMING ===\n")
        if first_response_time is not None:
            f.write(f"Time to first response (includes tools): {first_response_time:.3f}s\n")
        else:
            f.write("Time to first response (includes tools): N/A\n")
        if first_audio_chunk_time is not None:
            f.write(f"Time to first audio chunk: {first_audio_chunk_time:.3f}s\n")
        else:
            f.write("Time to first audio chunk: N/A\n")
        f.write(f"Total response time: {total_time:.3f}s\n")
        f.write(f"Model: {model}\n")

    print(f"\n✓ Response data saved: {response_json}")
    print(f"✓ Conversation transcript saved: {transcript_txt}")
    if first_response_time:
        print(f"First response time (includes tools): {first_response_time:.2f}s")
    if first_audio_chunk_time:
        print(f"First audio chunk time: {first_audio_chunk_time:.2f}s")
    print(f"Total time: {total_time:.2f}s")


def main():
    # Pre-parse for env file
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--env-file", default=".env")
    pre_args, _ = pre.parse_known_args()
    if pre_args.env_file and Path(pre_args.env_file).exists():
        load_dotenv(pre_args.env_file)
    else:
        load_dotenv()

    ap = argparse.ArgumentParser("Gemini Realtime test runner")
    ap.add_argument("episode_json", nargs="?", help="Path to test_voice_episodes/*_episode.json")
    ap.add_argument("out_dir", help="Output folder to write input.wav/output.wav and logs")
    ap.add_argument("--tts_sr", type=int, default=24_000, help="Expected output sample rate (pcm16)")
    ap.add_argument("--env-file", default=pre_args.env_file, help="Path to .env with GEMINI_API_KEY")
    ap.add_argument("--mode", choices=["audio"], default="audio", help="Currently only 'audio' is supported")
    # Only use a model override when the flag is explicitly provided; default None
    ap.add_argument("--model", default=None)
    ap.add_argument("--voice", default=os.getenv("GEMINI_VOICE", "Puck"))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    episode_data = None
    if args.mode == "audio" and args.episode_json:
        episode_path = Path(args.episode_json)
        if not episode_path.exists():
            raise RuntimeError(f"Missing episode_json: {episode_path}")
        episode_data = json.loads(episode_path.read_text())

    # Run the realtime roundtrip
    import asyncio
    asyncio.run(run_gemini_audio_roundtrip(args, episode_data))


if __name__ == "__main__":
    main()
