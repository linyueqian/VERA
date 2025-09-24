#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
sonic_run_episode.py
Single-episode Nova Sonic file-in → file-out runner with robust streaming.

What this does
--------------
- Loads AWS creds from env / .env
- (Optional) Reads an episode JSON (MRCR or normal) and injects context as SYSTEM history
- Streams USER audio @32ms cadence (512 samples @16k, interactive:true)
- Collects ASSISTANT audioOutput, writes output.wav (24k/mono/16-bit), plus response.json & conversation.txt
- Suppresses benign listener error: `ValidationException: Timed out waiting for input events`

Key fixes vs prior versions
---------------------------
- Stop condition = (completionEnd + quiet tail) OR (soft budget = input_duration × factor) OR (hard cap)
- First-audio metric = time from *after finishing all user audio* to first assistant audio (ignores barge-in)
"""

import os
import sys
import json
import time
import base64
import asyncio
import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import soundfile as sf
from dotenv import load_dotenv

# ====== audio/cadence ======
INPUT_SAMPLE_RATE = 16_000
OUTPUT_SAMPLE_RATE = 24_000
CHUNK_SIZE = 512                                  # 32ms @16k
FRAME_DUR = CHUNK_SIZE / INPUT_SAMPLE_RATE        # 0.032 s

# ====== default timing/limits (all overridable by CLI) ======
TAIL_QUIET_GAP = 1.2          # seconds of silence after last assistant chunk before we stop
ASSIST_MULTIPLIER = 2.0       # soft budget = input_duration * this
ASSIST_MAX_SECS = 240.0       # hard cap for assistant audio seconds
FIRST_AUDIO_TIMEOUT = 30.0    # how long we wait for first assistant audio before giving up (after input end)

# ====== AWS Bedrock imports ======
from aws_sdk_bedrock_runtime.client import (
    BedrockRuntimeClient,
    InvokeModelWithBidirectionalStreamOperationInput,
)
from aws_sdk_bedrock_runtime.models import (
    InvokeModelWithBidirectionalStreamInputChunk,
    BidirectionalInputPayloadPart,
    ValidationException,
)
from aws_sdk_bedrock_runtime.config import (
    Config,
    HTTPAuthSchemeResolver,
    SigV4AuthScheme,
)
from smithy_aws_core.credentials_resolvers.environment import (
    EnvironmentCredentialsResolver,
)

# ----------------- small utils -----------------
DEBUG = False
def dprint(msg: str):
    if DEBUG:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr)

def require_env(name: str) -> str:
    val = os.getenv(name)
    if not val:
        raise RuntimeError(f"Missing env: {name}")
    return val

def _mono(sig: np.ndarray) -> np.ndarray:
    return sig if sig.ndim == 1 else sig.mean(axis=1)

def _resample(sig: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return sig
    import torch
    import torchaudio.functional as AF
    wav_f32 = torch.from_numpy(sig.astype(np.float32) / 32768.0).unsqueeze(0)
    wav_rs = AF.resample(wav_f32, orig_sr, target_sr)
    return (wav_rs.squeeze().numpy() * 32768).astype(np.int16)

def _chunk(sig: np.ndarray, frame_len: int) -> List[np.ndarray]:
    pad = (-len(sig)) % frame_len
    if pad:
        sig = np.concatenate([sig, np.zeros(pad, dtype=sig.dtype)])
    return [sig[i:i+frame_len] for i in range(0, len(sig), frame_len)]

def load_first_user_audio(episode_json: Path) -> Path:
    data = json.loads(episode_json.read_text())
    eps = data.get("episodes", [])
    if not eps:
        raise RuntimeError("No episodes found in JSON")
    turns = eps[0].get("turns", [])
    ep_id = eps[0].get("id")
    for t in turns:
        if t.get("role") == "user" and t.get("audio_file"):
            p = Path(t["audio_file"])
            if p.exists():
                return p
            if ep_id:
                cand = (Path.cwd() / "test_voice_episodes/audio/" / f"{ep_id}.wav").resolve()
                if cand.exists():
                    return cand
            cand2 = (Path.cwd() / "test_voice_episodes/audio/" / Path(t["audio_file"]).name).resolve()
            if cand2.exists():
                return cand2
            return p  # will fail downstream
    raise RuntimeError("No user turn with audio_file found")

def to_mono_pcm16(path: Path, sr_target: int = 16_000) -> Tuple[np.ndarray, int]:
    wav, sr = sf.read(str(path), dtype="int16")
    if wav.ndim == 2:
        wav = wav.mean(axis=1).astype(np.int16)
    if sr == sr_target:
        return wav, sr
    # Try scipy first; fallback to torch
    try:
        import scipy.signal as ss
        g = np.gcd(sr, sr_target)
        up, down = sr_target // g, sr // g
        wav_f = wav.astype(np.float32) / 32768.0
        y = ss.resample_poly(wav_f, up, down)
        y = np.clip(y, -1.0, 1.0)
        return (y * 32767).astype(np.int16), sr_target
    except Exception:
        return _resample(wav, sr, sr_target), sr_target

def parse_mrcr_context(context: str):
    """Parse MRCR context document into conversation messages: [{'role': 'user'|'assistant', 'content': str}, ...]"""
    lines = context.splitlines()
    messages, current_role, current_content = [], None, []
    for line in lines:
        if line.startswith("User:"):
            if current_role and current_content:
                messages.append({"role": current_role, "content": "\n".join(current_content).strip()})
            current_role, current_content = "user", [line[5:].strip()]
        elif line.startswith("Assistant:"):
            if current_role and current_content:
                messages.append({"role": current_role, "content": "\n".join(current_content).strip()})
            current_role, current_content = "assistant", [line[10:].strip()]
        else:
            current_content.append(line)
    if current_role and current_content:
        messages.append({"role": current_role, "content": "\n".join(current_content).strip()})
    return messages
# ----------------------------------------------


class BedrockStreamManager:
    """
    Nova Sonic stream manager with robust listener:
      - sessionStart
      - promptStart (requests audio output w/ voice)
      - system TEXT (start/input/end) — can be default or injected MRCR history
      - USER AUDIO (interactive:true)
      - promptEnd / sessionEnd
    Listener enqueues audioOutput; it suppresses ValidationException timeouts gracefully.
    """

    def __init__(self, model_id: str, region: str, voice_id: str = "matthew",
                 max_tokens: int = 2048):
        self.model_id = model_id
        self.region = region
        self.voice_id = voice_id
        self.max_tokens = max_tokens

        self.prompt_name = "p"
        self.content_idx = 0
        self.content_name = f"c{self.content_idx}"

        cfg = Config(
            endpoint_uri=f"https://bedrock-runtime.{region}.amazonaws.com",
            region=region,
            aws_credentials_identity_resolver=EnvironmentCredentialsResolver(),
            http_auth_scheme_resolver=HTTPAuthSchemeResolver(),
            http_auth_schemes={"aws.auth#sigv4": SigV4AuthScheme()},
        )
        self.client = BedrockRuntimeClient(config=cfg)

        self.audio_out_q: "asyncio.Queue[Optional[bytes]]" = asyncio.Queue()
        self.is_active = False
        self.barge_in: bool = False
        self.interrupted = False
        self.listener_task: Optional[asyncio.Task] = None

        # Assistant turn tracking
        self.assistant_started = asyncio.Event()
        self.assistant_done = asyncio.Event()
        self.last_audio_ts: float = 0.0

    # ----- Event JSON templates (built at runtime for max_tokens/voice) -----
    def _start_session_event(self) -> str:
        return json.dumps({
            "event": {
                "sessionStart": {
                    "inferenceConfiguration": {
                        "maxTokens": self.max_tokens,
                        "topP": 0.9,
                        "temperature": 0.7
                    }
                }
            }
        })

    def _start_prompt_event(self) -> str:
        return json.dumps({
            "event": {
                "promptStart": {
                    "promptName": self.prompt_name,
                    "textOutputConfiguration": {"mediaType": "text/plain"},
                    "audioOutputConfiguration": {
                        "mediaType": "audio/lpcm",
                        "sampleRateHertz": OUTPUT_SAMPLE_RATE,
                        "sampleSizeBits": 16,
                        "channelCount": 1,
                        "voiceId": self.voice_id,
                        "encoding": "base64",
                        "audioType": "SPEECH",
                    },
                    "toolUseOutputConfiguration": {"mediaType": "application/json"},
                    "toolConfiguration": {"tools": []},
                }
            }
        })

    SYS_START = (
        '{"event":{"contentStart":{"promptName":"%s","contentName":"sys",'
        '"role":"SYSTEM","type":"TEXT","interactive":true,'
        '"textInputConfiguration":{"mediaType":"text/plain"}}}}'
    )
    SYS_TEXT = (
        '{"event":{"textInput":{"promptName":"%s","contentName":"sys","content":"%s"}}}'
    )
    SYS_END = '{"event":{"contentEnd":{"promptName":"%s","contentName":"sys"}}}'

    CONTENT_START_EVENT = (
        '{"event":{"contentStart":{"promptName":"%s","contentName":"%s","type":"AUDIO",'
        '"interactive":true,"role":"USER","audioInputConfiguration":{"mediaType":"audio/lpcm",'
        '"sampleRateHertz":16000,"sampleSizeBits":16,"channelCount":1,"audioType":"SPEECH","encoding":"base64"}}}}'
    )
    AUDIO_EVENT = (
        '{"event":{"audioInput":{"promptName":"%s","contentName":"%s","content":"%s"}}}'
    )
    CONTENT_END_EVENT = '{"event":{"contentEnd":{"promptName":"%s","contentName":"%s"}}}'
    PROMPT_END_EVENT = '{"event":{"promptEnd":{"promptName":"%s"}}}'
    SESSION_END_EVENT = '{"event":{"sessionEnd":{}}}'

    # -----------------------------------------------------------------------
    async def _send_event(self, event_json: str):
        dprint(f"⇢ {event_json[:90]}...")
        chunk = InvokeModelWithBidirectionalStreamInputChunk(
            value=BidirectionalInputPayloadPart(bytes_=event_json.encode("utf-8"))
        )
        await self.stream_rsp.input_stream.send(chunk)

    async def start_default(self, system_prompt: str):
        """Start with a simple system text block."""
        self.stream_rsp = await self.client.invoke_model_with_bidirectional_stream(
            InvokeModelWithBidirectionalStreamOperationInput(model_id=self.model_id)
        )
        self.is_active = True

        await self._send_event(self._start_session_event())
        await self._send_event(self._start_prompt_event())

        # SYSTEM text
        safe_prompt = system_prompt.replace("\\", "\\\\").replace('"', '\\"')
        await self._send_event(self.SYS_START % self.prompt_name)
        await self._send_event(self.SYS_TEXT % (self.prompt_name, safe_prompt))
        await self._send_event(self.SYS_END % self.prompt_name)

        self.listener_task = asyncio.create_task(self._listen_loop())

    async def start_with_context(self, system_message_json_events: list):
        """Start without default SYSTEM, then inject ready-made SYSTEM events (MRCR history)."""
        self.stream_rsp = await self.client.invoke_model_with_bidirectional_stream(
            InvokeModelWithBidirectionalStreamOperationInput(model_id=self.model_id)
        )
        self.is_active = True
        await self._send_event(self._start_session_event())
        await self._send_event(self._start_prompt_event())

        for raw in system_message_json_events:
            await self._send_event(raw)

        self.listener_task = asyncio.create_task(self._listen_loop())

    def next_content(self):
        """Move to next content block (for barge-in handling)."""
        self.content_idx += 1
        self.content_name = f"c{self.content_idx}"
        self.interrupted = False

    async def begin_audio(self):
        await self._send_event(self.CONTENT_START_EVENT % (self.prompt_name, self.content_name))

    def send_chunk(self, pcm_bytes: bytes):
        if not self.is_active:
            return
        b64 = base64.b64encode(pcm_bytes).decode("utf-8")
        asyncio.get_event_loop().create_task(
            self._send_event(self.AUDIO_EVENT % (self.prompt_name, self.content_name, b64))
        )

    async def end_audio(self):
        await self._send_event(self.CONTENT_END_EVENT % (self.prompt_name, self.content_name))

    async def close(self):
        if not self.is_active:
            return
        # Close prompt + session; listener will see sessionEnd or just finish quietly
        await self._send_event(self.PROMPT_END_EVENT % self.prompt_name)
        await self._send_event(self.SESSION_END_EVENT)
        try:
            await self.stream_rsp.input_stream.close()
        except Exception:
            pass
        self.is_active = False

        if self.listener_task and not self.listener_task.done():
            try:
                await asyncio.wait_for(self.listener_task, timeout=10.0)
            except Exception:
                dprint("listener did not finish in time; canceling.")

    async def _listen_loop(self):
        """Listener: collects audio; sets assistant_started/done; suppresses idle timeouts."""
        loop = asyncio.get_event_loop()
        try:
            while self.is_active:
                try:
                    output = await self.stream_rsp.await_output()
                    res = await output[1].receive()
                except ValidationException as e:
                    # Server-side idle; just keep going
                    if "Timed out waiting for input events" in str(e):
                        dprint("⇠ [idle timeout] continuing")
                        continue
                    dprint(f"⇠ [ValidationException] {e}")
                    continue
                except Exception as e:
                    dprint(f"⇠ [listener recv error] {e}")
                    continue

                if not getattr(res, "value", None) or not getattr(res.value, "bytes_", None):
                    continue

                msg_raw = res.value.bytes_.decode("utf-8", errors="ignore")
                dprint("⇠ " + (msg_raw[:120] + ("..." if len(msg_raw) > 120 else "")))
                try:
                    msg = json.loads(msg_raw)
                except Exception:
                    continue

                ev = msg.get("event", {})
                if not ev:
                    continue

                # Detect assistant audio start based on contentStart (ASSISTANT/AUDIO) or first audioOutput
                if "contentStart" in ev:
                    cs = ev["contentStart"]
                    role = cs.get("role")
                    ctype = cs.get("type")
                    if role == "ASSISTANT" and ctype == "AUDIO":
                        self.assistant_started.set()

                if "audioOutput" in ev:
                    try:
                        pcm = base64.b64decode(ev["audioOutput"]["content"])
                        await self.audio_out_q.put(pcm)
                        self.assistant_started.set()
                        self.last_audio_ts = loop.time()
                    except Exception:
                        pass

                if "textOutput" in ev:
                    txt = ev["textOutput"].get("content", "")
                    if '"interrupted"' in txt:
                        print(">>> INTERRUPTED FLAG:", txt)
                        self.barge_in = True
                        self.interrupted = True
                        # drain any stale audio quickly
                        try:
                            while True:
                                self.audio_out_q.get_nowait()
                        except asyncio.QueueEmpty:
                            pass

                if "completionEnd" in ev:
                    # Model indicates turn end
                    self.assistant_done.set()

                if "sessionEnd" in ev:
                    self.is_active = False
                    break
        finally:
            # Always release writer
            await self.audio_out_q.put(None)

# -------- MRCR context injection helpers --------
def build_mrcr_system_events_from_documents(docs: list, prompt_name: str) -> list:
    """Return raw JSON event strings (SYS_START → SYS_TEXT → SYS_END) with chat history injected."""
    if not docs:
        return []
    combined = "You are a helpful assistant."
    for doc in docs:
        content = doc.get("content", "")
        if not content:
            continue
        msgs = parse_mrcr_context(content)
        if msgs:
            combined += "\n\nPrevious conversation:\n\n"
            for m in msgs:
                role = "User" if m["role"].lower().startswith("user") else "Assistant"
                combined += f"{role}: {m['content']}\n\n"
        else:
            combined += "\n\nContext:\n" + content + "\n"

    esc = combined.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")

    SYS_START = (
        '{"event":{"contentStart":{"promptName":"%s","contentName":"sys",'
        '"role":"SYSTEM","type":"TEXT","interactive":true,'
        '"textInputConfiguration":{"mediaType":"text/plain"}}}}'
    ) % prompt_name
    SYS_TEXT = (
        '{"event":{"textInput":{"promptName":"%s","contentName":"sys","content":"%s"}}}'
    ) % (prompt_name, esc)
    SYS_END = '{"event":{"contentEnd":{"promptName":"%s","contentName":"sys"}}}' % prompt_name

    return [SYS_START, SYS_TEXT, SYS_END]
# ------------------------------------------------


async def run_episode(out_dir: Path,
                      episode_json: Optional[Path],
                      model: str,
                      region: str,
                      voice: str,
                      mode: str,
                      system_prompt: str,
                      tail_quiet_gap: float,
                      assist_multiplier: float,
                      assist_max_secs: float,
                      max_tokens: int,
                      first_audio_timeout: float):
    """
    Core runner.
    Stop rule = (completionEnd + tail_quiet_gap) OR (soft budget = input_duration * assist_multiplier) OR (hard cap).
    First-audio metric = first assistant audio AFTER user input has fully finished.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load episode + locate user audio + gather MRCR context docs
    input_wav_path = None
    user_text = ""
    context_docs = []
    if mode == "audio" and episode_json:
        episode = json.loads(episode_json.read_text())
        eps = episode.get("episodes", [])
        if eps:
            first = eps[0]
            for t in first.get("turns", []):
                if t.get("role") == "user":
                    user_text = t.get("text_content", "") or user_text
                    break
            context_docs = first.get("context_documents", [])
        user_audio = load_first_user_audio(episode_json)
        input_wav_path = out_dir / "input.wav"
        # symlink if possible; else copy
        try:
            if input_wav_path.exists() or input_wav_path.is_symlink():
                input_wav_path.unlink()
            os.symlink(user_audio, input_wav_path)
        except OSError:
            wav, sr = sf.read(str(user_audio))
            sf.write(str(input_wav_path), wav, sr)

    # Prepare AWS creds
    require_env("AWS_ACCESS_KEY_ID")
    require_env("AWS_SECRET_ACCESS_KEY")
    os.environ["AWS_DEFAULT_REGION"] = region

    mgr = BedrockStreamManager(model_id=model, region=region, voice_id=voice, max_tokens=max_tokens)

    # Start session
    if context_docs:
        sys_events = build_mrcr_system_events_from_documents(context_docs, mgr.prompt_name)
        await mgr.start_with_context(sys_events)
    else:
        await mgr.start_default(system_prompt=system_prompt)

    await mgr.begin_audio()

    # Timers and buffers
    t0 = time.time()
    audio_input_finished_ts: Optional[float] = None
    first_assistant_audio_ts: Optional[float] = None               # overall first audio (may occur during streaming)
    first_assistant_after_input_ts: Optional[float] = None         # we want THIS metric
    audio_buf = bytearray()

    # For writer coordination
    audio_input_finished_event = asyncio.Event()

    if mode == "audio" and input_wav_path:
        # Read/send user audio
        pcm16, sr = to_mono_pcm16(input_wav_path, sr_target=INPUT_SAMPLE_RATE)
        frames = _chunk(pcm16, CHUNK_SIZE)
        input_duration = len(pcm16) / INPUT_SAMPLE_RATE

        # Budgets
        soft_budget_secs = max(10.0, min(assist_max_secs, input_duration * assist_multiplier))
        hard_cap_secs = assist_max_secs

        async def writer_task():
            nonlocal first_assistant_audio_ts, first_assistant_after_input_ts, audio_buf
            total_written = 0
            temp_audio_chunks = []  # Store chunks temporarily

            start_ts = time.time()
            while True:
                    # Try to get next audio chunk (or None sentinel) with small timeout
                    data = None
                    got_sentinel = False
                    try:
                        data = await asyncio.wait_for(mgr.audio_out_q.get(), timeout=0.5)
                        if data is None:
                            got_sentinel = True
                    except asyncio.TimeoutError:
                        pass

                    now = time.time()

                    # If assistant never started and we've waited long enough, give up
                    if (audio_input_finished_event.is_set() and
                        not mgr.assistant_started.is_set() and
                        (now - (audio_input_finished_ts or t0)) >= FIRST_AUDIO_TIMEOUT):
                        dprint(f"[writer] no assistant audio after {FIRST_AUDIO_TIMEOUT}s, giving up")
                        break

                    # Hard cap guard
                    if (now - (audio_input_finished_ts or t0)) >= hard_cap_secs:
                        dprint(f"[writer] hard cap {hard_cap_secs}s reached – stopping")
                        break

                    # Collect incoming audio
                    if data and not got_sentinel:
                        if first_assistant_audio_ts is None:
                            first_assistant_audio_ts = now
                        if audio_input_finished_event.is_set() and first_assistant_after_input_ts is None:
                            first_assistant_after_input_ts = now

                        pcm = np.frombuffer(data, dtype=np.int16)
                        if pcm.size:
                            temp_audio_chunks.append(pcm)  # Collect chunks
                            audio_buf.extend(pcm.tobytes())
                            total_written += pcm.size
                            mgr.last_audio_ts = now  # redundant safeguard

                    # Stop if model said it's done and we've observed a quiet tail
                    if mgr.assistant_done.is_set():
                        if mgr.last_audio_ts and (now - mgr.last_audio_ts) >= tail_quiet_gap:
                            dprint(f"[writer] completionEnd + quiet tail {tail_quiet_gap}s – stopping")
                            break

                    # Soft budget fallback (relative to end of input if we have it; else session start)
                    ref = (audio_input_finished_ts or t0)
                    if (now - ref) >= soft_budget_secs:
                        if mgr.last_audio_ts and (now - mgr.last_audio_ts) >= tail_quiet_gap:
                            dprint(f"[writer] soft budget {soft_budget_secs:.1f}s + quiet tail – stopping")
                            break

                    # If listener ended (sentinel), give a small tail and exit
                    if got_sentinel:
                        if mgr.last_audio_ts and (now - mgr.last_audio_ts) < tail_quiet_gap:
                            await asyncio.sleep(tail_quiet_gap - (now - mgr.last_audio_ts))
                        break

            # Write all collected audio at once after the loop
            if temp_audio_chunks:
                with sf.SoundFile(str(out_dir / "output.wav"), "w",
                                  samplerate=OUTPUT_SAMPLE_RATE, channels=1, subtype="PCM_16") as fout:
                    for chunk in temp_audio_chunks:
                        fout.write(chunk)
                print(f"Wrote {len(temp_audio_chunks)} audio chunks to output.wav")

            secs = total_written / OUTPUT_SAMPLE_RATE
            print(f"writer: wrote {secs:.2f}s assistant audio")

        # Start writer first (so we never miss early audio)
        wt = asyncio.create_task(writer_task())

        # Stream audio with cadence
        for i, f in enumerate(frames):
            if mgr.barge_in:
                await mgr.end_audio()
                mgr.next_content()
                await mgr.begin_audio()
                mgr.barge_in = False
            mgr.send_chunk(f.tobytes())
            await asyncio.sleep(FRAME_DUR)
            if (i + 1) % 200 == 0:
                print(f"  sent {i+1}/{len(frames)} frames")

        # Mark end of user input
        audio_input_finished_ts = time.time()
        audio_input_finished_event.set()
        print(f"✓ Audio input finished at: {audio_input_finished_ts - t0:.3f}s from start")
        await mgr.end_audio()

        # If assistant never started after input end, wait up to FIRST_AUDIO_TIMEOUT to see if it starts
        if first_assistant_after_input_ts is None:
            try:
                await asyncio.wait_for(mgr.assistant_started.wait(), timeout=FIRST_AUDIO_TIMEOUT)
                # If assistant started only before input end, we still need *after* input end; wait for next chunk
                if first_assistant_audio_ts and first_assistant_audio_ts < audio_input_finished_ts:
                    # wait until we actually get an audioOutput after input finish (writer sets it)
                    # If nothing arrives, we’ll fall back to budgets in writer.
                    pass
            except asyncio.TimeoutError:
                dprint(f"[runner] no assistant audio within {FIRST_AUDIO_TIMEOUT}s after input end")

        # Let writer decide proper stop (completionEnd+quiet or budgets), then close
        await wt
        await mgr.close()

    else:
        # TTS-only mode (no user audio streamed here)
        await mgr.end_audio()
        await asyncio.sleep(0.25)
        await mgr.close()

    # Build structured outputs
    total_time = time.time() - t0
    out_wav = out_dir / "output.wav"
    wrote_audio = out_wav.exists() and out_wav.stat().st_size > 0

    # Metrics we report
    time_to_first_audio_after_input = None
    if audio_input_finished_ts and first_assistant_after_input_ts:
        time_to_first_audio_after_input = first_assistant_after_input_ts - audio_input_finished_ts

    resp = {
        "user_question": user_text,
        "assistant_response": "",  # audio-focused
        "timing": {
            "audio_input_finished_offset": (audio_input_finished_ts - t0) if audio_input_finished_ts else None,
            "time_to_first_assistant_audio_after_input": time_to_first_audio_after_input,
            "total_elapsed": total_time,
        },
        "audio": {
            "input_path": str(input_wav_path) if input_wav_path else None,
            "output_path": str(out_wav) if wrote_audio else None,
            "bytes": out_wav.stat().st_size if wrote_audio else 0,
            "sample_rate": OUTPUT_SAMPLE_RATE if wrote_audio else None,
        },
        "meta": {
            "mode": mode,
            "model": model,
            "region": region,
            "voice": voice,
            "tail_quiet_gap": tail_quiet_gap,
            "assist_multiplier": assist_multiplier,
            "assist_max_secs": assist_max_secs,
            "max_tokens": max_tokens,
        }
    }
    (out_dir / "response.json").write_text(json.dumps(resp, indent=2))

    with open(out_dir / "conversation.txt", "w") as f:
        f.write("=== SONIC CONVERSATION TRANSCRIPT ===\n\n")
        f.write(f"User Question: {user_text}\n\n")
        f.write("(Audio response captured in output.wav)\n\n")
        if time_to_first_audio_after_input is not None:
            f.write(f"Time to first assistant audio (after input finished): {time_to_first_audio_after_input:.3f}s\n")
        if audio_input_finished_ts is not None:
            f.write(f"Audio input finished at: {audio_input_finished_ts - t0:.3f}s from start\n")
        f.write(f"Total elapsed: {total_time:.3f}s\n")
        f.write(f"Model: {model}\nRegion: {region}\nVoice: {voice}\n")

    print("\n=== SONIC SUMMARY ===")
    print(f"Time to first assistant audio (after input): "
          f"{time_to_first_audio_after_input:.3f}s" if time_to_first_audio_after_input is not None else
          "Time to first assistant audio (after input): N/A")
    if audio_input_finished_ts is not None:
        print(f"Audio input finished at: {audio_input_finished_ts - t0:.3f}s from start")
    print(f"Total time: {total_time:.3f}s")
    print(f"Output: {out_wav if wrote_audio else '(no audio)'}")


def main():
    # Early parse to load .env
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--env-file", default=".env")
    pre_args, _ = pre.parse_known_args()
    if pre_args.env_file and Path(pre_args.env_file).exists():
        load_dotenv(pre_args.env_file)
    else:
        load_dotenv()

    ap = argparse.ArgumentParser("Nova Sonic single-episode runner")
    ap.add_argument("out_dir", help="Output folder (created if missing)")
    ap.add_argument("episode_json", nargs="?", help="Path to *_episode.json (required for mode=audio)")
    ap.add_argument("--mode", choices=["audio", "tts"], default="audio",
                    help="'audio': stream episode audio; 'tts': request speech-only reply")
    ap.add_argument("--region", default="us-east-1")
    ap.add_argument("--model", default="amazon.nova-sonic-v1:0")
    ap.add_argument("--voice", default="matthew")
    ap.add_argument("--system-prompt", default="You are a helpful assistant. Speak your replies out loud.")
    ap.add_argument("--max-tokens", type=int, default=2048, help="sessionStart.inferenceConfiguration.maxTokens")
    ap.add_argument("--tail-quiet-gap", type=float, default=TAIL_QUIET_GAP)
    ap.add_argument("--assist-multiplier", type=float, default=ASSIST_MULTIPLIER,
                    help="soft budget multiplier: input_duration * multiplier")
    ap.add_argument("--assist-max-secs", type=float, default=ASSIST_MAX_SECS,
                    help="hard cap on assistant audio seconds")
    ap.add_argument("--first-audio-timeout", type=float, default=FIRST_AUDIO_TIMEOUT,
                    help="max seconds to wait for first assistant audio after input finished")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    global DEBUG
    DEBUG = args.debug

    out_dir = Path(args.out_dir)
    episode_json = Path(args.episode_json) if args.episode_json else None

    try:
        asyncio.run(
            run_episode(
                out_dir=out_dir,
                episode_json=episode_json,
                model=args.model,
                region=args.region,
                voice=args.voice,
                mode=args.mode,
                system_prompt=args.system_prompt,
                tail_quiet_gap=args.tail_quiet_gap,
                assist_multiplier=args.assist_multiplier,
                assist_max_secs=args.assist_max_secs,
                max_tokens=args.max_tokens,
                first_audio_timeout=args.first_audio_timeout,
            )
        )
    except Exception as e:
        print("Sonic inference failed:", e)
        raise


if __name__ == "__main__":
    main()
