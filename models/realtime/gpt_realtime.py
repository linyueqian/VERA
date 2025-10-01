import os
import json
import base64
import argparse
import urllib.parse
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import soundfile as sf
from websocket import create_connection, WebSocketException
from dotenv import load_dotenv


def require_env(name: str) -> str:
    val = os.getenv(name)
    if not val:
        raise RuntimeError(f"Missing env: {name}")
    return val


def find_episode_by_audio_file(episode_json: Path, target_audio_file: Path) -> dict:
    """Find the episode that matches the target audio file."""
    data = json.loads(episode_json.read_text())
    eps = data.get("episodes", [])
    if not eps:
        raise RuntimeError("No episodes found in JSON")

    target_stem = target_audio_file.stem
    print(f"Looking for episode matching audio file: {target_stem}")

    for episode in eps:
        ep_id = episode.get("id", "")
        if ep_id == target_stem:
            print(f"Found exact episode match: {ep_id}")
            return episode

    for episode in eps:
        turns = episode.get("turns", [])
        for turn in turns:
            if turn.get("role") == "user" and turn.get("audio_file"):
                audio_path = Path(turn["audio_file"])
                if audio_path.stem == target_stem or audio_path.name == target_audio_file.name:
                    print(f"Found episode by audio file match: {episode.get('id')}")
                    return episode

    print(f"WARNING: No episode found matching {target_stem}, using first episode: {eps[0].get('id')}")
    return eps[0]

def load_user_audio_for_episode(episode: dict, target_audio_file: Path = None) -> Path:
    """Load the user audio file for a specific episode."""
    turns = episode.get("turns", [])
    ep_id = episode.get("id")

    for t in turns:
        if t.get("role") == "user" and t.get("audio_file"):
            p = Path(t["audio_file"])
            if p.exists():
                return p

            if target_audio_file and target_audio_file.exists():
                print(f"Using provided target audio file: {target_audio_file}")
                return target_audio_file

            if ep_id:
                cand = (Path.cwd() / "test_voice_episodes/audio/" / f"{ep_id}.wav").resolve()
                if cand.exists():
                    return cand
            cand2 = (Path.cwd() / "test_voice_episodes/audio/" / Path(t["audio_file"]).name).resolve()
            if cand2.exists():
                return cand2
            return p
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


def parse_mrcr_context(context: str):
    """Parse MRCR context document into conversation messages"""
    messages = []

    lines = context.split('\n')
    current_role = None
    current_content = []

    for line in lines:
        if line.startswith('User:'):
            if current_role and current_content:
                messages.append({"role": current_role, "content": '\n'.join(current_content).strip()})
            current_role = "user"
            current_content = [line[5:].strip()]
        elif line.startswith('Assistant:'):
            if current_role and current_content:
                messages.append({"role": current_role, "content": '\n'.join(current_content).strip()})
            current_role = "assistant"
            current_content = [line[10:].strip()]
        else:
            if current_content is not None:
                current_content.append(line)

    if current_role and current_content:
        messages.append({"role": current_role, "content": '\n'.join(current_content).strip()})

    return messages


def main():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--env-file", default=".env")
    pre_args, _ = pre.parse_known_args()
    if pre_args.env_file and Path(pre_args.env_file).exists():
        load_dotenv(pre_args.env_file)
    else:
        load_dotenv()

    ap = argparse.ArgumentParser("Azure GPT Realtime test runner")
    ap.add_argument("episode_json", nargs="?", help="Path to test_voice_episodes/*_episode.json (ignored in --mode tts)")
    ap.add_argument("out_dir", help="Output folder to write input.wav and/or output.wav")
    ap.add_argument("--tts_sr", type=int, default=24_000, help="Expected output sample rate (pcm16)")
    ap.add_argument("--env-file", default=pre_args.env_file, help="Path to .env with AZURE_* vars")
    ap.add_argument("--mode", choices=["audio", "tts"], default="audio", help="'audio': stream episode audio; 'tts': request audio-only reply")
    ap.add_argument("--target-audio", type=Path, help="Specific audio file to use (will find matching episode)")
    args = ap.parse_args()

    episode_json = Path(args.episode_json)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    episode_data = None
    selected_episode = None
    context_documents = []
    target_audio_file = None

    if args.mode == "audio" and episode_json:
        episode_data = json.loads(episode_json.read_text())

        if args.target_audio:
            target_audio_file = args.target_audio
            print(f"Using specified target audio: {target_audio_file}")
        else:
            output_stem = out_dir.name
            if output_stem.startswith("vera_"):
                possible_audio_files = [
                    Path(f"data/final_dataset/voice/aime_voice_episodes_audio/{output_stem}.wav"),
                    Path(f"data/final_dataset/voice/mrcr_voice_episodes_audio/{output_stem}.wav"),
                    Path(f"data/final_dataset/voice/simpleqa_voice_episodes_audio/{output_stem}.wav"),
                    Path(f"data/final_dataset/voice/gpqa_voice_episodes_audio/{output_stem}.wav"),
                    Path(f"data/final_dataset/voice/browsecomp_voice_episodes_audio/{output_stem}.wav"),
                ]
                for audio_file in possible_audio_files:
                    if audio_file.exists() and audio_file.stem == output_stem:
                        target_audio_file = audio_file
                        print(f"Found exact matching audio file: {target_audio_file}")
                        break

                if not target_audio_file:
                    print(f"WARNING: No exact audio file found for {output_stem}")

        if target_audio_file:
            selected_episode = find_episode_by_audio_file(episode_json, target_audio_file)
        else:
            if episode_data.get("episodes"):
                selected_episode = episode_data["episodes"][0]
                print("No target audio specified, using first episode")

        if selected_episode:
            context_documents = selected_episode.get("context_documents", [])
            print(f"Found {len(context_documents)} context documents for episode {selected_episode.get('id')}")

    input_wav_path = out_dir / "input.wav"
    if args.mode == "audio":
        if not episode_json:
            raise RuntimeError("episode_json is required in audio mode")
        if not selected_episode:
            raise RuntimeError("No episode selected")
        user_audio = load_user_audio_for_episode(selected_episode, target_audio_file)
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

    endpoint = require_env("AZURE_OPENAI_ENDPOINT").strip().rstrip("/")
    host = endpoint.replace("https://", "").replace("http://", "")
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-realtime")
    api_key = require_env("AZURE_OPENAI_API_KEY")
    api_version = os.getenv("OPENAI_API_VERSION", "2025-04-01-preview")

    url = (
        f"wss://{host}/openai/realtime"
        f"?api-version={urllib.parse.quote(api_version)}"
        f"&deployment={urllib.parse.quote(deployment)}"
    )
    headers = [f"api-key: {api_key}", "User-Agent: azure-realtime-vera/1.0"]

    print("About to connect...")
    print("Host:", host)
    print("Deployment:", deployment)
    print("API Version:", api_version)
    print("URL:", url)
    print("Headers:", headers)
    try:
        ws = create_connection(url, header=headers, timeout=60)
        print("Connected successfully!")
    except Exception as e:
        print("Connection failed:", e)
        raise
    try:
        session_config = {
            "modalities": ["audio", "text"],
            "model": "gpt-realtime",
            "voice": "alloy",
            "output_audio_format": "pcm16",
            "input_audio_format": "pcm16",
            "input_audio_transcription": {
                "model": "whisper-1"
            },
            "instructions": "You are a concise assistant. Return only the final answer when obvious.",
            "turn_detection": {
                "type": "server_vad",
                "threshold": 0.8,
                "prefix_padding_ms": 300,
                "silence_duration_ms": 1000,
                "create_response": False,
                "interrupt_response": False
            },
        }
        
        ws.send(
            json.dumps(
                {
                    "type": "session.update",
                    "session": session_config,
                }
            )
        )
        print("Session update sent.")

        print("Waiting for session.updated...")
        session_updated = False
        while not session_updated:
            try:
                msg = ws.recv()
                if isinstance(msg, str):
                    data = json.loads(msg)
                    msg_type = data.get("type")
                    print(f"Received during session setup: {msg_type}")
                    if msg_type == "session.updated":
                        session_updated = True
                        print("Session updated confirmed.")
                        break
            except Exception as e:
                print(f"Error waiting for session.updated: {e}")
                break

        if context_documents:
            print(f"Injecting {len(context_documents)} context documents...")

            is_mrcr = False
            if selected_episode:
                episode_id = selected_episode.get("id", "").lower()
                track = selected_episode.get("track", "")
                is_mrcr = "mrcr" in episode_id or track == "long_context"

            for i, doc in enumerate(context_documents):
                content = doc.get("content", "")
                if content and is_mrcr:
                    print(f"Injecting MRCR conversation context from document {i+1}")
                    ws.send(
                        json.dumps(
                            {
                                "type": "conversation.item.create",
                                "item": {
                                    "type": "message",
                                    "role": "system",
                                    "content": [{"type": "input_text", "text": f"Previous conversation history:\n\n{content}"}],
                                },
                            }
                        )
                    )
                elif content:
                    ws.send(
                        json.dumps(
                            {
                                "type": "conversation.item.create",
                                "item": {
                                    "type": "message",
                                    "role": "assistant",
                                    "content": [{"type": "input_text", "text": content}],
                                },
                            }
                        )
                    )
            print(f"Context injection complete.")

        if args.mode == "audio":
            pcm16, sr = to_mono_pcm16(input_wav_path, target_sr=24_000)

            frame = sr // 10
            for i in range(0, len(pcm16), frame):
                chunk = pcm16[i : i + frame]
                if chunk.size == 0:
                    continue
                ws.send(
                    json.dumps(
                        {
                            "type": "input_audio_buffer.append",
                            "audio": base64.b64encode(chunk.tobytes()).decode("ascii"),
                        }
                    )
                )
            ws.send(json.dumps({"type": "input_audio_buffer.commit"}))

            print("Requesting response after audio input...")
            audio_input_complete_time = time.time()
            ws.send(json.dumps({"type": "response.create", "response": {"modalities": ["audio", "text"]}}))
        else:
            audio_input_complete_time = None
            print("Sending TTS test message...")
            ws.send(
                json.dumps(
                    {
                        "type": "conversation.item.create",
                        "item": {
                            "type": "message",
                            "role": "user",
                            "content": [{"type": "input_text", "text": "Please say: Hello"}],
                        },
                    }
                )
            )
            print("Requesting TTS response...")
            audio_input_complete_time = time.time()
            ws.send(
                json.dumps(
                    {
                        "type": "response.create",
                        "response": {"modalities": ["audio", "text"]},
                    }
                )
            )

        print("Starting to wait for responses...")
        audio_buf = bytearray()
        text_out = []
        response_created = False

        pending_function_calls = {}

        response_count = 0
        expecting_second_response = False

        conversation_transcript = []
        audio_transcript_parts = []
        text_response_parts = []
        first_audio_response_time = None
        first_response_received_time = None
        total_response_time = 0
        web_searches_performed = []

        user_question = ""
        if args.mode == "audio" and selected_episode:
            turns = selected_episode.get("turns", [])
            for turn in turns:
                if turn.get("role") == "user":
                    user_question = turn.get("text_content", "")
                    break

        response_timeout = time.time() + 120
        last_message_time = time.time()
        silence_timeout = 10
        while time.time() < response_timeout:
            try:
                ws.settimeout(5.0)
                msg = ws.recv()
                last_message_time = time.time()

                if isinstance(msg, (bytes, bytearray)):
                    audio_buf.extend(msg)
                    print(f"Received {len(msg)} bytes of binary audio data")
                    continue
                try:
                    data = json.loads(msg)
                except Exception as e:
                    print(f"Failed to parse JSON: {e}, raw message: {msg[:100]}...")
                    continue

                t = data.get("type")
                print(f"Received message type: {t}")

                if t in ("response.audio.delta",):
                    if first_audio_response_time is None and audio_input_complete_time is not None:
                        first_audio_response_time = time.time() - audio_input_complete_time
                        print(f"Time to first AUDIO response: {first_audio_response_time:.3f}s")

                    if first_response_received_time is None:
                        first_response_received_time = time.time() - audio_input_complete_time if audio_input_complete_time else 0
                        print(f"Time to first response (any): {first_response_received_time:.3f}s")

                    b64 = data.get("delta") or data.get("audio") or ""
                    try:
                        audio_buf.extend(base64.b64decode(b64))
                        print(f"Received audio delta: {len(b64)} base64 chars")
                    except Exception as e:
                        print(f"Failed to decode audio delta: {e}")
                elif t in ("response.text.delta", "response.audio_transcript.delta"):
                    if first_response_received_time is None and audio_input_complete_time is not None:
                        first_response_received_time = time.time() - audio_input_complete_time
                        print(f"Time to first response (text): {first_response_received_time:.3f}s")

                    delta = data.get("delta") or ""
                    text_out.append(delta)
                    text_response_parts.append(delta)
                    print(f"Received {'text' if t == 'response.text.delta' else 'audio transcript'} delta: {delta}")
                elif t in ("response.created",):
                    response_created = True
                    response_count += 1
                    print(f"Response {response_count} created, waiting for content...")
                elif t in ("response.function_call_arguments.delta", "response.function_call.delta"):
                    call_id = data.get("call_id")
                    if call_id:
                        if call_id not in pending_function_calls:
                            pending_function_calls[call_id] = {"arguments": ""}
                        pending_function_calls[call_id]["arguments"] += data.get("delta", "")
                        print(f"Function call delta for {call_id}: {data.get('delta', '')}")
                elif t in ("response.function_call_arguments.done", "response.function_call.done"):
                    call_id = data.get("call_id")
                    function_name = data.get("name")
                    arguments_str = data.get("arguments", "")
                    
                    print(f"Function call completed: {function_name} with call_id {call_id}")
                    print(f"Arguments: {arguments_str}")
                    print("Note: Function calls are not supported in this simplified mode")
                elif t in ("response.audio.done", "response.done", "response.error"):
                    print(f"Response {response_count} finished with type: {t}")
                    if not expecting_second_response or response_count >= 2:
                        print(f"Exiting after {response_count} response(s)")
                        break
                    else:
                        print(f"Waiting for second response (current count: {response_count})")
                        response_created = False
                elif t in ("error",):
                    print(f"Error received: {data}")
                    break
                else:
                    print(f"<< {t}: {msg[:200]}...")

            except Exception as e:
                print(f"Error receiving message: {e}")
                if len(audio_buf) > 0 and time.time() - last_message_time > silence_timeout:
                    print(f"No messages for {silence_timeout}s and have audio data - ending conversation")
                    break
                continue

        if time.time() >= response_timeout:
            print("Response timeout reached - ending conversation")

        total_response_time = time.time() - audio_input_complete_time if audio_input_complete_time else 0

        if hasattr(args, 'tts_sr'):
            sample_rate = args.tts_sr
        elif isinstance(args, dict) and 'tts_sr' in args:
            sample_rate = args['tts_sr']
        else:
            sample_rate = 24000

        out_wav = out_dir / "output.wav"
        if audio_buf:
            pcm = np.frombuffer(bytes(audio_buf), dtype=np.int16)
            sf.write(str(out_wav), pcm, sample_rate, subtype="PCM_16")
            print("Wrote:", out_wav)
        else:
            print("No audio received; only text:", "".join(text_out))

        full_text_response = "".join(text_response_parts)

        conversation_transcript.append({
            "role": "user",
            "content": user_question,
            "type": "audio_input"
        })

        if full_text_response or len(audio_buf) > 0:
            conversation_transcript.append({
                "role": "assistant",
                "content": full_text_response,
                "type": "audio_response" if len(audio_buf) > 0 else "text_response",
                "audio_length_bytes": len(audio_buf) if len(audio_buf) > 0 else None
            })

        response_data = {
            "user_question": user_question,
            "assistant_response": full_text_response,
            "conversation_transcript": conversation_transcript,
            "timing": {
                "time_to_first_response": first_response_received_time,
                "time_to_first_audio_response": first_audio_response_time,
                "total_response_time": total_response_time,
                "audio_input_complete_time": audio_input_complete_time
            },
            "audio_info": {
                "input_file": str(input_wav_path) if (hasattr(args, 'mode') and args.mode == "audio") or (isinstance(args, dict) and args.get('mode') == "audio") else None,
                "output_file": str(out_wav) if len(audio_buf) > 0 else None,
                "audio_length_bytes": len(audio_buf),
                "sample_rate": sample_rate
            },
            "web_searches": [],
            "metadata": {
                "mode": args.mode if hasattr(args, 'mode') else (args.get('mode', 'unknown') if isinstance(args, dict) else 'unknown'),
                "episode_id": selected_episode.get("id") if selected_episode else None,
                "context_documents_count": len(context_documents)
            }
        }

        response_json = out_dir / "response.json"
        with open(response_json, "w") as f:
            json.dump(response_data, f, indent=2)
        print(f"Saved structured response data to: {response_json}")

        transcript_txt = out_dir / "conversation.txt"
        with open(transcript_txt, "w") as f:
            f.write("=== CONVERSATION TRANSCRIPT ===\n\n")
            f.write(f"User Question: {user_question}\n\n")
            f.write(f"Assistant Response: {full_text_response}\n\n")
            f.write("=== TIMING ===\n")
            f.write(f"Time to first response (any): {first_response_received_time:.3f}s\n" if first_response_received_time else "Time to first response: N/A\n")
            f.write(f"Time to first audio response: {first_audio_response_time:.3f}s\n" if first_audio_response_time else "Time to first audio response: N/A\n")
            f.write(f"Total response time: {total_response_time:.3f}s\n")
            f.write("\n=== WEB SEARCHES ===\n")
            f.write("No web searches performed (simplified mode)\n")
        print(f"Saved conversation transcript to: {transcript_txt}")

        print("\n=== SUMMARY ===")
        print(f"User question: {user_question[:100]}..." if len(user_question) > 100 else f"User question: {user_question}")
        print(f"Response text: {full_text_response[:100]}..." if len(full_text_response) > 100 else f"Response text: {full_text_response}")
        print(f"Time to first response (any): {first_response_received_time:.3f}s" if first_response_received_time else "Time to first response: N/A")
        print(f"Time to first AUDIO response: {first_audio_response_time:.3f}s" if first_audio_response_time else "Time to first audio response: N/A")
        print(f"Total response time: {total_response_time:.3f}s")
        print(f"Audio output: {len(audio_buf)} bytes")
        print("Web searches: 0 (simplified mode)")

    finally:
        ws.close()
        print("Closed.")


if __name__ == "__main__":
    main()
