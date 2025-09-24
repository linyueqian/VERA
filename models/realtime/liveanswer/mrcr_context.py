"""
MRCR context handling for LiveAnswer.
Based on azure_gpt_realtime approach.
"""

import json
from typing import List, Dict, Any, Optional
from pathlib import Path


def parse_mrcr_context(context: str) -> List[Dict[str, str]]:
    """Parse MRCR context document into conversation messages"""
    messages = []

    # Split by User: and Assistant: markers
    lines = context.split('\n')
    current_role = None
    current_content = []

    for line in lines:
        if line.startswith('User:'):
            if current_role and current_content:
                messages.append({"role": current_role, "content": '\n'.join(current_content).strip()})
            current_role = "user"
            current_content = [line[5:].strip()]  # Remove 'User:' prefix
        elif line.startswith('Assistant:'):
            if current_role and current_content:
                messages.append({"role": current_role, "content": '\n'.join(current_content).strip()})
            current_role = "assistant"
            current_content = [line[10:].strip()]  # Remove 'Assistant:' prefix
        else:
            if current_content is not None:
                current_content.append(line)

    # Add the last message
    if current_role and current_content:
        messages.append({"role": current_role, "content": '\n'.join(current_content).strip()})

    return messages


def load_context_documents_from_audio_file(audio_file_path: str) -> List[Dict[str, Any]]:
    """
    Load context documents from episode JSON based on audio file path.
    Follows the same pattern as azure_gpt_realtime.
    """
    audio_path = Path(audio_file_path)

    # Try to find corresponding episode JSON
    episode_json_candidates = [
        # Same directory, replace .wav with _episode.json
        audio_path.parent / f"{audio_path.stem}_episode.json",
        # test_voice_episodes directory structure
        audio_path.parent.parent / "episodes" / f"{audio_path.stem}_episode.json",
        # Current directory test_voice_episodes
        Path.cwd() / "test_voice_episodes" / "episodes" / f"{audio_path.stem}_episode.json",
    ]

    # Add test_voice_episodes direct files based on audio file type
    audio_stem = audio_path.stem.lower()
    if "mrcr" in audio_stem:
        episode_json_candidates.append(Path.cwd() / "test_voice_episodes" / "test_mrcr_episode.json")
    elif "browsecomp" in audio_stem:
        episode_json_candidates.append(Path.cwd() / "test_voice_episodes" / "test_browsecomp_episode.json")
    elif "aime" in audio_stem:
        episode_json_candidates.append(Path.cwd() / "test_voice_episodes" / "test_aime_episode.json")

    episode_json = None
    print(f"!!!MRCR: Looking for episode JSON for audio file: {audio_file_path}")
    for candidate in episode_json_candidates:
        print(f"!!!MRCR: Checking candidate: {candidate}")
        if candidate.exists():
            episode_json = candidate
            print(f"!!!MRCR: Found episode JSON: {episode_json}")
            break

    if not episode_json:
        print(f"!!!MRCR: No episode JSON found for audio file: {audio_file_path}")
        print(f"!!!MRCR: Tried candidates: {episode_json_candidates}")
        return []

    try:
        episode_data = json.loads(episode_json.read_text())
        if episode_data.get("episodes"):
            first_episode = episode_data["episodes"][0]
            context_documents = first_episode.get("context_documents", [])
            print(f"!!!MRCR: Found {len(context_documents)} context documents from {episode_json}")
            if context_documents:
                print(f"!!!MRCR: First context document has {len(context_documents[0].get('content', ''))} characters")
            return context_documents
    except Exception as e:
        print(f"Error loading context documents from {episode_json}: {e}")

    return []


def inject_mrcr_context_into_messages(
    messages: List[Dict[str, str]],
    context_documents: List[Dict[str, Any]],
    episode_id: Optional[str] = None
) -> List[Dict[str, str]]:
    """
    Inject MRCR context documents into message history.
    Based on azure_gpt_realtime approach.
    """
    if not context_documents:
        return messages

    print(f"Injecting {len(context_documents)} context documents...")

    # Determine if this is MRCR
    is_mrcr = False
    if episode_id:
        is_mrcr = "mrcr" in episode_id.lower()

    # Insert context documents before the conversation
    context_messages = []

    for i, doc in enumerate(context_documents):
        content = doc.get("content", "")
        if content and is_mrcr:
            # For MRCR, inject the full conversation as a system message
            print(f"Injecting MRCR conversation context from document {i+1}")
            parsed_messages = parse_mrcr_context(content)

            # Convert conversation to system context
            context_text = "Previous conversation:\n\n"
            for msg in parsed_messages:
                role = msg["role"].title()
                context_text += f"{role}: {msg['content']}\n\n"

            context_messages.append({
                "role": "system",
                "content": f"You have access to the following conversation history:\n\n{context_text.strip()}"
            })
        elif content:
            # For non-MRCR, add as single assistant message
            context_messages.append({
                "role": "assistant",
                "content": f"Previous context: {content}"
            })

    print(f"Context injection complete.")

    # Return context messages + original messages
    return context_messages + messages


def is_mrcr_episode(audio_file_path: str) -> bool:
    """Check if this is an MRCR episode based on file path."""
    path_str = str(audio_file_path).lower()
    return "mrcr" in path_str