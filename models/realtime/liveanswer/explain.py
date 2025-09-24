import json
from typing import Optional, List

import requests  # type: ignore

from .utils import _env


class ExplainSynthesizer:
    """
    Incrementally produces spoken explanation text by calling Groq for continuation
    (assistant prefill style).
    """

    def __init__(self, request: str):
        self.request: str = request
        self.all_thought: List[Optional[str]] = []
        self.spoken_explanation: str = ""
        self.finished: bool = False

        # Track consecutive dummy explanations to prevent infinite loops
        # Give generous tolerance since thinking model may be slow
        self._consecutive_dummy_count: int = 0
        self._max_consecutive_dummy: int = 10  # Allow up to 10 dummy responses (thinking time)

        self._last_groq_messages: List[dict] = []
        self._last_groq_response: Optional[str] = None

    def push_thought(self, s: Optional[str]) -> None:
        self.all_thought.append(s)

    def _groq_chat_completion(self, messages: List[dict], max_tokens: int) -> str:
        if self._last_groq_messages == messages:
            return self._last_groq_response
        self._last_groq_messages = messages
        self._last_groq_response = self.__groq_chat_completion(messages, max_tokens)
        return self._last_groq_response

    def __groq_chat_completion(self, messages: List[dict], max_tokens: int) -> str:
        api_key = _env("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY missing")

        groq_base = _env("GROQ_ENDPOINT", "https://api.groq.com/openai/v1")
        url = f"{groq_base.rstrip('/')}" \
              f"/chat/completions"
        model = _env("GROQ_MODEL", "llama-3.3-70b-versatile")
        print(f"!!!Groq API: model={model}, max_tokens={max_tokens}")

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        # Increase temperature for Template3 to encourage more generation
        temperature = 0.9 if max_tokens > 1000 else 0.7

        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_completion_tokens": max_tokens,
        }

        # Increase timeout for longer responses
        timeout = 60 if max_tokens > 1000 else 30
        try:
            resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"].strip()
            finish_reason = data["choices"][0].get("finish_reason", "unknown")
            print(f"!!!Groq API response: finish_reason={finish_reason}, content_length={len(content)}")
            return content
        except Exception as e:
            print(f"!!!Groq API error: {e}")
            raise

    def pop_more_explanation(self, max_token: int = 32) -> Optional[str]:
        if self.finished:
            print(f"!!!pop_more_explanation: Already finished, returning None")
            return None

        has_any_thought = len(self.all_thought) > 0
        last_thought = self.all_thought[-1] if has_any_thought else None
        non_none_thoughts = [t for t in self.all_thought if t is not None]
        all_thought_text = (" ".join(non_none_thoughts)).strip()

        print(f"!!!pop_more_explanation: has_any_thought={has_any_thought}, last_thought={'None' if last_thought is None else 'Some'}, len(all_thought)={len(self.all_thought)}")

        # Use different system prompt for Template3 (finalization) to encourage comprehensive response
        if last_thought is None and has_any_thought:
            # Template3: Need comprehensive final explanation
            system_prompt = (
                "You are a thorough, clear explainer providing a complete final explanation. "
                "Generate natural spoken-style text that fully explains the solution. "
                "Write exactly as if spoken aloud. Avoid symbols, equations, code fences, or special characters; "
                "use plain words instead. Express relations in words (e.g., x=y -> 'x equals y'). "
                "Provide a COMPLETE and COMPREHENSIVE explanation. Do not be too concise - be thorough."
            )
        else:
            # Template1 and Template2: Regular concise style
            system_prompt = (
                "You are a concise, clear explainer. Generate natural spoken-style text. "
                "Avoid lists unless necessary. Keep continuity with the prior assistant text. "
                "Write exactly as if spoken aloud. Avoid symbols, equations, code fences, or special characters; "
                "use plain words instead. Express relations in words (e.g., x=y -> 'x equals y'). Keep punctuation minimal and natural."
                "Use short sentences and phrases if possible. Avoid long sentences and paragraphs."
            )

        assistant_prefill = self.spoken_explanation.strip()
        final_answer = False
        if not has_any_thought:
            # Template1: no solver thoughts yet → confirm + typically how to proceed
            print(f"!!!Using Template1: No solver thoughts yet")
            user_template = (
                "Begin the spoken explanation. Start with a very brief rephrase of the user's "
                "request (one short sentence) to confirm understanding, then briefly state what you would "
                "typically do to approach it, and continue naturally. Do not include any disclaimers about inability or limitations. "
                "Avoid lists unless necessary; keep it concise and fluid.\n\n"
                f"User request: {self.request}"
            )
        elif last_thought is None:
            # Template3: finalization
            max_token = 2048  # Increase token budget for comprehensive explanation
            final_answer = True
            print(f"!!!Using Template3: Finalization with max_token={max_token}")
            print(f"!!!All solver thoughts collected: {len(non_none_thoughts)} thoughts, {len(all_thought_text)} chars")
            user_template = (
                "The conversation is concluding. Please provide a COMPREHENSIVE and DETAILED final explanation that:\n"
                "1. Fully explains the solution approach and reasoning\n"
                "2. Clearly states the final answer\n"
                "3. Explains WHY this answer is correct\n"
                "4. Should be at least 3-4 paragraphs long for completeness\n"
                "Continue from where you left off, but ensure the explanation is thorough and complete. "
                "Do not stop until you have fully explained the solution.\n\n"
                + (f"All solver thoughts so far: {all_thought_text}\n\n" if all_thought_text else "")
                + f"User request: {self.request}"
            )
        else:
            # Template2: ongoing with accumulated thoughts
            print(f"!!!Using Template2: Ongoing with {len(non_none_thoughts)} thoughts")
            user_template = (
                "Continue the spoken explanation naturally. Keep it fluid and avoid abrupt topic jumps. "
                "Be sure to include all latest updates from the accumulated reasoning (all_thought_text) as quickly as possible.\n\n"
                + (f"Use the overall reasoning so far: {all_thought_text}\n\n" if all_thought_text else "")
                + f"User request: {self.request}"
            )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_template},
        ]

        # Use TTS-friendly filler text instead of newlines
        dumb_explanation = "I'm still thinking about this problem. Let me work through the details."
        def remove_dumb_words(s: str) -> str:
            return s.replace(dumb_explanation, "")

        if assistant_prefill:
            messages.append({"role": "assistant", "content": remove_dumb_words(assistant_prefill)})

        print(f"!!!max_token: {max_token}")
        print(f"!!!Calling Groq with messages count: {len(messages)}")
        chunk = self._groq_chat_completion(messages=messages, max_tokens=max_token)
        print(f"!!!Groq returned chunk length: {len(chunk)} chars")
        if has_any_thought and last_thought is None:
            print(f"!!!Setting finished=True (Template3 completed)")
            self.finished = True
        if not chunk.strip():
            if not final_answer:
                self._consecutive_dummy_count += 1
                print(f"!!!Empty chunk returned ({self._consecutive_dummy_count}/{self._max_consecutive_dummy}), using dumb explanation")

                # Only stop if we've exceeded the maximum dummy responses
                if self._consecutive_dummy_count >= self._max_consecutive_dummy:
                    print(f"!!!Too many consecutive dummy responses ({self._max_consecutive_dummy}), ending conversation")
                    self.finished = True
                    return None

                chunk = dumb_explanation
            else:
                # For final answer (Template3), if we get empty response, just end cleanly
                print(f"!!!Empty chunk in final answer - ending conversation")
                self.finished = True
                return None  # Signal end of conversation
        else:
            # Reset counter when we get a real response
            if self._consecutive_dummy_count > 0:
                print(f"!!!Got real response, resetting dummy counter from {self._consecutive_dummy_count}")
                self._consecutive_dummy_count = 0

        print(f"!!!chunk: {chunk[:200]}..." if len(chunk) > 200 else f"!!!chunk: {chunk}")

        # if self.spoken_explanation and not self.spoken_explanation.endswith(" "):
        #     self.spoken_explanation += " "
        self.spoken_explanation += chunk
        return chunk
