import json
import urllib.request
import urllib.error


class LLMError(Exception):
    pass


class LLMClient:
    """
    Thin client for a local OpenAI-compatible /v1/chat/completions endpoint
    (this is what Ollama, vLLM, and llama.cpp-server all expose), so no
    external API is called at inference time — satisfies the competition's
    "self-host only, no external API" rule.

    Uses only the standard library (no `requests`/`openai` package) so this
    doesn't add a dependency the team's environment might not have.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434/v1",
        model: str = "qwen3:8b",
        timeout: int = 300,
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.temperature = temperature
        self.max_tokens = max_tokens

    def chat(self, system_prompt: str, user_prompt: str) -> str:

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            # Qwen3's "thinking mode" free-reasons before answering, which
            # both slows down a batch job across ~100 docs and makes JSON
            # parsing less reliable. We want direct JSON, not a reasoning
            # trace. Ollama/vLLM read this via chat_template_kwargs.
            "chat_template_kwargs": {"enable_thinking": False},
        }

        data = json.dumps(payload).encode("utf-8")

        request = urllib.request.Request(
            url=f"{self.base_url}/chat/completions",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = json.loads(response.read().decode("utf-8"))

        except urllib.error.URLError as e:
            raise LLMError(
                f"Could not reach LLM endpoint at {self.base_url}. "
                f"Is the server running (e.g. `ollama serve`)? Original error: {e}"
            ) from e

        try:
            return body["choices"][0]["message"]["content"]

        except (KeyError, IndexError) as e:
            raise LLMError(
                f"Unexpected response shape from LLM endpoint: {body}"
            ) from e


class FakeLLMClient:
    """
    Drop-in replacement for LLMClient used in tests/dev, since we can't run
    an actual model in this environment. Returns pre-scripted responses
    keyed by a substring of the user prompt, so pipeline logic (parsing,
    span location, retrieval, offset mapping, export) can be fully built
    and tested before a real model is wired in.
    """

    def __init__(self, scripted_responses: dict[str, str]):
        self.scripted_responses = scripted_responses
        self.calls = []

    def chat(self, system_prompt: str, user_prompt: str) -> str:

        self.calls.append(user_prompt)

        for key, response in self.scripted_responses.items():
            if key in user_prompt:
                return response

        return "[]"
