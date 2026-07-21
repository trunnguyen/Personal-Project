import json
import time
import socket
import urllib.request
import urllib.error


RESPONSE_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "text": {"type": "string"},
            "type": {
                "type": "string",
                "enum": ["TRIỆU_CHỨNG", "TÊN_XÉT_NGHIỆM", "KẾT_QUẢ_XÉT_NGHIỆM", "CHẨN_ĐOÁN", "THUỐC"],
            },
            "assertions": {
                "type": "array",
                "items": {"type": "string", "enum": ["isNegated", "isFamily", "isHistorical"]},
            },
            "lookup_term": {"type": ["string", "null"]},
        },
        "required": ["text", "type", "assertions", "lookup_term"],
    },
}

class LLMError(Exception):
    pass


class LLMClient:
    """
    Uses Ollama's /api/generate endpoint with raw=true — we build the exact
    ChatML prompt ourselves instead of letting Ollama's chat template do it.

    Why: two different ways of asking the model not to "think" (the API
    "think": false parameter, then the in-prompt "/no_think" directive)
    both failed — the model kept producing full prose reasoning traces
    regardless. That means the chat template in this Ollama build isn't
    correctly wiring up Qwen3's thinking toggle at all, for either
    mechanism. The fix is to stop relying on the template: we write the
    ChatML prompt by hand and pre-fill the assistant's turn with an
    already-CLOSED empty <think></think> block. The model has no way to
    generate reasoning into a thinking block its own output already shows
    as finished, so it has to continue straight into the real answer. This
    is the standard, template-independent workaround for hybrid-reasoning
    models like Qwen3 when the serving layer's own thinking-control flags
    aren't reliable.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "qwen3:4b",
        timeout: int = 600,
        temperature: float = 0.0,
        max_tokens: int = 2500,
        num_ctx: int = 8192,
        keep_alive: str = "30m",
        verbose: bool = True,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.num_ctx = num_ctx
        self.keep_alive = keep_alive
        self.verbose = verbose

    @staticmethod
    def _build_raw_prompt(system_prompt: str, user_prompt: str) -> str:

        return (
            f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n<think>\n\n</think>\n\n"
        )

    def chat(self, system_prompt: str, user_prompt: str) -> str:

        raw_prompt = self._build_raw_prompt(system_prompt, user_prompt)

        payload = {
            "model": self.model,
            "prompt": raw_prompt,
            "raw": True,
            "stream": False,
            "format": RESPONSE_SCHEMA,
            "keep_alive": self.keep_alive,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
                "num_ctx": self.num_ctx,
                "stop": ["<|im_end|>", "<|im_start|>"],
            },
        }

        data = json.dumps(payload).encode("utf-8")

        request = urllib.request.Request(
            url=f"{self.base_url}/api/generate",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        if self.verbose:
            print(f"  [llm] model={self.model} (raw+prefilled) sending request ({len(data)} bytes)...", end="", flush=True)

        t0 = time.time()

        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = json.loads(response.read().decode("utf-8"))

        except (socket.timeout, TimeoutError) as e:
            raise LLMError(
                f"LLM request timed out after {self.timeout}s even with the "
                f"prefilled empty think block. If this still happens, "
                f"generation itself (not thinking) is the bottleneck now."
            ) from e

        except urllib.error.URLError as e:
            raise LLMError(
                f"Could not reach LLM endpoint at {self.base_url}. "
                f"Is the server running (e.g. `ollama serve`)? Original error: {e}"
            ) from e

        finally:
            if self.verbose:
                print(f" done in {time.time() - t0:.1f}s")

        # /api/generate response shape is {"response": "...", "done": true, ...}
        # — different again from both /v1/chat/completions and /api/chat.
        try:
            return body["response"]

        except KeyError as e:
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