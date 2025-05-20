"""
 image_to_markdown.py

 A lightweight library that converts an image (or a single‑page PDF rasterised to an image) to GitHub‑flavoured Markdown using a vision‑capable LLM.

 Usage
 -----
 >>> from image_to_markdown import ImageToMarkdown
 >>> md = ImageToMarkdown(provider="openai").convert(image_path="./diagram.png")
 >>> print(md)

 The class is provider‑agnostic; add more back‑ends by extending `_make_client` and `_dispatch`.
 """

from __future__ import annotations
#
import os
import base64
import logging
from typing import Any, Callable, Dict, List, Literal

import requests
from requests import Session

try:
    # Optional dependency; if unavailable, decorator becomes a no‑op.
    from tenacity import retry, wait_exponential, stop_after_attempt  # type: ignore
except ImportError:  # pragma: no cover
    def retry(*_a: Any, **_kw: Any):  # type: ignore
        return lambda f: f  # noqa: E501

# ---------------------------------------------------------------------------
# Configuration & constants
# ---------------------------------------------------------------------------

LOGGER = logging.getLogger("image_to_markdown")
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)

# The default transcription prompt (single source of truth).
TRANSCRIBE_PROMPT = """**Role:** You are a transcription assistant. When given an image or PDF, reproduce every visible element as GitHub‑flavored Markdown for direct ingestion into documentation systems.

**OUTPUT RULES**
1. **Return only the Markdown text.** Never add explanations, comments, or fences (``` or ~~~).
2. **Preserve document structure in order:**
* Headings `#` – `######`
* Paragraphs (blank line between)
* Lists — unordered (`-`, `*`, `+`) and ordered (`1.`), with nesting via 2‑space indents
* Tables (pipe syntax); place any caption in *italics* directly below the table
* Blockquotes (`>`)
* Code blocks (```), inline code (`\``) — keep original whitespace
* Horizontal rules (`---`)
3. **Preserve formatting:** **bold**, *italic*, ***bold‑italic***, ~~strikethrough~~.
4. **Links:** `[text](href)` — keep original URLs.
5. **Images:** `![concise alt text](placeholder)`; if a caption exists, put it in *italics* on the next line.
6. **Math:** inline `$…$`, display `$$ … $$`.
7. **Header/footer or page numbers:** keep exactly where they appear; separate pages with `---`.
8. **Copy verbatim.** Do not correct typos, re‑flow lines, or infer missing content.
9. If something is unreadable, insert `[UNREADABLE]`.

*Return the Markdown immediately and nothing else.*"""

_ENV_VARS: Dict[str, str] = {
    "together": "TOGETHER_API_KEY",
    "groq": "GROQ_API_KEY",
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}

# ---------------------------------------------------------------------------
# Helper functions (public‑agnostic)
# ---------------------------------------------------------------------------

def _require_api_key(provider: str, api_key: str | None) -> str:
    """Return a usable API key, falling back to environment variables."""
    env_name = _ENV_VARS.get(provider, "")
    key = api_key or os.getenv(env_name)
    if key:
        return key
    raise ValueError(
        f"API key required for provider '{provider}'. Set env var '{env_name}' or pass api_key="
    )

def _encode_image(path: str) -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    LOGGER.info("Encoding local image '%s' → base64", path)
    with open(path, "rb") as fh:
        return base64.b64encode(fh.read()).decode()

_SESSION: Session = requests.Session()

def _download_to_base64(url: str) -> str:
    LOGGER.info("Downloading '%s'", url)
    resp = _SESSION.get(url, timeout=15)
    resp.raise_for_status()
    return base64.b64encode(resp.content).decode()

# ---------------------------------------------------------------------------
# Main public class
# ---------------------------------------------------------------------------

class ImageToMarkdown:
    """Convert an image (or PDF rasterised to an image) into Markdown using an LLM."""

    def __init__(
        self,
        provider: Literal["together", "groq", "openai", "anthropic"],
        api_key: str | None = None,
        *,
        default_temperature: float = 0.0,
        request_timeout: int = 90,
    ) -> None:
        self.provider = provider.lower()
        self.api_key = _require_api_key(self.provider, api_key)
        self.client = self._make_client()
        self.default_temperature = float(default_temperature)
        self.timeout = int(request_timeout)
        LOGGER.info("Provider '%s' initialised", self.provider)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def convert(
        self,
        *,
        image_path: str | None = None,
        image_url: str | None = None,
        prompt: str | None = None,
        temperature: float | None = None,
        json_output: bool = False,
    ) -> str | dict[str, str]:
        """Convert *one* image to Markdown.

        Parameters
        ----------
        image_path / image_url
            Exactly one source must be provided.
        prompt
            Override the default system prompt.
        temperature
            Overrides the instance default.
        json_output
            Return a mapping `{ "markdown": md }` instead of a bare string.
        """
        if (image_path is None) == (image_url is None):
            raise ValueError("Exactly one of image_path or image_url must be provided.")

        prompt = prompt or TRANSCRIBE_PROMPT
        base64_img = _encode_image(image_path) if image_path else _download_to_base64(image_url)  # type: ignore[arg-type]
        messages = self._build_messages(base64_img, prompt)
        md = self._dispatch(messages, temperature or self.default_temperature)
        return {"markdown": md} if json_output else md

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _make_client(self):  # noqa: C901 (complexity)
        if self.provider == "together":
            from together import Together  # type: ignore

            return Together(api_key=self.api_key)
        if self.provider == "groq":
            from groq import Groq  # type: ignore

            return Groq(api_key=self.api_key)
        if self.provider == "openai":
            import openai  # type: ignore

            return openai.OpenAI(api_key=self.api_key)
        if self.provider == "anthropic":
            import anthropic  # type: ignore

            return anthropic.Anthropic(api_key=self.api_key)
        raise ValueError(f"Unsupported provider '{self.provider}'.")

    @staticmethod
    def _build_messages(base64_image: str, prompt: str) -> List[Dict[str, Any]]:  # type: ignore[name‑defined]
        return [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                ],
            }
        ]

    # ------------------------------ dispatch ------------------------------

    def _dispatch(self, messages: List[Dict[str, Any]], temperature: float) -> str:  # type: ignore[name‑defined]
        if self.provider == "together":
            return self._with_together(messages, temperature)
        if self.provider == "groq":
            return self._with_groq(messages, temperature)
        if self.provider == "openai":
            return self._with_openai(messages, temperature)
        if self.provider == "anthropic":
            return self._with_anthropic(messages, temperature)
        raise RuntimeError("Dispatch reached unsupported provider.")

    # ------------------------- provider implementations -------------------

    @retry(wait=wait_exponential(multiplier=1, min=4, max=30), stop=stop_after_attempt(5), reraise=True)
    def _with_together(self, messages: List[Dict[str, Any]], temperature: float) -> str:  # type: ignore[name‑defined]
        LOGGER.info("→ Together | llama‑3‑vision")
        resp = self.client.chat.completions.create(
            model="meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo",
            messages=messages,
            temperature=temperature,
            timeout=self.timeout,
        )
        return resp.choices[0].message.content.strip()

    @retry(wait=wait_exponential(multiplier=1, min=4, max=30), stop=stop_after_attempt(5), reraise=True)
    def _with_groq(self, messages: List[Dict[str, Any]], temperature: float) -> str:  # type: ignore[name‑defined]
        LOGGER.info("→ Groq | llama‑4‑scout‑17b")
        resp = self.client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=messages,
            temperature=temperature,
            timeout=self.timeout,
        )
        return resp.choices[0].message.content.strip()

    @retry(wait=wait_exponential(multiplier=1, min=4, max=30), stop=stop_after_attempt(5), reraise=True)
    def _with_openai(self, messages: List[Dict[str, Any]], temperature: float) -> str:  # type: ignore[name‑defined]
        LOGGER.info("→ OpenAI | gpt‑4o")
        resp = self.client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=temperature,
            timeout=self.timeout,
        )
        return resp.choices[0].message.content.strip()

    @retry(wait=wait_exponential(multiplier=1, min=4, max=30), stop=stop_after_attempt(5), reraise=True)
    def _with_anthropic(self, messages: List[Dict[str, Any]], temperature: float) -> str:  # type: ignore[name‑defined]
        LOGGER.info("→ Anthropic | claude‑3‑sonnet")
        # anthropic has a slightly different payload shape – rebuild content.
        img_url = messages[0]["content"][1]["image_url"]["url"]  # type: ignore[index]
        prompt_text = messages[0]["content"][0]["text"]  # type: ignore[index]
        img_data = img_url.split(",", 1)[1]
        content = [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": img_data,
                },
            },
            {"type": "text", "text": prompt_text},
        ]
        resp = self.client.messages.create(
            model="claude-3-7-sonnet-20250219",
            temperature=temperature,
            max_tokens=4096,
            timeout=self.timeout,
            messages=[{"role": "user", "content": content}],
        )
        return resp.content[0].text.strip()  # type: ignore[index]
