"""
Genie (LLM backend) adapters for chonkie.SlumberChunker, used by
`agentic_chunk` in chunking_strategies.py.

SlumberChunker's chunking ALGORITHM is Chonkie's own, tested implementation
— a "genie" is just the pluggable LLM backend it calls internally to decide
split points. Chonkie ships OpenAI/Gemini/Groq/Cerebras/Azure genies out of
the box; Anthropic and Ollama aren't among them, so this file adds two
minimal adapters implementing BaseGenie's documented `generate()` interface
— the officially supported extension point, not a reimplementation of any
chunking logic.
"""

import os

from chonkie.genie import BaseGenie, OpenAIGenie


class AnthropicGenie(BaseGenie):
    def __init__(self, model: str | None = None):
        import anthropic

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set. Add it to your .env file.")
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model or os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-5")

    def generate(self, prompt: str) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text


class OllamaGenie(BaseGenie):
    def __init__(self, model: str | None = None):
        self.url = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
        self.model = model or os.environ.get("OLLAMA_MODEL", "llama3.1")

    def generate(self, prompt: str) -> str:
        import requests

        response = requests.post(
            self.url,
            json={"model": self.model, "prompt": prompt, "stream": False},
            timeout=120,
        )
        response.raise_for_status()
        return response.json()["response"]


def get_genie(provider: str | None = None) -> BaseGenie:
    """Return a Chonkie-compatible genie for whichever provider LLM_PROVIDER
    (or `provider`) names: 'anthropic' | 'openai' | 'ollama'."""
    provider = (provider or os.environ.get("LLM_PROVIDER", "anthropic")).lower()

    if provider == "anthropic":
        return AnthropicGenie()
    if provider == "openai":
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set. Add it to your .env file.")
        return OpenAIGenie(model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"), api_key=api_key)
    if provider == "ollama":
        return OllamaGenie()

    raise ValueError(f"Unknown LLM_PROVIDER '{provider}'. Use 'anthropic', 'openai', or 'ollama'.")
