"""
Pluggable LLM providers for the generation step.

Pick a provider with the LLM_PROVIDER env var: "anthropic" | "openai" | "ollama".
Anthropic and OpenAI need an API key (read from .env, never logged or printed).
Ollama runs fully locally and needs no key at all.
"""

import os


def generate(prompt: str, provider: str | None = None, model: str | None = None) -> str:
    """Route to the configured LLM provider and return its text response."""
    provider = (provider or os.environ.get("LLM_PROVIDER", "ollama")).lower()

    if provider == "anthropic":
        return _generate_anthropic(prompt, model or os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5"))
    if provider == "openai":
        return _generate_openai(prompt, model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini"))
    if provider == "ollama":
        return _generate_ollama(prompt, model or os.environ.get("OLLAMA_MODEL", "qwen3:4b"))

    raise ValueError(f"Unknown LLM_PROVIDER '{provider}'. Use 'anthropic', 'openai', or 'ollama'.")


def _generate_anthropic(prompt: str, model: str) -> str:
    import anthropic

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set. Add it to your .env file.")
    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(model=model, max_tokens=1024, messages=[{"role": "user", "content": prompt}])
    return response.content[0].text


def _generate_openai(prompt: str, model: str) -> str:
    from openai import OpenAI

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Add it to your .env file.")
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(model=model, messages=[{"role": "user", "content": prompt}])
    return response.choices[0].message.content


def _generate_ollama(prompt: str, model: str) -> str:
    import requests

    url = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
    response = requests.post(url, json={"model": model, "prompt": prompt, "stream": False}, timeout=180)
    response.raise_for_status()
    return response.json()["response"]
