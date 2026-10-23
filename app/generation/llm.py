"""Thin LLM wrapper. Defaults to Ollama running locally, which costs nothing.

Keeping generation behind this one function means you can swap in a hosted API
later by changing this file only -- nothing else in the codebase imports a
provider SDK.
"""

from __future__ import annotations

import httpx

from app.config import settings


class LLMUnavailable(RuntimeError):
    pass


def complete(system: str, user: str, temperature: float = 0.0, timeout: float = 120.0) -> str:
    url = f"{settings.llm_base_url.rstrip('/')}/api/chat"
    payload = {
        "model": settings.llm_model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "options": {"temperature": temperature},
    }
    try:
        resp = httpx.post(url, json=payload, timeout=timeout)
        resp.raise_for_status()
    except httpx.ConnectError as exc:
        raise LLMUnavailable(
            "Could not reach Ollama at "
            f"{settings.llm_base_url}. Is it running? Try: ollama serve"
        ) from exc
    except httpx.HTTPStatusError as exc:
        raise LLMUnavailable(
            f"Ollama returned {exc.response.status_code}. "
            f"Have you pulled the model?  ollama pull {settings.llm_model}"
        ) from exc
    return resp.json()["message"]["content"].strip()


def health() -> dict:
    try:
        resp = httpx.get(f"{settings.llm_base_url.rstrip('/')}/api/tags", timeout=5)
        models = [m["name"] for m in resp.json().get("models", [])]
        return {"ok": True, "configured_model": settings.llm_model, "available": models}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
