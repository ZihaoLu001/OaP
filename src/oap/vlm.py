"""Shared client for the system's current vision-language requests.

Program synthesis and scene-prior extraction use :func:`ask_vlm` so model,
credential, timeout, and image encoding policy have one implementation.
Success verification does not call a VLM: only the TaskProgram's
measured ``terminal`` predicates can certify success.

Model choice is one env var (``OAP_VLM_MODEL``) with one default, so
"which model did this episode use" has a single answer and the ablation is a
single knob. Adding OpenAI did not add a second knob: the provider is READ OFF
the model id (``claude-*`` -> Anthropic, ``gpt-*``/``o*`` -> OpenAI), so an
episode whose artifact says ``gpt-...`` cannot have been served by Anthropic.
``OAP_VLM_PROVIDER`` exists only to override an id this module cannot
route, and it must still name a provider that exists.

The key is the user's (``ANTHROPIC_API_KEY`` or ``OPENAI_API_KEY``, whichever
the routed provider needs); this module never creates, copies or stores one.
"""
from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import Any

__all__ = ["ask_vlm", "vlm_model_from_env", "provider_for_model",
           "VLM_MODEL_ENV", "VLM_TIMEOUT_ENV", "VLM_PROVIDER_ENV"]

VLM_MODEL_ENV = "OAP_VLM_MODEL"
VLM_TIMEOUT_ENV = "OAP_VLM_TIMEOUT"
VLM_PROVIDER_ENV = "OAP_VLM_PROVIDER"
_LEGACY_MODEL_ENV = "OAP_ANTHROPIC_MODEL"   # pre-unification spelling
_DEFAULT_MODEL = "claude-opus-4-8"

_PROVIDERS = ("anthropic", "openai")
_KEY_ENV = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY"}


def vlm_model_from_env() -> str:
    """Resolve the one model id every consumer uses."""
    return (os.environ.get(VLM_MODEL_ENV)
            or os.environ.get(_LEGACY_MODEL_ENV)
            or _DEFAULT_MODEL)


def provider_for_model(model_id: str) -> str:
    """Which API serves this model id.

    Refuses an id it cannot route rather than defaulting to one, because a
    silent default would send the prompt to a provider the artifact does not
    name.
    """
    override = (os.environ.get(VLM_PROVIDER_ENV) or "").strip().lower()
    if override:
        if override not in _PROVIDERS:
            raise ValueError(
                f"{VLM_PROVIDER_ENV}={override!r} is not a provider; "
                f"expected one of {_PROVIDERS}")
        return override
    name = str(model_id).strip().lower()
    if name.startswith("claude"):
        return "anthropic"
    if name.startswith("gpt") or (name.startswith("o") and name[1:2].isdigit()):
        return "openai"
    raise ValueError(
        f"cannot route model id {model_id!r} to a provider; set "
        f"{VLM_PROVIDER_ENV} to one of {_PROVIDERS} if this is intentional")


def _require_key(provider: str) -> str:
    env = _KEY_ENV[provider]
    key = os.environ.get(env, "").strip()
    if not key:
        raise RuntimeError(
            f"{env} is not set. Export it (e.g. from a chmod-600 env file) "
            f"before running a stage that consults the VLM.")
    return key


def _image_bytes_and_media(image: Path) -> tuple[bytes, str]:
    data = Path(image).read_bytes()
    suffix = Path(image).suffix.lower()
    media = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
             ".webp": "image/webp", ".gif": "image/gif"}.get(suffix, "image/png")
    return data, media


def _image_block(image: Path) -> dict[str, Any]:
    data, media = _image_bytes_and_media(image)
    return {"type": "image",
            "source": {"type": "base64", "media_type": media,
                       "data": base64.b64encode(data).decode("ascii")}}


def _timeout() -> float:
    return float(os.environ.get(VLM_TIMEOUT_ENV, "90"))


def _ask_anthropic(*, system: str, user: str, image: Path | None,
                   image_b64: str | None, max_tokens: int,
                   model_id: str) -> tuple[str, str]:
    try:
        import anthropic
    except Exception as e:  # pragma: no cover - import guard
        raise RuntimeError(f"`pip install anthropic` to consult the VLM ({e})")

    client = anthropic.Anthropic(max_retries=0, timeout=_timeout())

    blocks: list[Any] = []
    if image is not None:
        blocks.append(_image_block(image))
    elif image_b64 is not None:
        blocks.append({"type": "image",
                       "source": {"type": "base64", "media_type": "image/png",
                                  "data": image_b64}})
    content: Any = user if not blocks else [*blocks, {"type": "text", "text": user}]

    msg = client.messages.create(model=model_id, max_tokens=int(max_tokens),
                                 system=system,
                                 messages=[{"role": "user", "content": content}])
    block = msg.content[0]
    if not isinstance(block, anthropic.types.TextBlock):
        raise RuntimeError(
            f"the VLM returned a non-text first block ({type(block).__name__})")
    return block.text, str(getattr(msg, "model", "") or model_id)


def _ask_openai(*, system: str, user: str, image: Path | None,
                image_b64: str | None, max_tokens: int,
                model_id: str) -> tuple[str, str]:
    try:
        import openai
    except Exception as e:  # pragma: no cover - import guard
        raise RuntimeError(f"`pip install openai` to consult the VLM ({e})")

    client = openai.OpenAI(max_retries=0, timeout=_timeout())

    parts: list[dict[str, Any]] = []
    if image is not None:
        data, media = _image_bytes_and_media(image)
        b64 = base64.b64encode(data).decode("ascii")
        parts.append({"type": "image_url",
                      "image_url": {"url": f"data:{media};base64,{b64}"}})
    elif image_b64 is not None:
        parts.append({"type": "image_url",
                      "image_url": {"url": f"data:image/png;base64,{image_b64}"}})
    # Image BEFORE text, matching the Anthropic branch, so a prompt that refers
    # back to "the image above" reads in the same order on both providers.
    parts.append({"type": "text", "text": user})

    # OpenAI has no `system=` kwarg; dropping it would silently discard the
    # instruction that makes the reply JSON-only.
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": parts}]

    # Reasoning-era models reject `max_tokens` outright, so the cap goes under
    # `max_completion_tokens`. Both names mean the same thing to the models that
    # still accept the old one.
    resp = client.chat.completions.create(
        model=model_id, messages=messages,
        max_completion_tokens=int(max_tokens))

    text = resp.choices[0].message.content
    if not text:
        raise RuntimeError(
            "the VLM returned no text (empty or refused completion); a blank "
            "reply must fail here rather than reach a parser as an empty prior")
    return text, str(getattr(resp, "model", "") or model_id)


def ask_vlm(*, system: str, user: str, image: Path | None = None,
            image_b64: str | None = None, max_tokens: int = 2048,
            model: str | None = None) -> tuple[str, str]:
    """Ask the VLM one question; return ``(text, resolved_model_id)``.

    ``image`` (a path) or ``image_b64`` (already-encoded PNG) attaches one
    image, placed BEFORE the text so a prompt that refers back to it reads in
    order.

    The returned id is the one the API answered with, not the one requested:
    aliases resolve to dated snapshots, and an artifact that recorded the alias
    would name a model that never ran.
    """
    model_id = model or vlm_model_from_env()
    provider = provider_for_model(model_id)
    _require_key(provider)
    ask = _ask_anthropic if provider == "anthropic" else _ask_openai
    return ask(system=system, user=user, image=image, image_b64=image_b64,
               max_tokens=max_tokens, model_id=model_id)
