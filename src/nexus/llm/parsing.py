"""Parsing + model-name helpers shared by the nexus LLM clients.

Extracted from `nexus.agents.research.utils` in the 2026-06-02 refactor so
that anything that needs to talk to an LLM (research, future agents, the
provenance writer's `BLOCKED`-status synthesizer, etc.) can reuse them
without depending on the research subpackage.
"""

import re


def clean_json_block(raw: str) -> str:
    """Strip ```json ... ``` markdown fences from an LLM response."""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)
    return raw.strip()


def is_responses_model(model: str) -> bool:
    """Check if model requires OpenAI's Responses API (e.g. gpt-5.x, codex-mini)."""
    return "gpt-5" in model or "codex" in model


def normalize_model_name(model: str) -> tuple[str, str]:
    """Normalize a model string to (aisuite_name, openai_name).

    Example: "openai:gpt-4o" -> ("openai:gpt-4o", "gpt-4o").
    """
    if model.startswith("openai:"):
        return model, model.split(":", 1)[1]
    return f"openai:{model}", model
